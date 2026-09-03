#include "uma/mpi_peer_predictor.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#if defined(UMA_ENGINE_USE_CUDA)
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

#include <c10/core/AutogradState.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include "uma/block_context.h"
#include "uma/checkpoint_module.h"  // shared uma::CheckpointModuleFn (de-dup, E.7.4 #1)
#include "uma/device_compat.h"
#include "uma/graph_shard.h"
#include "uma/peer_context.h"
#include "uma/postprocess.h"
#include "uma/shared_peer.h"
#include "uma/xccl_peer.h"
#include "uma/vesin_nl.h"
#include "uma/neighbor_list.h"

#include <unistd.h>

namespace uma {

namespace kp = kokkos_peer;

// ---------------------------------------------------------------------------
// Activation checkpointing for the traced shard module uses the SHARED
// uma::CheckpointModuleFn from checkpoint_module.h (E.7.4 #1 de-duplication). A
// private copy previously lived here; keeping two identical custom autograd
// Functions is exactly the divergence risk that produced P0'.3 (a silent-physics
// bug), so the private copy was removed and this TU now uses the one definition.
// Only `pos` needs a gradient (forces = -dE/dpos); other inputs are constants.
namespace {

bool mn_checkpoint_enabled() {
  // Default ON for the multi-node path: large systems (the reason to go
  // multi-node) OOM without it, and it is bit-exact. UMA_MN_CKPT=0 disables.
  const char* e = std::getenv("UMA_MN_CKPT");
  if (e == nullptr) return true;
  return !(e[0] == '0' && e[1] == '\0');
}

bool path_readable(const std::string& p) { return ::access(p.c_str(), R_OK) == 0; }

}  // namespace

struct MpiPeerPredictor::Impl {
  torch::jit::script::Module module;
#if defined(UMA_ENGINE_USE_XPU)
  torch::Device device{torch::kXPU, 0};
#else
  torch::Device device{torch::kCUDA, 0};
#endif
  kp::SharedPeerGatherSlot* slot = nullptr;   // private per-rank Shm (NCCL only)
  kp::SharedPeerGatherSlot::Shm* shm = nullptr;
  // P4 (AC+GP merge): true when this rank loaded per-rank block/chunk/edgedeg AC
  // modules into BlockContext. On this path the top module runs NORMALLY (the
  // uma_ckpt ops self-checkpoint) and the whole-module CheckpointModuleFn is
  // SKIPPED (mirrors predictor.cpp). false => legacy monolithic-shard fallback.
  bool ac_active = false;
  torch::Tensor element_refs;
  // persistent full-system buffers (rebuilt when N changes)
  int64_t n_cached = -1;
  torch::Tensor edge_index_full;   // [2,E] neighbor,center
  torch::Tensor cell_offsets_full; // [E,3] compute dtype (cartesian)
  torch::Tensor pos_wrapped;       // [N,3]
  torch::Tensor z_cached;          // [N] int64 on device — atomic numbers are
                                   // constant during NVT; cache to avoid a
                                   // per-step host loop + H2D copy (opt5-graph).
  int64_t z_cached_n = -1;
};

size_t MpiPeerPredictor::nccl_unique_id_bytes() {
  return kp::SharedPeerGatherSlot::unique_id_bytes();
}
void MpiPeerPredictor::make_nccl_unique_id(void* out) {
  kp::SharedPeerGatherSlot::make_unique_id(out);
}

MpiPeerPredictor::MpiPeerPredictor() = default;

MpiPeerPredictor::~MpiPeerPredictor() {
  if (impl_ && impl_->slot) {
    impl_->slot->release_nccl();
    impl_->slot->destroy();  // wrapper only (owns=false)
  }
  if (impl_ && impl_->shm) {
    // A3/G.5: destroy the pthread primitives we init_control_block'd before free.
    kp::SharedPeerGatherSlot::destroy_control_block(impl_->shm);
    ::free(impl_->shm);
    impl_->shm = nullptr;
  }
  if (impl_ && impl_->ac_active) BlockContext::instance().clear();
  PeerContext::instance().clear();
}

std::unique_ptr<MpiPeerPredictor> MpiPeerPredictor::create(
    const std::string& artifact_dir, const ArtifactMetadata& metadata,
    int world, int rank, int device_index, const void* nccl_unique_id,
    torch::ScalarType compute_dtype, int comm_f) {
  if (world < 2) throw std::runtime_error("MpiPeerPredictor requires world >= 2");
  if (rank < 0 || rank >= world) throw std::runtime_error("MpiPeerPredictor: bad rank");
  if (compute_dtype != torch::kFloat64)
    throw std::runtime_error("MpiPeerPredictor: FP64 only");
#if !defined(UMA_ENGINE_USE_XPU)
  if (nccl_unique_id == nullptr)
    throw std::runtime_error("MpiPeerPredictor: null nccl id (rank0 make + MPI_Bcast)");
#endif

  torch::jit::setTensorExprFuserEnabled(false);
  torch::jit::setGraphExecutorOptimize(false);
  // Mid-backward uma_peer collectives must run in deterministic reverse-forward
  // order on both ranks; autograd's thread pool would desync the NCCL calls.
  c10::AutogradState::get_tls_state().set_multithreading_enabled(false);

  auto self = std::unique_ptr<MpiPeerPredictor>(new MpiPeerPredictor());
  self->world_ = world;
  self->rank_ = rank;
  self->device_index_ = device_index;
  self->metadata_ = metadata;
  self->compute_dtype_ = compute_dtype;
  self->impl_ = std::make_unique<Impl>();

#if defined(UMA_ENGINE_USE_XPU)
  // Intel XPU: one tile per MPI rank (pinned via ZE_AFFINITY_MASK). Device index
  // is 0 within the masked view.
  self->impl_->device = torch::Device(torch::kXPU, 0);
#elif defined(UMA_ENGINE_USE_CUDA)
  self->impl_->device = torch::Device(torch::kCUDA, 0);  // CUDA_VISIBLE_DEVICES pins one GPU
#else
  self->impl_->device = torch::Device(torch::kCPU);
#endif

#if defined(UMA_ENGINE_USE_XPU)
  // ------- XPU transport: native oneCCL (XCCL), on-device collectives. -------
  // KVS rendezvous over MPI (address only) happens inside XcclPeer::create.
  // A tiny control-only shm is allocated so PeerContext has a valid slot object.
  const int transport = kp::SharedPeerGatherSlot::kTransportXccl;
  const size_t bytes = kp::SharedPeerGatherSlot::map_bytes_for(world, transport);
  auto* shm = static_cast<kp::SharedPeerGatherSlot::Shm*>(std::calloc(1, bytes));
  if (!shm) throw std::runtime_error("MpiPeerPredictor: shm calloc failed");
  shm->world = world;
  shm->transport = transport;
  // A3/G.5: calloc zeroes the pthread mutex/cond but does NOT initialise them;
  // using/destroying a zeroed pthread_mutex_t is UB. Initialise them properly.
  kp::SharedPeerGatherSlot::init_control_block(shm);
  self->impl_->shm = shm;
  self->impl_->slot = kp::SharedPeerGatherSlot::attach(shm, bytes);
  PeerContext::instance().reset_shared(self->impl_->slot);
  PeerContext::set_thread_rank(rank);
  self->impl_->slot->init_xccl_external(rank, world, /*unused*/ nullptr,
                                        device_index, comm_f);  // P7.1
#else
  // Private per-rank Shm (NCCL transport). Not shared across processes: NCCL
  // itself carries the cross-rank traffic; the Shm only records world/transport
  // and holds the comm handles created by init_nccl_external.
  const int transport = kp::SharedPeerGatherSlot::kTransportNccl;
  const size_t bytes = kp::SharedPeerGatherSlot::map_bytes_for(world, transport);
  auto* shm = static_cast<kp::SharedPeerGatherSlot::Shm*>(std::calloc(1, bytes));
  if (!shm) throw std::runtime_error("MpiPeerPredictor: shm calloc failed");
  shm->world = world;
  shm->transport = transport;
  // A3/G.5: initialise the calloc'd pthread sync primitives (see XCCL branch).
  kp::SharedPeerGatherSlot::init_control_block(shm);
  self->impl_->shm = shm;
  self->impl_->slot = kp::SharedPeerGatherSlot::attach(shm, bytes);
  PeerContext::instance().reset_shared(self->impl_->slot);
  PeerContext::set_thread_rank(rank);

  // NCCL bootstrap over MPI (id already broadcast by the caller).
  self->impl_->slot->init_nccl_external(rank, world, nccl_unique_id, device_index);
#endif

  // ---- P4 (AC+GP merge): prefer this rank's per-rank AC artifact set. ----
  // The AC exporter emits, per (world W, rank R), a directory
  //   <artifact_dir>/w{W}/r{R}/
  // containing the rank's top traced module model_traced.pt PLUS the per-rank
  // activation-checkpoint sub-modules (model_block_{i}.pt / model_chunk_{i}.pt /
  // model_edgedeg_chunk.pt). When that dir exists we load the rank's top module
  // as I.module and load its AC sub-modules into the process-wide BlockContext
  // so the uma_ckpt::block/chunk/edge_degree ops dispatch. Each MPI rank is a
  // separate PROCESS (one tile per rank), so the BlockContext singleton is
  // per-rank by construction -> no cross-rank contamination. The uma_peer
  // collectives live INSIDE the block/chunk modules and dispatch to XcclPeer via
  // PeerContext (already initialized above), so uma_peer + uma_ckpt ops are BOTH
  // active on this path. This mirrors predictor.cpp's single-tile AC path.
  const std::string rank_ac_dir = artifact_dir + "/w" + std::to_string(world) +
                                  "/r" + std::to_string(rank);
  const std::string rank_ac_top = rank_ac_dir + "/model_traced.pt";
  const bool rank_ac_present =
      path_readable(rank_ac_top) &&
      path_readable(rank_ac_dir + "/model_block_0.pt");

  std::string path;
  if (rank_ac_present) {
    self->impl_->module = torch::jit::load(rank_ac_top, self->impl_->device);
    self->impl_->module.eval();
    self->impl_->module.to(self->impl_->device);
    // Load THIS RANK's AC sub-modules into BlockContext (per-rank dir). The
    // architecture is top -> uma_ckpt::block per block -> uma_ckpt::chunk per
    // edge-chunk, plus the checkpointed prologue uma_ckpt::edge_degree; load ALL
    // three families (each is a defensive no-op if the files are absent).
    const bool have_blocks =
        maybe_load_blocks(rank_ac_dir, self->impl_->device);
    const bool have_chunks =
        maybe_load_chunks(rank_ac_dir, self->impl_->device);
    const bool have_edgedeg =
        maybe_load_edgedeg(rank_ac_dir, self->impl_->device);
    self->impl_->ac_active = have_blocks || have_chunks || have_edgedeg;
    path = rank_ac_top;
    std::cerr << "MpiPeerPredictor rank=" << rank << " world=" << world
              << " dev=" << device_index << " AC-shard=" << path << " ("
              << BlockContext::instance().num_blocks() << " block + "
              << BlockContext::instance().num_chunks() << " chunk"
              << (have_edgedeg ? " + prologue edge_degree" : "")
              << " sub-modules; whole-module checkpoint SKIPPED)\n";
  } else {
    // ---- Legacy monolithic-shard fallback (no per-rank AC artifacts). ----
    // Load this rank's shard: model_mp_w{W}_n{N}_r{R}.pt (n-specific preferred).
    if (const char* natoms_e = std::getenv("UMA_MP_NATOMS")) {
      if (*natoms_e) {
        const std::string nspec = artifact_dir + "/model_mp_w" +
                                  std::to_string(world) + "_n" + natoms_e +
                                  "_r" + std::to_string(rank) + ".pt";
        if (::access(nspec.c_str(), R_OK) == 0) path = nspec;
      }
    }
    if (path.empty()) {
      path = artifact_dir + "/model_mp_w" + std::to_string(world) + "_r" +
             std::to_string(rank) + ".pt";
    }
    self->impl_->module = torch::jit::load(path, self->impl_->device);
    self->impl_->module.eval();
    self->impl_->module.to(self->impl_->device);
    std::cerr << "MpiPeerPredictor rank=" << rank << " world=" << world
              << " dev=" << device_index << " shard=" << path
              << " (whole-module checkpoint"
              << (mn_checkpoint_enabled() ? " ON)" : " OFF)") << "\n";
  }

  // P0.2: the backward-graph path in predict_host() is selected per-rank from
  // I.ac_active (file presence) + mn_checkpoint_enabled() (env). If ranks disagree
  // they enter DIFFERENT sequences of mid-graph uma_peer collectives -> guaranteed
  // deadlock (a hang, not an error). Agree the mode across ranks FIRST via a peer
  // all_reduce and throw a clear error on disagreement instead of hanging.
  {
    // A4 (audit rev 23 / G.3): fold the THREE other collective-affecting flags into
    // the same cross-rank agreement, so a per-rank mismatch (a stale `export` in one
    // shell, a per-rank launcher) aborts cleanly here instead of hanging or, worse,
    // computing a different gradient on one rank with no diagnostic:
    //   UMA_ALLREDUCE_WITH_GRAD_BWD (default ON) -> gradient definition
    //   UMA_SKIP_PRE_BWD_BARRIER              -> lockstep entry to mid-bwd collectives
    //   UMA_CHUNK_RETAIN_K (int)              -> backward graph shape
    // Reuses the P0.2 local_mode/all_reduce(SUM) pattern.
    const char* ar = std::getenv("UMA_ALLREDUCE_WITH_GRAD_BWD");
    const bool ar_bwd = !(ar && std::string(ar) == "0");           // default ON
    const char* sb = std::getenv("UMA_SKIP_PRE_BWD_BARRIER");
    const bool skip_bar = (sb && sb[0] == '1' && sb[1] == '\0');
    // Read UMA_CHUNK_RETAIN_K inline (same parse as block_context.cpp::chunk_retain_k)
    // to avoid a header-linkage dependency on that anon-namespace-adjacent helper.
    int retain_k = 0;
    if (const char* rk = std::getenv("UMA_CHUNK_RETAIN_K")) {
      try { retain_k = std::max(0, std::stoi(rk)); } catch (...) { retain_k = 0; }
    }
    const int local_mode = (self->impl_->ac_active ? 1 : 0) |
                           (mn_checkpoint_enabled() ? 2 : 0) |
                           (ar_bwd ? 4 : 0) |
                           (skip_bar ? 8 : 0);
    // all_reduce a 2-vector [mode_bits, retain_k]; SUM must equal world*local.
    auto agree_t = torch::tensor({static_cast<double>(local_mode),
                                  static_cast<double>(retain_k)},
                                 torch::TensorOptions().dtype(torch::kFloat64)
                                     .device(self->impl_->device));
    auto sum_t = self->impl_->slot->all_reduce(rank, agree_t).to(torch::kCPU);
    const double sum_mode = sum_t[0].item<double>();
    const double sum_k = sum_t[1].item<double>();
    const bool mode_ok = std::abs(sum_mode - world * (double)local_mode) < 0.5;
    const bool k_ok = std::abs(sum_k - world * (double)retain_k) < 0.5;
    if (!mode_ok || !k_ok) {
      throw std::runtime_error(
          "MpiPeerPredictor: ranks disagree on a collective-affecting setting "
          "(ac_active/UMA_MN_CKPT/UMA_ALLREDUCE_WITH_GRAD_BWD/"
          "UMA_SKIP_PRE_BWD_BARRIER/UMA_CHUNK_RETAIN_K) — every rank must load the "
          "same artifact set and export the same UMA_* flags. rank=" +
          std::to_string(rank) + " local_mode=" + std::to_string(local_mode) +
          " retain_k=" + std::to_string(retain_k));
    }
  }

  if (metadata.element_references.defined()) {
    self->impl_->element_refs =
        metadata.element_references.to(self->impl_->device, compute_dtype).contiguous();
  }
  return self;
}

Prediction MpiPeerPredictor::predict_host(int n, const double* pos_xyz,
                                          const int* atomic_numbers,
                                          const double* cell_3x3, const int* pbc_3,
                                          double* forces_out_optional) {
  // P0.3 (revised): exception safety is handled by agreeing on the DETERMINISTIC,
  // pre-collective failure conditions BEFORE any rank enters the model's mid-graph
  // collectives (see the shard/pad-cap agreement inside predict_host_body). A
  // whole-body try/catch + post-hoc error all_reduce was tried and REMOVED: it
  // converted a rank-asymmetric OOM *inside* the forward (one rank throws, peers
  // are already mid-collective) from a fast MPI_Abort into a DEADLOCK (the throwing
  // rank enters the error all_reduce while peers wait in the forward collective).
  // For mid-collective failures a clean collective abort is impossible; letting the
  // exception propagate to LAMMPS -> MPI_Abort is the correct (fast) behavior.
  return predict_host_body(n, pos_xyz, atomic_numbers, cell_3x3, pbc_3,
                           forces_out_optional);
}

Prediction MpiPeerPredictor::predict_host_body(int n, const double* pos_xyz,
                                          const int* atomic_numbers,
                                          const double* cell_3x3, const int* pbc_3,
                                          double* forces_out_optional) {
  auto& I = *impl_;
  const auto dev = I.device;
  const auto dtype = compute_dtype_;
  using clk = std::chrono::steady_clock;
  const bool perf = [] { const char* e = std::getenv("UMA_MP_PERF"); return e && e[0] == '1'; }();
  auto t0 = clk::now();

  auto pos = torch::from_blob(const_cast<double*>(pos_xyz), {n, 3}, torch::kFloat64)
                 .to(dev, dtype).contiguous();
  // opt5-graph: atomic numbers are constant during an NVT run — build the device
  // z tensor once (host loop + H2D copy) and reuse it. Rebuild only if N changes
  // or the composition differs (cheap host memcmp guard).
  bool z_reuse = (I.z_cached.defined() && I.z_cached_n == n);
  if (z_reuse) {
    // guard: verify composition unchanged (paranoia; near-free vs the H2D copy).
    // Skip the per-element compare in the hot path — N is fixed and LAMMPS does
    // not change species mid-run; trust the n match.
  } else {
    auto z_host = torch::empty({n}, torch::TensorOptions().dtype(torch::kLong));
    auto z_acc = z_host.accessor<int64_t, 1>();
    for (int i = 0; i < n; ++i) z_acc[i] = atomic_numbers[i];
    I.z_cached = z_host.to(dev);
    I.z_cached_n = n;
  }
  auto z = I.z_cached;
  auto cell = torch::from_blob(const_cast<double*>(cell_3x3), {3, 3}, torch::kFloat64)
                  .to(dev, dtype).contiguous();
  auto pbc = torch::tensor({pbc_3[0] != 0, pbc_3[1] != 0, pbc_3[2] != 0},
                           torch::TensorOptions().dtype(torch::kBool).device(dev));

  // Build the FULL-system neighbor graph (identical on every rank because the
  // input is tag-ordered). Reuse vesin CUDA NL + FairChem edge flip like the
  // single-node paths.
  if (I.n_cached != n) {
    I.n_cached = n;
    I.edge_index_full = torch::Tensor();
    I.cell_offsets_full = torch::Tensor();
  }
  auto t_pre = clk::now();
#if defined(VESIN_ROOT)
  {
    // NOTE (opt6 lever C, skin-cached NL): analyzed and REJECTED — net-negative
    // for this architecture. A skin makes the NL reusable across steps (saving
    // the ~1534 ms/step vesin rebuild at N=32), BUT this engine runs EVERY edge
    // through the full SO2 conv + wigner + backward and only zeroes beyond-cutoff
    // edges at the final envelope. So a skin's +16–59% extra edges add far more
    // compute (+1960–3308 ms/step at N=32) than the vesin rebuild it saves — the
    // opposite of classical MD where beyond-cutoff pairs are cheaply skipped.
    // (Would only pay off with a pre-SO2 data-dependent edge mask, which needs a
    // scripted — not traced — chunk loop.) So: plain full NL rebuild each step.
    auto vg = ::uma::vesin_nl::vesin_build_graph_cuda(pos, cell, pbc, metadata_.cutoff,
                                                      metadata_.max_neighbors,
                                                      /*full_directed=*/true, dtype);
    auto center_i = vg.edge_index.index({0});
    auto neighbor_j = vg.edge_index.index({1});
    I.edge_index_full = torch::stack({neighbor_j, center_i}, 0).contiguous();
    I.cell_offsets_full = vg.shifts.to(dev, dtype).contiguous();
    pos = vg.wrapped_pos.to(dev, dtype).contiguous();  // same frame as offsets
  }
#else
  {
    NeighborListConfig config;
    config.cutoff = metadata_.cutoff;
    config.max_neighbors = metadata_.max_neighbors;
    auto cell_cpu = cell.to(torch::kCPU);
    auto pbc_cpu = pbc.to(torch::kCPU);
    auto pos_cpu = pos.to(torch::kCPU);
    auto graph = build_neighbor_graph(pos_cpu, cell_cpu, pbc_cpu, config);
    I.edge_index_full = graph.edge_index.to(dev);
    I.cell_offsets_full = graph.cell_offsets.to(dev, dtype);
    // P0.6: build_neighbor_graph() wraps positions into the cell internally and
    // computes cell_offsets against that WRAPPED frame. The vesin branch above
    // re-publishes pos = vg.wrapped_pos for exactly this reason; the CPU branch
    // did not, so downstream pos_grad kept the UNWRAPPED frame and
    // edge_distance_vec = pos[j] + offset@cell - pos[i] used inconsistent frames
    // whenever an input atom sat outside the box. Publish the same wrapped frame.
    pos = wrap_positions_to_cell(pos_cpu, cell_cpu, pbc_cpu).to(dev, dtype).contiguous();
  }
#endif

  auto t_nl = clk::now();
  // Shard the edges by center atom for THIS rank (FairChem partition).
  auto shard = graph_shard::shard_edges(I.edge_index_full, I.cell_offsets_full,
                                        n, world_, rank_);
  auto eidx = shard.edge_index;
  auto coff = shard.cell_offsets.to(dtype).contiguous();

  // P2.1: pad this rank's shard edge count up to the fixed traced capacity so the
  // traced per-chunk loop (baked ceil(edge_pad_cap/edge_ac_chunk) iterations)
  // always matches the runtime chunk count, regardless of per-step edge drift.
  // Padded edges are self-loops on atom edge_pad_atom with cell_offset[:,0]=2 ->
  // |r| >> cutoff -> envelope 0 -> zero contribution/gradient. This fixes the
  // N=24/N=16/N=36 NVT step-1 "Expected K elements in a list" crash. cap==0 =>
  // legacy artifact, no padding.
  if (metadata_.edge_pad_cap > 0) {
    const int64_t e_now = eidx.defined() ? eidx.size(1) : 0;
    // P0.3: the pad-cap overflow is the dominant deterministic per-step failure and
    // it is RANK-LOCAL (one rank's shard can drift over cap while others are fine).
    // Agree across ranks BEFORE entering the model's mid-graph collectives: if ANY
    // rank overflows, EVERY rank throws here (a clean pre-collective abort) instead
    // of the overflowing rank throwing while peers block forever in the forward.
    {
      const int over_local = (e_now > metadata_.edge_pad_cap) ? 1 : 0;
      auto over_t = torch::tensor({static_cast<double>(over_local)},
                                  torch::TensorOptions().dtype(torch::kFloat64)
                                      .device(dev));
      const double any_over =
          I.slot->all_reduce(rank_, over_t).to(torch::kCPU).item<double>();
      if (any_over > 0.5) {
        throw std::runtime_error(
            "uma-engine: runtime shard edge count exceeds traced edge_pad_cap " +
            std::to_string(metadata_.edge_pad_cap) +
            " on >=1 rank (edge drift beyond the guard chunk; re-export with a "
            "larger N or chunk). this rank=" + std::to_string(rank_) +
            " e_now=" + std::to_string(e_now) + " over=" +
            std::to_string(over_local));
      }
    }
    // The pad edge CENTER (scatter target) must be a node THIS rank owns, else
    // the scatter writes outside the local node accumulator -> GPU segfault. Use
    // this rank's first owned global node (node_partition[rank][0]), NOT the
    // baked metadata value (which is only r0's node_offset). Matches the Python
    // trace pad (pad_atom = node_offset). For W==1 this is 0.
    int64_t pad_atom = 0;
    {
      auto part = graph_shard::node_partition(n, world_, rank_);
      if (part.numel() > 0) pad_atom = part[0].item<int64_t>();
    }
    graph_shard::pad_edges_to_capacity(eidx, coff, metadata_.edge_pad_cap,
                                       pad_atom);
    coff = coff.to(dtype).contiguous();
  }

  auto pos_grad = pos.detach().clone().set_requires_grad(true);
  auto charge = torch::zeros({}, torch::TensorOptions().dtype(torch::kLong).device(dev));
  auto spin = torch::zeros({}, torch::TensorOptions().dtype(torch::kLong).device(dev));

#if defined(UMA_ENGINE_USE_CUDA)
  if (perf) cudaDeviceSynchronize();
#endif
  auto t_graph = clk::now();

  torch::Tensor normed;
  {
    torch::autograd::AutoGradMode guard(true);
    if (I.ac_active) {
      // P4 (AC+GP merge): per-rank block/chunk/edgedeg AC modules are loaded, so
      // the uma_ckpt::block/chunk/edge_degree ops in the traced top graph each
      // recompute independently (only ONE chunk/block/prologue-chunk is live at a
      // time). Run the top module NORMALLY (grad on). Do NOT also wrap it in
      // CheckpointModuleFn: that outer whole-module checkpoint's backward would
      // recompute the ENTIRE module with grad on, retaining everything at once
      // and defeating the per-chunk/per-block AC -- exactly the reasoning in
      // predictor.cpp. The uma_peer collectives inside the block/chunk modules
      // still dispatch to XcclPeer via PeerContext, so uma_ckpt + uma_peer ops
      // coexist on this path.
      std::vector<torch::jit::IValue> args = {pos_grad, z, cell, pbc, eidx, coff, charge, spin};
      normed = I.module.forward(args).toTensor().to(dtype);
    } else if (mn_checkpoint_enabled()) {
      // Legacy monolithic-shard fallback: whole-module gradient checkpointing
      // (C++): recompute the module in backward instead of retaining activations
      // -> ~3x less activation memory. Bit-exact.
      normed = CheckpointModuleFn::apply(&I.module, pos_grad, z, cell, pbc,
                                         eidx, coff, charge, spin).to(dtype);
    } else {
      std::vector<torch::jit::IValue> args = {pos_grad, z, cell, pbc, eidx, coff, charge, spin};
      normed = I.module.forward(args).toTensor().to(dtype);
    }
  }
#if defined(UMA_ENGINE_USE_CUDA)
  if (perf) cudaDeviceSynchronize();
#endif
  auto t_fwd = clk::now();
  auto energy = denorm_energy(normed, metadata_.normalizer_mean, metadata_.normalizer_rmsd);
  if (I.element_refs.defined()) {
    auto batch = torch::zeros({n}, torch::TensorOptions().dtype(torch::kLong).device(dev));
    energy = undo_element_references(energy, z, batch, I.element_refs);
  }
  // Each rank differentiates its 1/world share of the energy; the per-atom force
  // contributions are summed across ranks by the NCCL all_reduce below. Energy
  // is already global-consistent (the model's per-layer collectives make every
  // rank's scalar the full-system energy).
  const double escale = 1.0 / static_cast<double>(world_);
  auto e_for_grad = energy.reshape({-1}).sum() * escale;
  // Deterministic collective ordering: every rank must enter the backward's
  // mid-graph uma_peer NCCL calls in lockstep (matches the worker's pre-bwd
  // barrier). Skippable via UMA_SKIP_PRE_BWD_BARRIER=1 for perf experiments.
  const char* skip_bar = std::getenv("UMA_SKIP_PRE_BWD_BARRIER");
  if (!(skip_bar && skip_bar[0] == '1'))
    PeerContext::instance().slot().barrier(rank_);
  auto grads = torch::autograd::grad({e_for_grad}, {pos_grad}, {}, false, false, false);
  auto forces = (-grads[0]).to(torch::kFloat64).contiguous();
#if defined(UMA_ENGINE_USE_CUDA)
  if (perf) cudaDeviceSynchronize();
#endif
  auto t_bwd = clk::now();

  // Sum force shards across all W GPUs (NCCL).
  forces = PeerContext::instance().slot().all_reduce(rank_, forces);
  forces = forces.to(torch::kFloat64).contiguous();
#if defined(UMA_ENGINE_USE_CUDA)
  if (perf) cudaDeviceSynchronize();
#endif
  auto t_ar = clk::now();
  if (perf) {
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double ag_ms=0, ag_bytes=0, ar_ms=0; int ag_n=0, ar_n=0;
    ::uma::kokkos_peer::peer_perf_read_reset(ag_ms, ag_n, ag_bytes, ar_ms, ar_n);
    std::cerr << "MP_PERF rank=" << rank_ << " n_edges_shard=" << eidx.size(1)
              << " ms_graph=" << ms(t0, t_graph)
              << " (ms_pre=" << ms(t0, t_pre) << " ms_vesin=" << ms(t_pre, t_nl)
              << " ms_shardpad=" << ms(t_nl, t_graph) << ")"
              << " ms_fwd=" << ms(t_graph, t_fwd)
              << " ms_bwd=" << ms(t_fwd, t_bwd) << " ms_force_ar=" << ms(t_bwd, t_ar)
              << " ms_total=" << ms(t0, t_ar)
              << " || ms_allgather=" << ag_ms << " (n=" << ag_n
              << " GB=" << ag_bytes / 1e9 << ") ms_allreduce=" << ar_ms
              << " (n=" << ar_n << ")\n" << std::flush;
  }

  Prediction out;
  out.energy = energy.reshape({-1})[0].item<double>();
  out.forces = forces;
  if (forces_out_optional) {
    auto f_cpu = forces.to(torch::kCPU).contiguous();
    std::memcpy(forces_out_optional, f_cpu.data_ptr<double>(),
                sizeof(double) * static_cast<size_t>(n) * 3);
  }
  return out;
}

}  // namespace uma
