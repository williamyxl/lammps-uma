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
#include "uma/device_compat.h"
#include "uma/graph_shard.h"
#include "uma/peer_context.h"
#include "uma/postprocess.h"
#include "uma/shared_peer.h"
#include "uma/vesin_nl.h"
#include "uma/neighbor_list.h"

#include <unistd.h>

namespace uma {

namespace kp = kokkos_peer;

// ---------------------------------------------------------------------------
// C++ activation checkpointing for the traced shard module.
//
// The traced TorchScript shard cannot use torch.utils.checkpoint (it does not
// trace). This custom autograd Function reproduces gradient checkpointing at the
// C++ level: forward runs the module WITHOUT retaining the autograd graph (no
// intermediate activations kept), and backward RECOMPUTES the module forward
// with grad enabled to produce the vjp. This trades ~1 extra forward for ~3x
// less activation memory, exactly like torch.utils.checkpoint, and works with
// the opaque traced module. Enable with UMA_MN_CKPT=1.
//
// Only `pos` needs a gradient here (forces = -dE/dpos), so we checkpoint w.r.t.
// pos and treat the other inputs as constants captured by value.
namespace {

struct CheckpointModuleFn : public torch::autograd::Function<CheckpointModuleFn> {
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      torch::jit::script::Module* module,
      torch::Tensor pos,
      torch::Tensor z, torch::Tensor cell, torch::Tensor pbc,
      torch::Tensor eidx, torch::Tensor coff,
      torch::Tensor charge, torch::Tensor spin) {
    ctx->saved_data["module"] = reinterpret_cast<int64_t>(module);
    // Save inputs (NOT activations) for the backward recompute.
    ctx->save_for_backward({pos, z, cell, pbc, eidx, coff, charge, spin});
    torch::NoGradGuard no_grad;  // do not retain the forward graph
    std::vector<torch::jit::IValue> args = {pos, z, cell, pbc, eidx, coff, charge, spin};
    return module->forward(args).toTensor();
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto* module = reinterpret_cast<torch::jit::script::Module*>(
        ctx->saved_data["module"].toInt());
    auto saved = ctx->get_saved_variables();
    auto pos = saved[0].detach().set_requires_grad(true);
    auto z = saved[1], cell = saved[2], pbc = saved[3];
    auto eidx = saved[4], coff = saved[5], charge = saved[6], spin = saved[7];
    torch::Tensor normed;
    {
      torch::autograd::AutoGradMode grad_on(true);
      std::vector<torch::jit::IValue> args = {pos, z, cell, pbc, eidx, coff, charge, spin};
      normed = module->forward(args).toTensor();
    }
    auto grads = torch::autograd::grad({normed}, {pos}, {grad_outputs[0]},
                                       /*retain_graph=*/false,
                                       /*create_graph=*/false,
                                       /*allow_unused=*/true);
    torch::Tensor gpos = grads[0].defined()
                             ? grads[0]
                             : torch::zeros_like(pos);
    // Only pos (arg index 1) gets a gradient; module ptr + others are null.
    return {torch::Tensor(), gpos, torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor()};
  }
};

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
    ::free(impl_->shm);
    impl_->shm = nullptr;
  }
  if (impl_ && impl_->ac_active) BlockContext::instance().clear();
  PeerContext::instance().clear();
}

std::unique_ptr<MpiPeerPredictor> MpiPeerPredictor::create(
    const std::string& artifact_dir, const ArtifactMetadata& metadata,
    int world, int rank, int device_index, const void* nccl_unique_id,
    torch::ScalarType compute_dtype) {
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
  register_uma_peer_ops();

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
  self->impl_->shm = shm;
  self->impl_->slot = kp::SharedPeerGatherSlot::attach(shm, bytes);
  PeerContext::instance().reset_shared(self->impl_->slot);
  PeerContext::set_thread_rank(rank);
  self->impl_->slot->init_xccl_external(rank, world, /*unused*/ nullptr,
                                        device_index);
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
  auto& I = *impl_;
  const auto dev = I.device;
  const auto dtype = compute_dtype_;
  using clk = std::chrono::steady_clock;
  const bool perf = [] { const char* e = std::getenv("UMA_MP_PERF"); return e && e[0] == '1'; }();
  auto t0 = clk::now();

  auto pos = torch::from_blob(const_cast<double*>(pos_xyz), {n, 3}, torch::kFloat64)
                 .to(dev, dtype).contiguous();
  auto z = torch::empty({n}, torch::TensorOptions().dtype(torch::kLong));
  auto z_acc = z.accessor<int64_t, 1>();
  for (int i = 0; i < n; ++i) z_acc[i] = atomic_numbers[i];
  z = z.to(dev);
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
#if defined(VESIN_ROOT)
  {
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
    auto graph = build_neighbor_graph(pos.to(torch::kCPU), cell.to(torch::kCPU),
                                      pbc.to(torch::kCPU), config);
    I.edge_index_full = graph.edge_index.to(dev);
    I.cell_offsets_full = graph.cell_offsets.to(dev, dtype);
  }
#endif

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
    if (e_now > metadata_.edge_pad_cap) {
      throw std::runtime_error(
          "uma-engine: runtime shard edge count " + std::to_string(e_now) +
          " exceeds traced edge_pad_cap " +
          std::to_string(metadata_.edge_pad_cap) +
          " (edge drift beyond the guard chunk; re-export with a larger N or "
          "chunk). rank=" + std::to_string(rank_));
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
    std::cerr << "MP_PERF rank=" << rank_ << " n_edges_shard=" << eidx.size(1)
              << " ms_graph=" << ms(t0, t_graph) << " ms_fwd=" << ms(t_graph, t_fwd)
              << " ms_bwd=" << ms(t_fwd, t_bwd) << " ms_force_ar=" << ms(t_bwd, t_ar)
              << " ms_total=" << ms(t0, t_ar) << "\n" << std::flush;
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
