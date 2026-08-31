#include "uma/predictor.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include "uma/block_context.h"
#include "uma/checkpoint_module.h"
#include "uma/device_compat.h"
#include "uma/graph_parallel.h"
#include "uma/neighbor_list.h"
#include "uma/postprocess.h"
#include "uma/graph_shard.h"
#include "uma/vesin_nl.h"

namespace {

torch::Device resolve_device(torch::Device device) {
  torch::Device resolved = uma::resolve_device_compat(device);
  if (!resolved.is_cuda() && !resolved.is_xpu() && !device.is_cpu()) {
    std::cerr << "Warning: requested accelerator unavailable; falling back to CPU\n";
  }
  return resolved;
}

// Borrowed from graspa-mlip fairchem Predictor: avoid TorchScript TE hang.
void disable_torchscript_texpr_once() {
  static const bool done = [] {
    torch::jit::setTensorExprFuserEnabled(false);
    torch::jit::setGraphExecutorOptimize(false);
    return true;
  }();
  (void)done;
}

}  // namespace

namespace uma {

Predictor::Predictor(torch::jit::script::Module module, torch::Device device,
                     ArtifactMetadata metadata, int num_devices)
    : module_(std::move(module)),
      has_traced_module_(true),
      device_(device),
      metadata_(std::move(metadata)),
      compute_dtype_(metadata_.compute_dtype),
      num_devices_(num_devices) {
  disable_torchscript_texpr_once();
  module_.eval();
  module_.to(device_);
  if (metadata_.element_references.defined()) {
    element_refs_ =
        metadata_.element_references.to(device_, compute_dtype_).contiguous();
  }
  charge_ = torch::zeros({}, torch::TensorOptions().dtype(torch::kLong).device(device_));
  spin_ = torch::zeros({}, torch::TensorOptions().dtype(torch::kLong).device(device_));
}

Predictor::Predictor(std::unique_ptr<GraphParallelRuntime> gp, torch::Device device,
                     ArtifactMetadata metadata, int num_devices)
    : has_traced_module_(false),
      gp_(std::move(gp)),
      device_(device),
      metadata_(std::move(metadata)),
      compute_dtype_(metadata_.compute_dtype),
      num_devices_(num_devices) {
  disable_torchscript_texpr_once();
  if (metadata_.element_references.defined()) {
    // Denorm/refs happen inside FairChem for the GP path; keep refs for API symmetry.
    element_refs_ =
        metadata_.element_references.to(torch::kCPU, compute_dtype_).contiguous();
  }
}

Predictor::~Predictor() {
  // P0'.4: the single-tile Predictor populates the process-wide BlockContext
  // singleton (block/chunk/edge_degree modules) at construction but nothing
  // cleared it, so a redefinition or a second predictor left stale module
  // handles live. Clear on teardown. (~MpiPeerPredictor clears it for the GP
  // path; this covers the single-tile path.) Only the live (not moved-from)
  // object clears: from_artifact() loads the blocks then RETURNS BY VALUE, so the
  // moved-from temporary must NOT wipe the blocks the returned object needs.
  if (owns_block_context_) BlockContext::instance().clear();
}

Predictor::Predictor(Predictor&& other) noexcept
    : module_(std::move(other.module_)),
      has_traced_module_(other.has_traced_module_),
      owns_block_context_(other.owns_block_context_),
      gp_(std::move(other.gp_)),
      device_(other.device_),
      metadata_(std::move(other.metadata_)),
      compute_dtype_(other.compute_dtype_),
      num_devices_(other.num_devices_),
      n_cached_(other.n_cached_),
      pos_(std::move(other.pos_)),
      atomic_numbers_(std::move(other.atomic_numbers_)),
      cell_(std::move(other.cell_)),
      pbc_(std::move(other.pbc_)),
      edge_index_(std::move(other.edge_index_)),
      cell_offsets_(std::move(other.cell_offsets_)),
      charge_(std::move(other.charge_)),
      spin_(std::move(other.spin_)),
      batch_(std::move(other.batch_)),
      element_refs_(std::move(other.element_refs_)) {
  // Only this object now owns the BlockContext lifetime; the moved-from source
  // must not clear it on destruction.
  other.owns_block_context_ = false;
}

Predictor& Predictor::operator=(Predictor&& other) noexcept {
  if (this == &other) return *this;
  module_ = std::move(other.module_);
  has_traced_module_ = other.has_traced_module_;
  owns_block_context_ = other.owns_block_context_;
  gp_ = std::move(other.gp_);
  device_ = other.device_;
  metadata_ = std::move(other.metadata_);
  compute_dtype_ = other.compute_dtype_;
  num_devices_ = other.num_devices_;
  n_cached_ = other.n_cached_;
  pos_ = std::move(other.pos_);
  atomic_numbers_ = std::move(other.atomic_numbers_);
  cell_ = std::move(other.cell_);
  pbc_ = std::move(other.pbc_);
  edge_index_ = std::move(other.edge_index_);
  cell_offsets_ = std::move(other.cell_offsets_);
  charge_ = std::move(other.charge_);
  spin_ = std::move(other.spin_);
  batch_ = std::move(other.batch_);
  element_refs_ = std::move(other.element_refs_);
  other.owns_block_context_ = false;
  return *this;
}

Predictor Predictor::from_artifact(const std::string& artifact_dir,
                                   torch::Device device, int num_devices) {
  if (num_devices < 1) {
    throw std::runtime_error("from_artifact: num_devices must be >= 1");
  }
  const std::string metadata_path = artifact_dir + "/metadata.json";
  auto metadata = load_artifact_metadata(metadata_path);

  // Activation checkpointing (eager only; cannot be traced). UMA_EAGER_CKPT=1
  // routes single-GPU through the eager FairChem worker with checkpointing on,
  // trading ~1.33x step time for ~3x less activation memory (much bigger boxes).
  bool eager_ckpt = false;
  if (const char* e = std::getenv("UMA_EAGER_CKPT"))
    eager_ckpt = (e[0] == '1' && e[1] == '\0');

  if (num_devices > 1 || eager_ckpt) {
    // Same-node GP (num_devices>1) via GraphParallelRuntime, OR single-GPU eager
    // checkpointing (num_devices==1 + eager_ckpt). Default GP = C++ LibTorch MP
    // when model_mp_wN_r*.pt exists; Python only with UMA_PYTHON_GP_WORKER=1 or
    // the eager-checkpointing path.
    auto gp = GraphParallelRuntime::create(artifact_dir, metadata, num_devices,
                                           metadata.compute_dtype, eager_ckpt);
    return Predictor(std::move(gp), torch::Device(torch::kCPU), std::move(metadata),
                     num_devices);
  }

  device = resolve_device(device);
  const std::string traced_path = artifact_dir + "/model_traced.pt";
  auto module = torch::jit::load(traced_path, device);
  // Per-block traceable AC: if the artifact ships model_block_{i}.pt, the traced
  // top graph calls uma_ckpt::block(idx, ...). Load the per-block sub-modules
  // into the process-wide BlockContext BEFORE any forward so the op can dispatch.
  // Defensive: legacy artifacts without block files leave the monolithic path.
  // Option (j) PRIMARY path: per-CHUNK AC. If the artifact ships
  // model_chunk_{i}.pt, the traced block loop calls uma_ckpt::chunk(block_idx,
  // ...) per edge-chunk; load the chunk modules into the process-wide
  // BlockContext BEFORE any forward so the op can dispatch. Fall back to the
  // per-block path when only model_block_{i}.pt exists (legacy artifacts leave
  // the monolithic path).
  // Option (j) architecture: top model_traced.pt calls uma_ckpt::block per block
  // -> each block module calls uma_ckpt::chunk per edge-chunk. BOTH the block
  // modules (for uma_ckpt::block) AND the chunk modules (for uma_ckpt::chunk)
  // must be loaded. (Legacy per-block-only artifacts have no chunk modules;
  // legacy monolithic have neither.)
  const bool have_blocks = maybe_load_blocks(artifact_dir, device);
  const bool have_chunks = maybe_load_chunks(artifact_dir, device);
  // Option (P1-b): the PROLOGUE edge_degree_embedding is the last un-checkpointed
  // full-edge transient (12.82 GiB at N=18). If the artifact ships
  // model_edgedeg_chunk.pt, the traced prologue loop calls uma_ckpt::edge_degree
  // per edge-chunk; load the single edge_degree module into the process-wide
  // BlockContext BEFORE any forward so the op can dispatch. Defensive no-op when
  // absent (keeps the un-checkpointed full-edge prologue path).
  const bool have_edgedeg = maybe_load_edgedeg(artifact_dir, device);
  if (have_blocks || have_chunks || have_edgedeg) {
    std::cerr << "uma-engine: loaded " << BlockContext::instance().num_blocks()
              << " block + " << BlockContext::instance().num_chunks()
              << " chunk sub-modules (AC"
              << (have_chunks ? ", per-chunk option j" : ", per-block")
              << (have_edgedeg ? ", +prologue edge_degree P1-b" : "") << ")\n";
  }
  return Predictor(std::move(module), device, std::move(metadata), /*num_devices=*/1);
}

void Predictor::set_compute_dtype(torch::ScalarType dtype) {
  if (dtype != torch::kFloat32 && dtype != torch::kFloat64) {
    throw std::runtime_error("set_compute_dtype: only float32 and float64 are supported");
  }
  if (dtype != metadata_.compute_dtype) {
    throw std::runtime_error(
        "set_compute_dtype: requested dtype does not match TorchScript artifact "
        "(re-export with matching --dtype, or use the matching artifact)");
  }
  compute_dtype_ = dtype;
  n_cached_ = -1;
  if (gp_) {
    // Worker already initialized with artifact dtype at create(); dtype must match.
    return;
  }
  if (metadata_.element_references.defined()) {
    element_refs_ =
        metadata_.element_references.to(device_, compute_dtype_).contiguous();
  }
}

void Predictor::ensure_buffers(int64_t n) {
  if (n_cached_ == n && pos_.defined() && pos_.device() == device_ &&
      pos_.scalar_type() == compute_dtype_) {
    return;
  }
  const auto dtype = compute_dtype_;
  auto opts_f = torch::TensorOptions().dtype(dtype).device(device_);
  auto opts_l = torch::TensorOptions().dtype(torch::kLong).device(device_);
  auto opts_b = torch::TensorOptions().dtype(torch::kBool).device(device_);

  pos_ = torch::empty({n, 3}, opts_f);
  atomic_numbers_ = torch::empty({n}, opts_l);
  cell_ = torch::empty({3, 3}, opts_f);
  pbc_ = torch::empty({3}, opts_b);
  batch_ = torch::zeros({n}, opts_l);
  n_cached_ = n;
  edge_index_ = torch::Tensor();
  cell_offsets_ = torch::Tensor();
}

void Predictor::rebuild_neighbors() {
#if defined(VESIN_ROOT)
  if (device_.is_cuda() && torch::cuda::is_available() && pos_.is_cuda()) {
    // graspa-mlip torch_uma.h path: vesin CUDA NL + FairChem edge flip.
    auto vg = vesin_nl::vesin_build_graph_cuda(
        pos_, cell_, pbc_, metadata_.cutoff, metadata_.max_neighbors,
        /*full_directed=*/true, compute_dtype_);
    // vesin: row0=center, row1=neighbor; FairChem/UMA: row0=neighbor, row1=center.
    auto center_i = vg.edge_index.index({0});
    auto neighbor_j = vg.edge_index.index({1});
    edge_index_ = torch::stack({neighbor_j, center_i}, 0).contiguous();
    cell_offsets_ = vg.shifts.to(device_, compute_dtype_).contiguous();
    return;
  }
#endif
  NeighborListConfig config;
  config.cutoff = metadata_.cutoff;
  config.max_neighbors = metadata_.max_neighbors;
  auto graph = build_neighbor_graph(pos_.to(torch::kCPU), cell_.to(torch::kCPU),
                                    pbc_.to(torch::kCPU), config);
  edge_index_ = graph.edge_index.to(device_);
  cell_offsets_ = graph.cell_offsets.to(device_, compute_dtype_);
}

int64_t Predictor::stage_inputs(const torch::Tensor& pos,
                                const torch::Tensor& atomic_numbers,
                                const torch::Tensor& cell,
                                const torch::Tensor& pbc, int64_t charge,
                                int64_t spin) {
  if (!pos.defined() || pos.dim() != 2 || pos.size(1) != 3) {
    throw std::runtime_error("pos must be [N,3]");
  }
  const int64_t n = pos.size(0);
  const auto dtype = compute_dtype_;
  ensure_buffers(n);

  pos_.copy_(pos.to(device_, dtype));
  atomic_numbers_.copy_(atomic_numbers.to(device_, torch::kLong));

  auto cell_in = cell.to(device_, dtype);
  if (cell_in.dim() == 3) {
    cell_in = cell_in.squeeze(0);
  }
  cell_.copy_(cell_in);

  auto pbc_in = pbc.to(device_, torch::kBool);
  if (pbc_in.dim() == 2) {
    pbc_in = pbc_in.squeeze(0);
  }
  pbc_.copy_(pbc_in);

  charge_.fill_(charge);
  spin_.fill_(spin);
  return n;
}

Prediction Predictor::predict(const torch::Tensor& pos,
                              const torch::Tensor& atomic_numbers,
                              const torch::Tensor& cell,
                              const torch::Tensor& pbc, int64_t charge,
                              int64_t spin) {
  if (gp_) {
    return gp_->predict(pos, atomic_numbers, cell, pbc, charge, spin);
  }
  if (!has_traced_module_) {
    throw std::runtime_error("Predictor: no traced module and no GP runtime");
  }
  stage_inputs(pos, atomic_numbers, cell, pbc, charge, spin);

  // FairChem AtomicData wraps into the cell; model + edge offsets share that frame.
  pos_.copy_(wrap_positions_to_cell(pos_, cell_, pbc_));

  rebuild_neighbors();

  return predict_body();
}

Prediction Predictor::predict_extgraph(const torch::Tensor& pos,
                                       const torch::Tensor& atomic_numbers,
                                       const torch::Tensor& cell,
                                       const torch::Tensor& pbc,
                                       const torch::Tensor& edge_index,
                                       const torch::Tensor& cell_offsets,
                                       int64_t charge, int64_t spin) {
  if (gp_) {
    throw std::runtime_error(
        "predict_extgraph: external-graph path is single-tile Predictor only "
        "(GP runtime not yet supported)");
  }
  if (!has_traced_module_) {
    throw std::runtime_error("Predictor: no traced module and no GP runtime");
  }
  stage_inputs(pos, atomic_numbers, cell, pbc, charge, spin);

  // Caller-supplied neighbor graph. Edge convention MUST match rebuild_neighbors
  // (predictor.cpp:197): row0=neighbor j, row1=center i.
  if (!edge_index.defined() || edge_index.dim() != 2 || edge_index.size(0) != 2) {
    throw std::runtime_error("predict_extgraph: edge_index must be [2,E]");
  }
  if (!cell_offsets.defined() || cell_offsets.dim() != 2 ||
      cell_offsets.size(1) != 3) {
    throw std::runtime_error("predict_extgraph: cell_offsets must be [E,3]");
  }
  if (cell_offsets.size(0) != edge_index.size(1)) {
    throw std::runtime_error(
        "predict_extgraph: edge_index and cell_offsets edge count mismatch");
  }
  edge_index_ = edge_index.to(device_, torch::kLong).contiguous();
  cell_offsets_ = cell_offsets.to(device_, compute_dtype_).contiguous();

  // Intentionally do NOT call wrap_positions_to_cell here: the supplied offsets
  // already encode periodicity against the caller's (unwrapped) coordinates, and
  // edge_distance_vec = pos[j] + offset @ cell - pos[i] is translation-invariant.
  return predict_body();
}

Prediction Predictor::predict_body() {
  const auto dtype = compute_dtype_;
  // Clone so the persistent buffer is not marked requires_grad.
  auto pos_grad = pos_.detach().clone().set_requires_grad(true);

  // P0'.1 step 2: virial via the CELL + POSITION gradients (NOT a strain leaf).
  // The traced module already takes `pos` and `cell` as inputs; forces are
  // dE/dpos. For a graph potential with edge_vec = pos_j + S @ cell - pos_i, the
  // exact global virial (LAMMPS convention W, energy units) is
  //   W_ab = -( sum_i pos_i,a (dE/dpos_i)_b + sum_k cell_k,a (dE/dcell_k)_b )
  // symmetrized. This differentiates only EXISTING traced inputs (same as forces),
  // avoiding the matmul-derived strain leaf that segfaulted on XPU (jobs
  // 8792138/8792362). cell_grad is a leaf clone of cell_ so we can read dE/dcell.
  torch::Tensor cell_used = cell_;
  torch::Tensor cell_grad;
  if (want_virial_) {
    // The whole-module CheckpointModuleFn custom autograd Function only tracks
    // `pos`; it cannot yield dE/dcell. Require a non-checkpointed run for stress.
    const bool ckpt_active =
        BlockContext::instance().num_chunks() > 0 ||
        BlockContext::instance().num_blocks() > 0 ||
        BlockContext::instance().edgedeg_loaded() || checkpoint_enabled();
    TORCH_CHECK(!ckpt_active,
                "uma-engine: virial (UMA_COMPUTE_VIRIAL=1) requires a "
                "non-checkpointed run (plain artifact + UMA_CKPT=0).");
    cell_grad = cell_.detach().clone().set_requires_grad(true);
    cell_used = cell_grad;
  }
  torch::Tensor pos_used = pos_grad;

  // P2.1: pad the edge count up to the fixed traced capacity so the traced
  // per-chunk loop count matches the runtime edge count (fixes the N-specific
  // chunk-drift crash). Self-loops beyond cutoff -> zero contribution. cap==0 =>
  // legacy artifact, no padding.
  auto edge_index_run = edge_index_;
  auto cell_offsets_run = cell_offsets_;
  if (metadata_.edge_pad_cap > 0) {
    const int64_t e_now =
        edge_index_run.defined() ? edge_index_run.size(1) : 0;
    if (e_now > metadata_.edge_pad_cap) {
      throw std::runtime_error(
          "uma-engine: runtime edge count " + std::to_string(e_now) +
          " exceeds traced edge_pad_cap " +
          std::to_string(metadata_.edge_pad_cap) +
          " (edge drift beyond the guard chunk; re-export with a larger N or "
          "chunk).");
    }
    graph_shard::pad_edges_to_capacity(edge_index_run, cell_offsets_run,
                                       metadata_.edge_pad_cap,
                                       metadata_.edge_pad_atom);
    cell_offsets_run = cell_offsets_run.to(dtype).contiguous();
  }

  std::vector<torch::jit::IValue> args = {pos_used,
                                          atomic_numbers_,
                                          cell_used,
                                          pbc_,
                                          edge_index_run,
                                          cell_offsets_run,
                                          charge_,
                                          spin_};

  torch::Tensor normed_raw;
  {
    torch::autograd::AutoGradMode grad_guard(true);
    const bool per_chunk_ac = BlockContext::instance().num_chunks() > 0;
    const bool per_block_ac = BlockContext::instance().num_blocks() > 0;
    const bool prologue_ac = BlockContext::instance().edgedeg_loaded();
    if (per_chunk_ac || per_block_ac || prologue_ac) {
      // PER-CHUNK (option j) / PER-BLOCK / PROLOGUE-edge_degree (P1-b) activation
      // checkpointing is active (uma_ckpt::chunk / uma_ckpt::block /
      // uma_ckpt::edge_degree ops in the graph each recompute independently). Run
      // the top module NORMALLY (grad on) so only ONE chunk (or block, or
      // prologue edge-chunk) is live at a time. Do NOT also wrap the whole module
      // in CheckpointModuleFn: that outer checkpoint's backward recomputes the
      // ENTIRE module with grad on, retaining everything at once and defeating
      // per-chunk/per-block/prologue AC.
      normed_raw = module_.forward(args).toTensor();
    } else if (checkpoint_enabled()) {
      // Whole-module C++ activation checkpointing (no per-block modules):
      // recompute forward in backward instead of retaining activations.
      // P0'.3: pass the P2.1-PADDED edge_index_run/cell_offsets_run (not the raw
      // members) so a padded artifact taking this branch does not re-enter the
      // chunk-count drift crash. When edge_pad_cap==0 these equal the members.
      // P0'.1 step 2: the strain leaf cannot thread through this custom-Function
      // checkpoint (it captures pos/cell by value), so virial is unsupported here.
      TORCH_CHECK(!want_virial_,
                  "uma-engine: virial (stress) is not supported with whole-module "
                  "checkpointing (UMA_CKPT=1). Disable it to compute the virial.");
      if (metadata_.edge_pad_cap > 0) {
        TORCH_CHECK(edge_index_run.size(1) == metadata_.edge_pad_cap,
                    "uma-engine P0'.3: padded edge count ",
                    edge_index_run.size(1), " != edge_pad_cap ",
                    metadata_.edge_pad_cap);
      }
      normed_raw = CheckpointModuleFn::apply(&module_, pos_grad, atomic_numbers_,
                                             cell_, pbc_, edge_index_run,
                                             cell_offsets_run, charge_, spin_);
    } else {
      normed_raw = module_.forward(args).toTensor();
    }
  }
  auto normed = normed_raw.to(dtype);
  auto energy = denorm_energy(normed, metadata_.normalizer_mean,
                              metadata_.normalizer_rmsd);
  if (element_refs_.defined()) {
    energy = undo_element_references(energy, atomic_numbers_, batch_,
                                     element_refs_);
  }

  std::vector<torch::Tensor> grad_inputs = {pos_grad};
  if (want_virial_) grad_inputs.push_back(cell_grad);
  auto grads = torch::autograd::grad({energy.sum()}, grad_inputs,
                                     /*grad_outputs=*/{},
                                     /*retain_graph=*/false,
                                     /*create_graph=*/false,
                                     /*allow_unused=*/want_virial_);
  auto forces = (-grads[0]).to(torch::kFloat64).contiguous();

  Prediction out;
  out.energy = energy.reshape({-1})[0].item<double>();
  out.forces = forces;
  if (want_virial_) {
    // W_ab = -( sum_i pos_i,a (dE/dpos_i)_b + sum_k cell_k,a (dE/dcell_k)_b ),
    // symmetrized. dE/dpos = grads[0] (== -forces); dE/dcell = grads[1].
    auto dE_dpos = grads[0].to(torch::kFloat64);          // [N,3]
    auto pos64 = pos_grad.detach().to(torch::kFloat64);   // [N,3]
    // sum_i pos_i,a * dE/dpos_i,b  ->  [3,3] = pos^T @ dE_dpos
    auto Wp = torch::matmul(pos64.t(), dE_dpos);          // [3,3]
    auto W = Wp;
    if (grads.size() > 1 && grads[1].defined()) {
      auto dE_dcell = grads[1].to(torch::kFloat64);       // [3,3]
      auto cell64 = cell_grad.detach().to(torch::kFloat64);
      W = W + torch::matmul(cell64.t(), dE_dcell);        // + sum_k cell_k,a dE/dcell_k,b
    }
    W = (-0.5) * (W + W.t());                             // symmetrize, sign
    auto Wc = W.to(torch::kCPU).contiguous();
    auto a = Wc.accessor<double, 2>();
    out.has_virial = true;
    out.virial[0] = a[0][0];  // xx
    out.virial[1] = a[1][1];  // yy
    out.virial[2] = a[2][2];  // zz
    out.virial[3] = a[0][1];  // xy
    out.virial[4] = a[0][2];  // xz
    out.virial[5] = a[1][2];  // yz
  }
  return out;
}

Prediction Predictor::predict_host(int n, const float* pos_xyz,
                                   const int* atomic_numbers,
                                   const double* cell_3x3, const int* pbc_3,
                                   double* forces_out_optional) {
  if (gp_) {
    // Promote host FP32 positions to FP64 for the GP worker protocol.
    std::vector<double> pos_d(static_cast<size_t>(n) * 3);
    for (int i = 0; i < n * 3; ++i) pos_d[static_cast<size_t>(i)] = pos_xyz[i];
    return gp_->predict_host(n, pos_d.data(), atomic_numbers, cell_3x3, pbc_3,
                             forces_out_optional);
  }
  auto pos = torch::from_blob(const_cast<float*>(pos_xyz), {n, 3},
                              torch::kFloat32)
                 .clone();
  auto z = torch::empty({n}, torch::kLong);
  auto z_acc = z.accessor<int64_t, 1>();
  for (int i = 0; i < n; ++i) {
    z_acc[i] = atomic_numbers[i];
  }
  auto cell = torch::empty({3, 3}, torch::kFloat64);
  auto cell_acc = cell.accessor<double, 2>();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      cell_acc[i][j] = cell_3x3[3 * i + j];
    }
  }
  auto pbc = torch::tensor({pbc_3[0] != 0, pbc_3[1] != 0, pbc_3[2] != 0},
                           torch::kBool);

  auto pred = predict(pos, z, cell, pbc, 0, 0);
  if (forces_out_optional) {
    auto f_cpu = pred.forces.to(torch::kCPU).contiguous();
    std::memcpy(forces_out_optional, f_cpu.data_ptr<double>(),
                sizeof(double) * static_cast<size_t>(n) * 3);
  }
  return pred;
}

Prediction Predictor::predict_host(int n, const double* pos_xyz,
                                   const int* atomic_numbers,
                                   const double* cell_3x3, const int* pbc_3,
                                   double* forces_out_optional) {
  if (gp_) {
    return gp_->predict_host(n, pos_xyz, atomic_numbers, cell_3x3, pbc_3,
                             forces_out_optional);
  }
  auto pos = torch::from_blob(const_cast<double*>(pos_xyz), {n, 3},
                              torch::kFloat64)
                 .clone();
  auto z = torch::empty({n}, torch::kLong);
  auto z_acc = z.accessor<int64_t, 1>();
  for (int i = 0; i < n; ++i) {
    z_acc[i] = atomic_numbers[i];
  }
  auto cell = torch::empty({3, 3}, torch::kFloat64);
  auto cell_acc = cell.accessor<double, 2>();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      cell_acc[i][j] = cell_3x3[3 * i + j];
    }
  }
  auto pbc = torch::tensor({pbc_3[0] != 0, pbc_3[1] != 0, pbc_3[2] != 0},
                           torch::kBool);

  auto pred = predict(pos, z, cell, pbc, 0, 0);
  if (forces_out_optional) {
    auto f_cpu = pred.forces.to(torch::kCPU).contiguous();
    std::memcpy(forces_out_optional, f_cpu.data_ptr<double>(),
                sizeof(double) * static_cast<size_t>(n) * 3);
  }
  return pred;
}

Prediction Predictor::predict_host_extgraph(
    int n, const double* pos_xyz, const int* atomic_numbers, const double* cell9,
    const int* pbc3, int64_t n_edges, const int64_t* edge_index_2E,
    const double* cell_offsets_E3, double* forces_out) {
  if (gp_) {
    throw std::runtime_error(
        "predict_host_extgraph: external-graph path is single-tile Predictor "
        "only (GP runtime not yet supported)");
  }
  // Build pos/z/cell/pbc tensors from raw pointers (mirrors predict_host FP64).
  auto pos = torch::from_blob(const_cast<double*>(pos_xyz), {n, 3},
                              torch::kFloat64)
                 .clone();
  auto z = torch::empty({n}, torch::kLong);
  auto z_acc = z.accessor<int64_t, 1>();
  for (int i = 0; i < n; ++i) {
    z_acc[i] = atomic_numbers[i];
  }
  auto cell = torch::empty({3, 3}, torch::kFloat64);
  auto cell_acc = cell.accessor<double, 2>();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      cell_acc[i][j] = cell9[3 * i + j];
    }
  }
  auto pbc = torch::tensor({pbc3[0] != 0, pbc3[1] != 0, pbc3[2] != 0},
                           torch::kBool);

  // Caller-supplied neighbor graph. Edge convention: row0=neighbor, row1=center
  // (SAME as rebuild_neighbors output, predictor.cpp:197). Integer periodic
  // shifts are supplied as double in cell_offsets_E3 [n_edges,3].
  auto edge_index = torch::from_blob(const_cast<int64_t*>(edge_index_2E),
                                     {2, n_edges}, torch::kInt64)
                        .clone();
  auto cell_offsets = torch::from_blob(const_cast<double*>(cell_offsets_E3),
                                       {n_edges, 3}, torch::kFloat64)
                          .clone();

  auto pred = predict_extgraph(pos, z, cell, pbc, edge_index, cell_offsets, 0, 0);
  if (forces_out) {
    auto f_cpu = pred.forces.to(torch::kCPU).contiguous();
    std::memcpy(forces_out, f_cpu.data_ptr<double>(),
                sizeof(double) * static_cast<size_t>(n) * 3);
  }
  return pred;
}

/* ---------------------------------------------------------------------- */
/* DD k=4: per-atom energy + owned-only backprop.                         */
/* ---------------------------------------------------------------------- */

Prediction Predictor::predict_host_extgraph_dd(
    int n, int nlocal, const double* pos_xyz, const int* atomic_numbers,
    const double* cell9, const int* pbc3, int64_t n_edges,
    const int64_t* edge_index_2E, const double* cell_offsets_E3,
    double* energy_out, double* forces_out) {
  if (gp_)
    throw std::runtime_error("predict_host_extgraph_dd: single-tile only");
  if (!has_traced_module_)
    throw std::runtime_error("predict_host_extgraph_dd: no traced module");
  if (nlocal < 0 || nlocal > n)
    throw std::runtime_error("predict_host_extgraph_dd: bad nlocal");

  // Stage inputs (mirror predict_host_extgraph), then set the external graph.
  auto pos = torch::from_blob(const_cast<double*>(pos_xyz), {n, 3},
                              torch::kFloat64).clone();
  auto z = torch::empty({n}, torch::kLong);
  auto z_acc = z.accessor<int64_t, 1>();
  for (int i = 0; i < n; ++i) z_acc[i] = atomic_numbers[i];
  auto cell = torch::empty({3, 3}, torch::kFloat64);
  auto cell_acc = cell.accessor<double, 2>();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) cell_acc[i][j] = cell9[3 * i + j];
  auto pbc = torch::tensor({pbc3[0] != 0, pbc3[1] != 0, pbc3[2] != 0},
                           torch::kBool);
  auto edge_index = torch::from_blob(const_cast<int64_t*>(edge_index_2E),
                                     {2, n_edges}, torch::kInt64).clone();
  auto cell_offsets = torch::from_blob(const_cast<double*>(cell_offsets_E3),
                                       {n_edges, 3}, torch::kFloat64).clone();

  stage_inputs(pos, z, cell, pbc, 0, 0);
  if (edge_index.size(0) != 2)
    throw std::runtime_error("predict_host_extgraph_dd: edge_index must be [2,E]");
  edge_index_ = edge_index.to(device_, torch::kLong).contiguous();
  cell_offsets_ = cell_offsets.to(device_, compute_dtype_).contiguous();

  auto pred = predict_body_dd(nlocal, energy_out);
  if (forces_out) {
    auto f_cpu = pred.forces.to(torch::kCPU).contiguous();
    std::memcpy(forces_out, f_cpu.data_ptr<double>(),
                sizeof(double) * static_cast<size_t>(n) * 3);
  }
  return pred;
}

Prediction Predictor::predict_body_dd(int nlocal, double* energy_out) {
  const auto dtype = compute_dtype_;
  auto pos_grad = pos_.detach().clone().set_requires_grad(true);
  std::vector<torch::jit::IValue> args = {pos_grad,   atomic_numbers_,
                                          cell_,      pbc_,
                                          edge_index_, cell_offsets_,
                                          charge_,    spin_};

  torch::Tensor node_e_raw;
  {
    torch::autograd::AutoGradMode grad_guard(true);
    // DD artifact top returns a tuple (node_energy[n], total). Per-chunk/block/
    // prologue AC ops recompute independently, so run the module normally.
    auto out = module_.forward(args);
    torch::Tensor first;
    if (out.isTuple()) {
      first = out.toTuple()->elements()[0].toTensor();  // node_energy[n]
    } else {
      throw std::runtime_error(
          "predict_body_dd: DD artifact must return (node_energy, total); got a "
          "single tensor (re-export with UMA_DD_HALO=1)");
    }
    node_e_raw = first;
  }

  // Per-atom physical energy. denorm is E*rmsd + mean; mean is a PER-SYSTEM
  // constant, so for a per-atom vector apply rmsd per-atom and add mean ONCE
  // (below, to the owned sum) rather than per-atom. UMA omat mean is 0.0, so this
  // is exact; the explicit split avoids a silent per-atom mean bug if mean != 0.
  const double norm_mean = metadata_.normalizer_mean;
  auto node_e = node_e_raw.reshape({-1}).to(dtype) *
                torch::tensor(metadata_.normalizer_rmsd,
                              node_e_raw.options().dtype(dtype));
  if (element_refs_.defined()) {
    // PER-ATOM element reference: node_e[i] += refs[z[i]]. NOTE: the scalar path
    // uses undo_element_references (scatter_add by batch into one system energy),
    // which is WRONG for a per-atom vector (it would pile every ref onto atom 0).
    // Here add the per-atom reference directly, preserving sum-identity.
    auto refs = element_refs_.to(node_e.device(), node_e.scalar_type());
    auto z = atomic_numbers_.to(torch::kLong);
    node_e = node_e + refs.index({z});
  }

  // Backprop from the OWNED-only energy sum. Backpropagating the whole-subsystem
  // sum would inject spurious gradients from ghost-energy terms (also counted on
  // the ghost's owner); the owned-only root + the halo backward (ghost grad ->
  // owner) gives the exact global force on each owned atom.
  const int64_t n = node_e.size(0);
  const int64_t nl = std::min<int64_t>(nlocal, n);
  // Owned energy sum; add the per-system normalizer mean once (0 for omat).
  auto e_owned = node_e.narrow(0, 0, nl).sum();
  if (norm_mean != 0.0)
    e_owned = e_owned + torch::tensor(norm_mean, node_e.options());

  auto grads = torch::autograd::grad({e_owned}, {pos_grad},
                                     /*grad_outputs=*/{},
                                     /*retain_graph=*/false,
                                     /*create_graph=*/false,
                                     /*allow_unused=*/false);
  auto forces = (-grads[0]).to(torch::kFloat64).contiguous();

  // Per-node energy out (host FP64); caller sums owned rows into eng_vdwl.
  if (energy_out) {
    auto e_cpu = node_e.to(torch::kCPU, torch::kFloat64).contiguous();
    std::memcpy(energy_out, e_cpu.data_ptr<double>(),
                sizeof(double) * static_cast<size_t>(n));
  }

  Prediction out;
  out.energy = e_owned.item<double>();   // this rank's OWNED energy contribution
  out.forces = forces;
  return out;
}

}  // namespace uma
