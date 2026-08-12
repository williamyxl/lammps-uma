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
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include "uma/graph_shard.h"
#include "uma/peer_context.h"
#include "uma/postprocess.h"
#include "uma/shared_peer.h"
#include "uma/vesin_nl.h"
#include "uma/neighbor_list.h"

namespace uma {

namespace kp = kokkos_peer;

struct MpiPeerPredictor::Impl {
  torch::jit::script::Module module;
  torch::Device device{torch::kCUDA, 0};
  kp::SharedPeerGatherSlot* slot = nullptr;   // private per-rank Shm (NCCL only)
  kp::SharedPeerGatherSlot::Shm* shm = nullptr;
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
  if (nccl_unique_id == nullptr)
    throw std::runtime_error("MpiPeerPredictor: null nccl id (rank0 make + MPI_Bcast)");

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

#if defined(UMA_ENGINE_USE_CUDA)
  self->impl_->device = torch::Device(torch::kCUDA, 0);  // CUDA_VISIBLE_DEVICES pins one GPU
#else
  self->impl_->device = torch::Device(torch::kCPU);
#endif

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

  // Load this rank's shard: model_mp_w{W}_n{N}_r{R}.pt (n-specific preferred).
  std::string path;
  if (const char* natoms_e = std::getenv("UMA_MP_NATOMS")) {
    if (*natoms_e) {
      const std::string nspec = artifact_dir + "/model_mp_w" + std::to_string(world) +
                                "_n" + natoms_e + "_r" + std::to_string(rank) + ".pt";
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

  if (metadata.element_references.defined()) {
    self->impl_->element_refs =
        metadata.element_references.to(self->impl_->device, compute_dtype).contiguous();
  }
  std::cerr << "MpiPeerPredictor rank=" << rank << " world=" << world
            << " dev=" << device_index << " shard=" << path << "\n";
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

  auto pos_grad = pos.detach().clone().set_requires_grad(true);
  auto charge = torch::zeros({}, torch::TensorOptions().dtype(torch::kLong).device(dev));
  auto spin = torch::zeros({}, torch::TensorOptions().dtype(torch::kLong).device(dev));

#if defined(UMA_ENGINE_USE_CUDA)
  if (perf) cudaDeviceSynchronize();
#endif
  auto t_graph = clk::now();

  std::vector<torch::jit::IValue> args = {pos_grad, z, cell, pbc, eidx, coff, charge, spin};
  torch::Tensor normed;
  {
    torch::autograd::AutoGradMode guard(true);
    normed = I.module.forward(args).toTensor().to(dtype);
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
