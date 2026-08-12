#include "uma/predictor.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include "uma/graph_parallel.h"
#include "uma/neighbor_list.h"
#include "uma/postprocess.h"
#include "uma/vesin_nl.h"

namespace {

torch::Device resolve_device(torch::Device device) {
  if (device.is_cuda() && !torch::cuda::is_available()) {
    std::cerr << "Warning: CUDA requested but unavailable; falling back to CPU\n";
    return torch::kCPU;
  }
  return device;
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

Predictor::~Predictor() = default;
Predictor::Predictor(Predictor&&) noexcept = default;
Predictor& Predictor::operator=(Predictor&&) noexcept = default;

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

  // FairChem AtomicData wraps into the cell; model + edge offsets share that frame.
  pos_.copy_(wrap_positions_to_cell(pos_, cell_, pbc_));

  rebuild_neighbors();

  // Clone so the persistent buffer is not marked requires_grad.
  auto pos_grad = pos_.detach().clone().set_requires_grad(true);

  std::vector<torch::jit::IValue> args = {pos_grad,
                                          atomic_numbers_,
                                          cell_,
                                          pbc_,
                                          edge_index_,
                                          cell_offsets_,
                                          charge_,
                                          spin_};

  torch::Tensor normed_raw;
  {
    torch::autograd::AutoGradMode grad_guard(true);
    normed_raw = module_.forward(args).toTensor();
  }
  auto normed = normed_raw.to(dtype);
  auto energy = denorm_energy(normed, metadata_.normalizer_mean,
                              metadata_.normalizer_rmsd);
  if (element_refs_.defined()) {
    energy = undo_element_references(energy, atomic_numbers_, batch_,
                                     element_refs_);
  }

  auto grads = torch::autograd::grad({energy.sum()}, {pos_grad},
                                     /*grad_outputs=*/{},
                                     /*retain_graph=*/false,
                                     /*create_graph=*/false,
                                     /*allow_unused=*/false);
  auto forces = (-grads[0]).to(torch::kFloat64).contiguous();

  Prediction out;
  out.energy = energy.reshape({-1})[0].item<double>();
  out.forces = forces;
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

}  // namespace uma
