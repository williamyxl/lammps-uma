#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "uma/metadata.h"
#include "uma/predictor.h"

namespace uma {

/// Native Kokkos+LibTorch multi-GPU evaluator (no Python / Ray / c10d).
/// Process-per-rank (TorchScript is not multi-thread safe for mid-forward
/// collectives). Requires ``model_mp_w{N}_r{R}.pt`` from export_mp_artifact.py.
class LibtorchMpRuntime {
 public:
  static std::unique_ptr<LibtorchMpRuntime> try_create(
      const std::string& artifact_dir, const ArtifactMetadata& metadata,
      int num_devices, torch::ScalarType compute_dtype);

  static bool artifacts_present(const std::string& artifact_dir, int num_devices);

  ~LibtorchMpRuntime();

  LibtorchMpRuntime(const LibtorchMpRuntime&) = delete;
  LibtorchMpRuntime& operator=(const LibtorchMpRuntime&) = delete;

  Prediction predict(const torch::Tensor& pos, const torch::Tensor& atomic_numbers,
                     const torch::Tensor& cell, const torch::Tensor& pbc,
                     int64_t charge = 0, int64_t spin = 0);

  Prediction predict_host(int n, const double* pos_xyz, const int* atomic_numbers,
                          const double* cell_3x3, const int* pbc_3,
                          double* forces_out_optional = nullptr);

  int num_devices() const { return num_devices_; }
  const std::string& backend() const { return backend_; }

 private:
  struct Worker;
  struct Impl;

  LibtorchMpRuntime(int num_devices, ArtifactMetadata metadata,
                    torch::ScalarType compute_dtype);

  void rebuild_neighbors_full(torch::Device build_dev);

  int num_devices_ = 1;
  ArtifactMetadata metadata_;
  torch::ScalarType compute_dtype_ = torch::kFloat64;
  std::string backend_ = "kokkos_libtorch_vesin";
  std::unique_ptr<Impl> impl_;
  torch::Tensor element_refs_;

  int64_t n_cached_ = -1;
  torch::Tensor pos0_;
  torch::Tensor z0_;
  torch::Tensor cell0_;
  torch::Tensor pbc0_;
  torch::Tensor edge_index_;
  torch::Tensor cell_offsets_;
  torch::Tensor batch0_;
};

}  // namespace uma
