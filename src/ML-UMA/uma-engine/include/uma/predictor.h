#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "uma/metadata.h"

namespace uma {

struct Prediction {
  double energy = 0.0;   // physical energy (FP32 or FP64 model → host double)
  /// Forces [N,3] float64; typically on the engine device.
  torch::Tensor forces;
};

/// GPU-persistent UMA inference: energy + forces via autograd on energy.
///
/// Compute dtype (positions, cell, energy) is set from artifact metadata and
/// may be overridden via set_compute_dtype() for LAMMPS `precision mixed|double`.
/// Forces are always FP64.
class Predictor {
 public:
  static Predictor from_artifact(const std::string& artifact_dir,
                                 torch::Device device = torch::kCUDA);

  Prediction predict(const torch::Tensor& pos,             // [N,3]
                     const torch::Tensor& atomic_numbers,  // [N] int64
                     const torch::Tensor& cell,            // [3,3]
                     const torch::Tensor& pbc,             // [3] bool
                     int64_t charge = 0, int64_t spin = 0);

  /// Host FP32 positions (downcasts if artifact is FP64).
  Prediction predict_host(int n, const float* pos_xyz, const int* atomic_numbers,
                          const double* cell_3x3, const int* pbc_3,
                          double* forces_out_optional = nullptr);

  /// Host FP64 positions (preferred for float64 / double precision).
  Prediction predict_host(int n, const double* pos_xyz, const int* atomic_numbers,
                          const double* cell_3x3, const int* pbc_3,
                          double* forces_out_optional = nullptr);

  /// Override position/energy compute dtype (must match TorchScript artifact dtype).
  void set_compute_dtype(torch::ScalarType dtype);

  const ArtifactMetadata& metadata() const { return metadata_; }
  torch::Device device() const { return device_; }
  torch::ScalarType compute_dtype() const { return compute_dtype_; }

 private:
  Predictor(torch::jit::script::Module module, torch::Device device,
            ArtifactMetadata metadata);

  void ensure_buffers(int64_t n);
  void rebuild_neighbors();

  torch::jit::script::Module module_;
  torch::Device device_;
  ArtifactMetadata metadata_;
  torch::ScalarType compute_dtype_ = torch::kFloat32;

  // Persistent device tensors (reused when N unchanged).
  int64_t n_cached_ = -1;
  torch::Tensor pos_;              // compute_dtype [N,3]
  torch::Tensor atomic_numbers_;   // int64 [N]
  torch::Tensor cell_;             // compute_dtype [3,3]
  torch::Tensor pbc_;              // bool [3]
  torch::Tensor edge_index_;       // int64 [2,E]
  torch::Tensor cell_offsets_;     // compute_dtype [E,3]
  torch::Tensor charge_;
  torch::Tensor spin_;
  torch::Tensor batch_;
  torch::Tensor element_refs_;     // compute_dtype on device
};

}  // namespace uma
