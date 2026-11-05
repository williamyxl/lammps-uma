#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "uma/metadata.h"
#include "uma/predictor.h"

namespace uma {

class LibtorchMpRuntime;

/// Same-node graph-parallel backend for ``devices > 1``.
///
/// C++ LibTorch MP (``LibtorchMpRuntime`` + ``uma_peer`` + vesin) when
/// ``model_mp_w{N}_r*.pt`` artifacts exist. Multi-rank Intel GPU runs use
/// one MPI rank per device instead of this same-node fork path.
class GraphParallelRuntime {
 public:
  static std::unique_ptr<GraphParallelRuntime> create(
      const std::string& artifact_dir, const ArtifactMetadata& metadata,
      int num_devices, torch::ScalarType compute_dtype,
      bool activation_checkpointing = false);

  ~GraphParallelRuntime();

  GraphParallelRuntime(const GraphParallelRuntime&) = delete;
  GraphParallelRuntime& operator=(const GraphParallelRuntime&) = delete;

  Prediction predict(const torch::Tensor& pos, const torch::Tensor& atomic_numbers,
                     const torch::Tensor& cell, const torch::Tensor& pbc,
                     int64_t charge = 0, int64_t spin = 0);

  Prediction predict_host(int n, const double* pos_xyz, const int* atomic_numbers,
                          const double* cell_3x3, const int* pbc_3,
                          double* forces_out_optional = nullptr);

  int num_devices() const { return num_devices_; }
  const std::string& backend() const { return backend_; }
  const std::string& checkpoint() const { return checkpoint_; }

 private:
  GraphParallelRuntime(int num_devices, std::string checkpoint, std::string backend);

  int num_devices_ = 1;
  std::string checkpoint_;
  std::string backend_ = "libtorch_vesin";

  std::unique_ptr<LibtorchMpRuntime> cpp_mp_;
};

}  // namespace uma
