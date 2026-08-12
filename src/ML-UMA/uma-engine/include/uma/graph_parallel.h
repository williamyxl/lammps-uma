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
/// **Default:** C++ LibTorch MP (``LibtorchMpRuntime`` + ``uma_peer`` + vesin)
/// when ``model_mp_w{N}_r*.pt`` artifacts exist.
///
/// **Opt-in Python:** ``UMA_PYTHON_GP_WORKER=1`` → ``uma_native_gp_worker.py``
/// (or Ray if ``UMA_ALLOW_RAY_GP=1``). Not the product path.
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

  void write_line(const std::string& line);
  std::string read_line();
  void read_exact(void* dst, size_t n);
  void write_exact(const void* src, size_t n);
  void shutdown_worker();

  int num_devices_ = 1;
  std::string checkpoint_;
  std::string backend_ = "kokkos_libtorch_vesin";
  pid_t child_pid_ = -1;
  int to_child_fd_ = -1;    // parent writes
  int from_child_fd_ = -1;  // parent reads

  std::unique_ptr<LibtorchMpRuntime> cpp_mp_;
};

/// Resolve checkpoint for eager GP (metadata → env → lab default).
std::string resolve_gp_checkpoint(const std::string& artifact_dir,
                                  const ArtifactMetadata& metadata);

/// Locate Python GP worker script (legacy opt-in only).
std::string resolve_gp_worker_script();

}  // namespace uma
