#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "uma/metadata.h"
#include "uma/predictor.h"

namespace uma {

/// Same-node graph-parallel backend for ``devices > 1``.
///
/// Ships FairChem eager GP (``load_predict_unit(..., workers=N)`` / Ray + NCCL)
/// via a persistent ``uma_gp_worker.py`` subprocess. devices=1 stays on the
/// traced LibTorch ``Predictor`` path — this class is never used for N=1.
///
/// Not a serial DevicePool: workers>1 runs real FairChem ParallelMLIPPredictUnit.
class GraphParallelRuntime {
 public:
  static std::unique_ptr<GraphParallelRuntime> create(
      const std::string& artifact_dir, const ArtifactMetadata& metadata,
      int num_devices, torch::ScalarType compute_dtype);

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
  std::string backend_ = "fairchem_eager_python";
  pid_t child_pid_ = -1;
  int to_child_fd_ = -1;    // parent writes
  int from_child_fd_ = -1;  // parent reads
};

/// Resolve checkpoint for eager GP (metadata → env → lab default).
std::string resolve_gp_checkpoint(const std::string& artifact_dir,
                                  const ArtifactMetadata& metadata);

/// Locate ``uma_gp_worker.py`` (env UMA_GP_WORKER or engine python dir).
std::string resolve_gp_worker_script();

}  // namespace uma
