#include "uma/graph_parallel.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include "uma/libtorch_mp.h"
#include "uma/runtime_config.h"

namespace uma {

std::unique_ptr<GraphParallelRuntime> GraphParallelRuntime::create(
    const std::string& artifact_dir, const ArtifactMetadata& metadata,
    int num_devices, torch::ScalarType compute_dtype,
    bool activation_checkpointing) {
  if (const char* e = uma::runtime_get("UMA_PYTHON_GP_WORKER")) {
    if (e[0] == '1' && e[1] == '\0') {
      throw std::runtime_error(
          "GraphParallelRuntime: the Python GP worker was removed. "
          "Use C++ LibTorch MP artifacts (model_mp_wN_r*.pt) or one MPI rank "
          "per GPU with devices 1.");
    }
  }
  if (activation_checkpointing) {
    throw std::runtime_error(
        "GraphParallelRuntime: eager Python checkpointing was removed. "
        "Use pair_style keyword ac on the traced single-tile path, or "
        "graph-parallel with ac off.");
  }
  if (num_devices <= 1) {
    throw std::runtime_error(
        "GraphParallelRuntime::create requires num_devices > 1");
  }

  if (LibtorchMpRuntime::artifacts_present(artifact_dir, num_devices)) {
    auto cpp = LibtorchMpRuntime::try_create(artifact_dir, metadata, num_devices,
                                             compute_dtype);
    if (!cpp) {
      throw std::runtime_error(
          "GraphParallelRuntime: C++ LibTorch MP artifacts present but "
          "LibtorchMpRuntime::try_create failed");
    }
    auto rt = std::unique_ptr<GraphParallelRuntime>(new GraphParallelRuntime(
        num_devices, /*checkpoint=*/artifact_dir, cpp->backend()));
    rt->cpp_mp_ = std::move(cpp);
    return rt;
  }
  throw std::runtime_error(
      "GraphParallelRuntime: C++ LibTorch MP artifacts missing under " +
      artifact_dir + " (need model_mp_w" + std::to_string(num_devices) +
      "_r*.pt from python/export_mp_artifact.py). On Intel GPU use one MPI "
      "rank per device with devices 1, not same-node fork workers.");
}

GraphParallelRuntime::GraphParallelRuntime(int num_devices, std::string checkpoint,
                                           std::string backend)
    : num_devices_(num_devices),
      checkpoint_(std::move(checkpoint)),
      backend_(std::move(backend)) {}

GraphParallelRuntime::~GraphParallelRuntime() { cpp_mp_.reset(); }

Prediction GraphParallelRuntime::predict_host(int n, const double* pos_xyz,
                                              const int* atomic_numbers,
                                              const double* cell_3x3,
                                              const int* pbc_3,
                                              double* forces_out_optional) {
  if (!cpp_mp_) {
    throw std::runtime_error("GraphParallelRuntime: C++ LibTorch MP is not loaded");
  }
  return cpp_mp_->predict_host(n, pos_xyz, atomic_numbers, cell_3x3, pbc_3,
                               forces_out_optional);
}

Prediction GraphParallelRuntime::predict(const torch::Tensor& pos,
                                         const torch::Tensor& atomic_numbers,
                                         const torch::Tensor& cell,
                                         const torch::Tensor& pbc, int64_t charge,
                                         int64_t spin) {
  if (!cpp_mp_) {
    throw std::runtime_error("GraphParallelRuntime: C++ LibTorch MP is not loaded");
  }
  return cpp_mp_->predict(pos, atomic_numbers, cell, pbc, charge, spin);
}

}  // namespace uma
