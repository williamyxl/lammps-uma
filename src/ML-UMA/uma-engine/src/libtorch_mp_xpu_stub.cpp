// XPU build stub for the CUDA/NCCL C++ LibTorch multi-process runtime.
//
// The real LibtorchMpRuntime (libtorch_mp.cpp) is CUDA/NCCL-only (includes
// cuda_runtime_api.h, uma_peer NCCL collectives). On XPU we support the
// single-GPU EAGER Python-worker path (UMA_EAGER_CKPT=1), which lives entirely
// in graph_parallel.cpp (POSIX fork + pipe protocol to uma_gp_worker.py) and
// does NOT use LibtorchMpRuntime. This stub satisfies the linker for the
// !force_python branch, which is never taken with UMA_EAGER_CKPT=1.

#include "uma/libtorch_mp.h"

#include <stdexcept>

namespace uma {

// Complete the opaque Impl so the (never-instantiated) destructor links.
struct LibtorchMpRuntime::Impl {};

bool LibtorchMpRuntime::artifacts_present(const std::string& /*artifact_dir*/,
                                         int /*num_devices*/) {
  return false;  // no C++ MP shards on XPU -> forces the eager Python path
}

std::unique_ptr<LibtorchMpRuntime> LibtorchMpRuntime::try_create(
    const std::string& /*artifact_dir*/, const ArtifactMetadata& /*metadata*/,
    int /*num_devices*/, torch::ScalarType /*compute_dtype*/) {
  return nullptr;  // graph_parallel.cpp then reports: use UMA_EAGER_CKPT=1
}

LibtorchMpRuntime::LibtorchMpRuntime(int num_devices, ArtifactMetadata metadata,
                                     torch::ScalarType compute_dtype)
    : num_devices_(num_devices),
      metadata_(std::move(metadata)),
      compute_dtype_(compute_dtype) {}

LibtorchMpRuntime::~LibtorchMpRuntime() = default;

void LibtorchMpRuntime::rebuild_neighbors_full(torch::Device /*build_dev*/) {}

Prediction LibtorchMpRuntime::predict(const torch::Tensor&, const torch::Tensor&,
                                      const torch::Tensor&, const torch::Tensor&,
                                      int64_t, int64_t) {
  throw std::runtime_error(
      "LibtorchMpRuntime unavailable on XPU build (use UMA_EAGER_CKPT=1)");
}

Prediction LibtorchMpRuntime::predict_host(int, const double*, const int*,
                                           const double*, const int*, double*) {
  throw std::runtime_error(
      "LibtorchMpRuntime unavailable on XPU build (use UMA_EAGER_CKPT=1)");
}

}  // namespace uma
