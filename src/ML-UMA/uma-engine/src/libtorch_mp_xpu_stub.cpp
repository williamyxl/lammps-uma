// XPU build stub for the CUDA/NCCL C++ LibTorch multi-process runtime.
//
// The real LibtorchMpRuntime (libtorch_mp.cpp) is CUDA/NCCL-only. On Intel
// GPU, same-node fork workers are not used: graph-parallel is one MPI rank
// per device via MpiPeerPredictor. This stub satisfies the linker.

#include "uma/libtorch_mp.h"

#include <stdexcept>

namespace uma {

// Complete the opaque Impl so the (never-instantiated) destructor links.
struct LibtorchMpRuntime::Impl {};

bool LibtorchMpRuntime::artifacts_present(const std::string& /*artifact_dir*/,
                                         int /*num_devices*/) {
  return false;
}

std::unique_ptr<LibtorchMpRuntime> LibtorchMpRuntime::try_create(
    const std::string& /*artifact_dir*/, const ArtifactMetadata& /*metadata*/,
    int /*num_devices*/, torch::ScalarType /*compute_dtype*/) {
  return nullptr;
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
      "LibtorchMpRuntime unavailable on XPU build; use one MPI rank per GPU");
}

Prediction LibtorchMpRuntime::predict_host(int, const double*, const int*,
                                           const double*, const int*, double*) {
  throw std::runtime_error(
      "LibtorchMpRuntime unavailable on XPU build; use one MPI rank per GPU");
}

}  // namespace uma
