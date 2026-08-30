// XPU single-tile build stub for the same-node graph-parallel backend.
//
// The real GraphParallelRuntime (graph_parallel.cpp + libtorch_mp.cpp +
// mpi_peer_predictor.cpp) is CUDA/NCCL-only (includes cuda_runtime_api.h /
// nccl.h at file scope) and cannot compile against torch+xpu. Phase 2/3 target
// single-tile (num_devices==1), which never constructs a GraphParallelRuntime.
//
// This stub provides just enough symbols for predictor.cpp to link:
//   * GraphParallelRuntime::create()  -> throws (devices>1 unsupported on XPU)
//   * ~GraphParallelRuntime()         -> defined (unique_ptr member needs it)
//   * resolve_gp_checkpoint / resolve_gp_worker_script helpers
//
// Multi-tile on XPU is Phase 4 (oneCCL/MPI edge-parallel), which will replace
// this stub with an XPU-native collective path.

#include "uma/graph_parallel.h"

#include <stdexcept>

namespace uma {

// Minimal definition so unique_ptr<LibtorchMpRuntime> has a complete type here.
class LibtorchMpRuntime {};

std::unique_ptr<GraphParallelRuntime> GraphParallelRuntime::create(
    const std::string& /*artifact_dir*/, const ArtifactMetadata& /*metadata*/,
    int /*num_devices*/, torch::ScalarType /*compute_dtype*/,
    bool /*activation_checkpointing*/) {
  throw std::runtime_error(
      "uma-engine (XPU build): same-node graph-parallel (devices>1) is not "
      "supported yet; use devices=1 (single tile). Multi-tile XPU is Phase 4.");
}

GraphParallelRuntime::~GraphParallelRuntime() = default;

GraphParallelRuntime::GraphParallelRuntime(int num_devices, std::string checkpoint,
                                           std::string backend)
    : num_devices_(num_devices),
      checkpoint_(std::move(checkpoint)),
      backend_(std::move(backend)) {}

Prediction GraphParallelRuntime::predict(const torch::Tensor&, const torch::Tensor&,
                                         const torch::Tensor&, const torch::Tensor&,
                                         int64_t, int64_t) {
  throw std::runtime_error("GraphParallelRuntime::predict unavailable on XPU build");
}

Prediction GraphParallelRuntime::predict_host(int, const double*, const int*,
                                              const double*, const int*, double*) {
  throw std::runtime_error("GraphParallelRuntime::predict_host unavailable on XPU build");
}

void GraphParallelRuntime::write_line(const std::string&) {}
std::string GraphParallelRuntime::read_line() { return {}; }
void GraphParallelRuntime::read_exact(void*, size_t) {}
void GraphParallelRuntime::write_exact(const void*, size_t) {}
void GraphParallelRuntime::shutdown_worker() {}

std::string resolve_gp_checkpoint(const std::string& /*artifact_dir*/,
                                  const ArtifactMetadata& /*metadata*/) {
  return {};
}

std::string resolve_gp_worker_script() { return {}; }

}  // namespace uma
