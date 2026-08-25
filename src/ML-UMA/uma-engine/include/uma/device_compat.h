#pragma once

// Device abstraction for the UMA engine: CUDA (NVIDIA) or XPU (Intel/Aurora).
//
// The engine was originally CUDA-only. On Aurora we use torch 2.13.0+xpu
// (native XPU, no IPEX). This header centralizes device availability, the
// device-count / sync helpers, and a resolve() that falls back to CPU when the
// requested accelerator is unavailable, mirroring graspa-mlip/src_clean/
// mlip_device.h. Single-tile (num_devices==1) needs no collectives.
//
// Build: torch-XPU wheels expose torch::xpu::* and at::hasXPU(). We guard the
// XPU calls so a CUDA-only libtorch still compiles.

#include <torch/torch.h>

namespace uma {

inline bool cuda_ok() {
  return torch::cuda::is_available();
}

inline bool xpu_ok() {
#if defined(UMA_ENGINE_USE_XPU)
  return at::hasXPU() && torch::xpu::is_available();
#else
  return false;
#endif
}

// Best available accelerator, else CPU. Prefers whatever libtorch was built for.
inline torch::Device default_device() {
#if defined(UMA_ENGINE_USE_XPU)
  if (xpu_ok()) return torch::Device(torch::kXPU, 0);
#endif
  if (cuda_ok()) return torch::Device(torch::kCUDA, 0);
  return torch::Device(torch::kCPU);
}

// Resolve a requested device, falling back to CPU if its backend is absent.
inline torch::Device resolve_device_compat(torch::Device device) {
  if (device.is_cuda() && !cuda_ok()) {
    return torch::kCPU;
  }
#if defined(UMA_ENGINE_USE_XPU)
  if (device.is_xpu() && !xpu_ok()) {
    return torch::kCPU;
  }
#endif
  return device;
}

inline void device_synchronize(const torch::Device& device) {
  if (device.is_cuda() && cuda_ok()) {
    torch::cuda::synchronize();
  }
#if defined(UMA_ENGINE_USE_XPU)
  else if (device.is_xpu() && xpu_ok()) {
    torch::xpu::synchronize(device.index() < 0 ? 0 : device.index());
  }
#endif
}

inline int accelerator_device_count() {
#if defined(UMA_ENGINE_USE_XPU)
  if (xpu_ok()) return static_cast<int>(torch::xpu::device_count());
#endif
  if (cuda_ok()) return static_cast<int>(torch::cuda::device_count());
  return 0;
}

}  // namespace uma
