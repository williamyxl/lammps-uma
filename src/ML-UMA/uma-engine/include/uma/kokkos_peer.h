#pragma once

// Same-node peer collectives for native UMA graph-parallel (no MPI / no c10d).
// Semantics mirror FairChem gp_utils gather/reduce used in escn_md(_block).
//
// Transport: CUDA peer memcpy + device synchronize (Kokkos fence analogue from
// src/KOKKOS/comm_kokkos.cpp: fence before/after device ptr traffic). Host
// staging (.to) remains as fallback when peer access is unavailable.

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#if defined(UMA_ENGINE_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace uma {
namespace kokkos_peer {

inline std::vector<int64_t> size_list(int64_t n_atoms, int world_size) {
  if (world_size < 1) {
    throw std::runtime_error("kokkos_peer::size_list: world_size < 1");
  }
  std::vector<int64_t> out(static_cast<size_t>(world_size));
  for (int i = 0; i < world_size; ++i) {
    out[static_cast<size_t>(i)] =
        n_atoms / world_size + (i < (n_atoms % world_size) ? 1 : 0);
  }
  return out;
}

inline int64_t padded_local_size(int64_t n_atoms, int world_size) {
  if (world_size < 1) {
    throw std::runtime_error("kokkos_peer::padded_local_size: world_size < 1");
  }
  return n_atoms / world_size + (n_atoms % world_size ? 1 : 0);
}

// Pad leading dim to padded_size (FairChem pad_input).
inline torch::Tensor pad_nodes(const torch::Tensor& input, int64_t padded_size) {
  if (input.size(0) == padded_size) return input;
  if (input.size(0) > padded_size) {
    throw std::runtime_error("kokkos_peer::pad_nodes: input larger than pad");
  }
  auto opts = input.options();
  std::vector<int64_t> zshape = input.sizes().vec();
  zshape[0] = padded_size - input.size(0);
  auto z = torch::zeros(zshape, opts);
  return torch::cat({input, z}, /*dim=*/0).contiguous();
}

// Fence analogue: synchronize the current CUDA device (and optionally all).
inline void fence_device(c10::DeviceIndex dev = -1) {
#if defined(UMA_ENGINE_USE_CUDA)
  if (!torch::cuda::is_available()) return;
  if (dev >= 0) {
    cudaSetDevice(static_cast<int>(dev));
  }
  cudaDeviceSynchronize();
#else
  (void)dev;
#endif
}

inline void fence_all(int world_size) {
#if defined(UMA_ENGINE_USE_CUDA)
  if (!torch::cuda::is_available()) return;
  const int ndev = static_cast<int>(torch::cuda::device_count());
  const int n = std::min(world_size, ndev);
  for (int i = 0; i < n; ++i) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
#else
  (void)world_size;
#endif
}

// Enable peer access between all visible CUDA devices (best-effort).
inline void enable_peer_access(int world_size) {
#if defined(UMA_ENGINE_USE_CUDA)
  if (!torch::cuda::is_available()) return;
  const int ndev = static_cast<int>(torch::cuda::device_count());
  const int n = std::min(world_size, ndev);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      int can = 0;
      cudaDeviceCanAccessPeer(&can, i, j);
      if (!can) continue;
      cudaSetDevice(i);
      // cudaErrorPeerAccessAlreadyEnabled is fine.
      cudaError_t st = cudaDeviceEnablePeerAccess(j, 0);
      if (st != cudaSuccess && st != cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();  // clear
      } else {
        cudaGetLastError();
      }
    }
  }
#else
  (void)world_size;
#endif
}

// Device→device copy preferring cudaMemcpyPeer; falls back to .to().
inline torch::Tensor peer_copy(const torch::Tensor& src, torch::Device dst_dev) {
  auto s = src.contiguous();
  if (s.device() == dst_dev) return s;
  if (!s.is_cuda() || !dst_dev.is_cuda()) {
    return s.to(dst_dev, /*non_blocking=*/false).contiguous();
  }
#if defined(UMA_ENGINE_USE_CUDA)
  auto out = torch::empty(s.sizes(), s.options().device(dst_dev));
  const int src_id = static_cast<int>(s.get_device());
  const int dst_id = static_cast<int>(dst_dev.index());
  fence_device(src_id);
  const cudaError_t st = cudaMemcpyPeer(out.data_ptr(), dst_id, s.data_ptr(), src_id,
                                        static_cast<size_t>(s.nbytes()));
  if (st != cudaSuccess) {
    cudaGetLastError();
    return s.to(dst_dev, /*non_blocking=*/false).contiguous();
  }
  fence_device(dst_id);
  return out;
#else
  return s.to(dst_dev, /*non_blocking=*/false).contiguous();
#endif
}

// all_gather along dim0: each device contributes a padded local shard;
// returns concatenated unpadded [n_atoms, ...] on `out_device`.
inline torch::Tensor all_gather_nodes(const std::vector<torch::Tensor>& locals,
                                      int64_t n_atoms, torch::Device out_device) {
  const int world = static_cast<int>(locals.size());
  if (world < 1) {
    throw std::runtime_error("kokkos_peer::all_gather_nodes: empty");
  }
  auto sizes = size_list(n_atoms, world);
  std::vector<torch::Tensor> parts;
  parts.reserve(static_cast<size_t>(world));
  for (int r = 0; r < world; ++r) {
    auto t = peer_copy(locals[static_cast<size_t>(r)], out_device);
    const auto want = sizes[static_cast<size_t>(r)];
    if (t.size(0) < want) {
      throw std::runtime_error("kokkos_peer::all_gather_nodes: shard too small");
    }
    parts.push_back(t.narrow(0, 0, want).contiguous());
  }
  return torch::cat(parts, /*dim=*/0).contiguous();
}

// Sum-reduce tensors onto out_device.
inline torch::Tensor all_reduce_sum(const std::vector<torch::Tensor>& parts,
                                    torch::Device out_device) {
  if (parts.empty()) {
    throw std::runtime_error("kokkos_peer::all_reduce_sum: empty");
  }
  torch::Tensor acc = peer_copy(parts[0], out_device).clone();
  for (size_t i = 1; i < parts.size(); ++i) {
    acc = acc + peer_copy(parts[i], out_device);
  }
  return acc;
}

// Thread-safe same-process gather slot (Kokkos-visible devices, single MPI rank).
// Generation-matched barrier — avoids pairing phase-N wait with phase-N+1.
class PeerGatherSlot {
 public:
  explicit PeerGatherSlot(int world) : world_(world), bufs_(static_cast<size_t>(world)) {
    if (world < 1) throw std::runtime_error("PeerGatherSlot: world < 1");
  }

  torch::Tensor all_gather_concat(int rank, const torch::Tensor& local, int64_t n_atoms) {
    auto locals = exchange_(rank, local);
    return all_gather_nodes(locals, n_atoms, local.device());
  }

  torch::Tensor all_reduce(int rank, const torch::Tensor& local) {
    auto locals = exchange_(rank, local);
    return all_reduce_sum(locals, local.device());
  }

  // Barrier only (exchange ignored payloads).
  void barrier(int rank) {
    auto dummy = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat64));
    if (torch::cuda::is_available() && rank >= 0) {
      dummy = dummy.to(torch::Device(torch::kCUDA, rank));
    }
    (void)exchange_(rank, dummy);
  }

  int world() const { return world_; }

 private:
  std::vector<torch::Tensor> exchange_(int rank, const torch::Tensor& local) {
    if (rank < 0 || rank >= world_) {
      throw std::runtime_error("PeerGatherSlot: bad rank");
    }
    std::unique_lock<std::mutex> lk(mu_);
    const int gen = gen_;
    bufs_[static_cast<size_t>(rank)] = local.contiguous();
    ++nwrite_;
    if (nwrite_ == world_) {
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return nwrite_ == world_ || gen_ != gen; });
    }
    if (gen_ != gen) {
      throw std::runtime_error("PeerGatherSlot: generation advanced early");
    }
    std::vector<torch::Tensor> locals = bufs_;
    ++nread_;
    if (nread_ == world_) {
      for (auto& b : bufs_) b = torch::Tensor();
      nwrite_ = 0;
      nread_ = 0;
      ++gen_;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return gen_ != gen; });
    }
    return locals;
  }

  int world_;
  std::mutex mu_;
  std::condition_variable cv_;
  int gen_ = 0;
  int nwrite_ = 0;
  int nread_ = 0;
  std::vector<torch::Tensor> bufs_;
};

}  // namespace kokkos_peer
}  // namespace uma
