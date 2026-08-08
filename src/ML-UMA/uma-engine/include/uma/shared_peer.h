#pragma once

// Process-shared peer gather/reduce for multi-process LibTorch MP.
// Threads cannot run concurrent jit::Module::forward (deadlock). Each GP rank
// is a process; control sync uses PTHREAD_PROCESS_SHARED.
//
// Transports (UMA_PEER_TRANSPORT=shm|cuda_ipc):
//   shm      — host-staged payload in mmap (legacy / fallback)
//   cuda_ipc — device payload via cudaIpcMemHandle_t; shm holds only control +
//              handles + nbytes. Default when CUDA is available.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#include <torch/torch.h>

#include "uma/kokkos_peer.h"

#if defined(UMA_ENGINE_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace uma {
namespace kokkos_peer {

class SharedPeerGatherSlot {
 public:
  static constexpr size_t kMaxBytesPerRank = 256ull * 1024ull * 1024ull;
  static constexpr int kTransportShm = 0;
  static constexpr int kTransportCudaIpc = 1;

#if defined(UMA_ENGINE_USE_CUDA)
  using IpcHandle = cudaIpcMemHandle_t;
#else
  struct IpcHandle {
    char bytes[64];
  };
#endif

  struct Shm {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int gen;
    int nwrite;
    int nread;
    int world;
    int transport;       // kTransportShm | kTransportCudaIpc
    int ipc_init_count;  // ranks that published IPC handles
    // trailing per rank (see rank_stride):
    //   int64_t nbytes
    //   [cuda_ipc] IpcHandle
    //   [shm]      char payload[kMaxBytesPerRank]
  };

  static int select_transport() {
    if (const char* e = std::getenv("UMA_PEER_TRANSPORT")) {
      if (std::strcmp(e, "shm") == 0) return kTransportShm;
      if (std::strcmp(e, "cuda_ipc") == 0) return kTransportCudaIpc;
    }
#if defined(UMA_ENGINE_USE_CUDA)
    if (torch::cuda::is_available()) return kTransportCudaIpc;
#endif
    return kTransportShm;
  }

  static size_t rank_stride_for(int transport) {
    size_t n = sizeof(int64_t);
    if (transport == kTransportCudaIpc) {
      n += sizeof(IpcHandle);
    } else {
      n += kMaxBytesPerRank;
    }
    return n;
  }

  static size_t map_bytes_for(int world, int transport) {
    if (world < 1) throw std::runtime_error("SharedPeerGatherSlot: world < 1");
    return sizeof(Shm) + static_cast<size_t>(world) * rank_stride_for(transport);
  }

  static const char* transport_name(int transport) {
    return transport == kTransportCudaIpc ? "cuda_ipc" : "shm";
  }

  static SharedPeerGatherSlot* create(int world) {
    const int transport = select_transport();
    const size_t bytes = map_bytes_for(world, transport);
    void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
      throw std::runtime_error("SharedPeerGatherSlot: mmap failed");
    }
    auto* shm = static_cast<Shm*>(mem);
    std::memset(shm, 0, bytes);
    shm->world = world;
    shm->transport = transport;
    init_sync_primitives_(shm);
    return new SharedPeerGatherSlot(shm, bytes, /*owns=*/true);
  }

  // Map an existing memfd / inherited fd (worker attach; does not own unmap).
  static SharedPeerGatherSlot* attach(Shm* shm, size_t map_bytes) {
    return new SharedPeerGatherSlot(shm, map_bytes, /*owns=*/false);
  }

  size_t map_bytes() const { return map_bytes_; }
  Shm* raw() { return shm_; }
  int transport() const { return shm_ ? shm_->transport : kTransportShm; }

  void destroy() {
    release_cuda_ipc();
    if (!shm_) {
      delete this;
      return;
    }
    if (owns_) {
      pthread_mutex_destroy(&shm_->mu);
      pthread_cond_destroy(&shm_->cv);
      munmap(shm_, map_bytes_);
    }
    shm_ = nullptr;
    delete this;
  }

  int world() const { return shm_->world; }

  // Worker-only: allocate CUDA send buffer, publish IPC handle, open remotes.
  // No-op for shm transport or if already initialized.
  void init_cuda_ipc(int rank) {
    if (!shm_ || shm_->transport != kTransportCudaIpc) return;
    if (ipc_ready_) return;
#if !defined(UMA_ENGINE_USE_CUDA)
    throw std::runtime_error("SharedPeerGatherSlot: cuda_ipc requested without CUDA");
#else
    if (rank < 0 || rank >= shm_->world) {
      throw std::runtime_error("SharedPeerGatherSlot: bad rank for init_cuda_ipc");
    }
    my_rank_ = rank;
    void* buf = nullptr;
    cudaError_t st = cudaMalloc(&buf, kMaxBytesPerRank);
    if (st != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMalloc IPC buf: ") +
                               cudaGetErrorString(st));
    }
    local_dev_buf_ = buf;
    IpcHandle handle{};
    st = cudaIpcGetMemHandle(&handle, local_dev_buf_);
    if (st != cudaSuccess) {
      cudaFree(local_dev_buf_);
      local_dev_buf_ = nullptr;
      throw std::runtime_error(std::string("cudaIpcGetMemHandle: ") +
                               cudaGetErrorString(st));
    }

    pthread_mutex_lock(&shm_->mu);
    *ipc_handle_ptr(rank) = handle;
    ++shm_->ipc_init_count;
    if (shm_->ipc_init_count == shm_->world) {
      pthread_cond_broadcast(&shm_->cv);
    } else {
      while (shm_->ipc_init_count < shm_->world) {
        pthread_cond_wait(&shm_->cv, &shm_->mu);
      }
    }
    pthread_mutex_unlock(&shm_->mu);

    remote_dev_ptrs_.assign(static_cast<size_t>(shm_->world), nullptr);
    remote_dev_ptrs_[static_cast<size_t>(rank)] = local_dev_buf_;
    for (int r = 0; r < shm_->world; ++r) {
      if (r == rank) continue;
      void* remote = nullptr;
      st = cudaIpcOpenMemHandle(&remote, *ipc_handle_ptr(r),
                               cudaIpcMemLazyEnablePeerAccess);
      if (st != cudaSuccess) {
        release_cuda_ipc();
        throw std::runtime_error(std::string("cudaIpcOpenMemHandle rank ") +
                                 std::to_string(r) + ": " +
                                 cudaGetErrorString(st));
      }
      remote_dev_ptrs_[static_cast<size_t>(r)] = remote;
    }
    ipc_ready_ = true;
    std::cerr << "SharedPeerGatherSlot: cuda_ipc ready rank=" << rank
              << " world=" << shm_->world << "\n";
#endif
  }

  void release_cuda_ipc() {
#if defined(UMA_ENGINE_USE_CUDA)
    if (!remote_dev_ptrs_.empty()) {
      for (int r = 0; r < static_cast<int>(remote_dev_ptrs_.size()); ++r) {
        if (r == my_rank_) continue;
        void* p = remote_dev_ptrs_[static_cast<size_t>(r)];
        if (p) {
          cudaIpcCloseMemHandle(p);
        }
      }
      remote_dev_ptrs_.clear();
    }
    if (local_dev_buf_) {
      cudaFree(local_dev_buf_);
      local_dev_buf_ = nullptr;
    }
#endif
    ipc_ready_ = false;
    my_rank_ = -1;
  }

  torch::Tensor all_gather_concat(int rank, const torch::Tensor& local,
                                  int64_t n_atoms) {
    auto parts = exchange_(rank, local);
    return all_gather_nodes(parts, n_atoms, local.device());
  }

  torch::Tensor all_reduce(int rank, const torch::Tensor& local) {
    auto parts = exchange_(rank, local);
    return all_reduce_sum(parts, local.device());
  }

  void barrier(int rank) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
#if defined(UMA_ENGINE_USE_CUDA)
    if (shm_ && shm_->transport == kTransportCudaIpc && torch::cuda::is_available()) {
      opts = opts.device(torch::Device(torch::kCUDA, 0));
    }
#endif
    auto dummy = torch::empty({0}, opts);
    (void)exchange_(rank, dummy);
  }

 private:
  SharedPeerGatherSlot(Shm* shm, size_t map_bytes, bool owns)
      : shm_(shm), map_bytes_(map_bytes), owns_(owns) {}

  static void init_sync_primitives_(Shm* shm) {
    pthread_mutexattr_t mattr;
    pthread_condattr_t cattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_condattr_init(&cattr);
    pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&shm->mu, &mattr);
    pthread_cond_init(&shm->cv, &cattr);
    pthread_mutexattr_destroy(&mattr);
    pthread_condattr_destroy(&cattr);
  }

  size_t rank_stride() const { return rank_stride_for(shm_->transport); }

  char* rank_base(int rank) {
    char* base = reinterpret_cast<char*>(shm_ + 1);
    return base + static_cast<size_t>(rank) * rank_stride();
  }

  int64_t* nbytes_ptr(int rank) {
    return reinterpret_cast<int64_t*>(rank_base(rank));
  }

  IpcHandle* ipc_handle_ptr(int rank) {
    return reinterpret_cast<IpcHandle*>(nbytes_ptr(rank) + 1);
  }

  char* payload_ptr(int rank) {
    return reinterpret_cast<char*>(nbytes_ptr(rank) + 1);
  }

  std::vector<torch::Tensor> exchange_(int rank, const torch::Tensor& local) {
    if (shm_->transport == kTransportCudaIpc) {
      return exchange_cuda_ipc_(rank, local);
    }
    return exchange_host_(rank, local);
  }

  torch::Tensor make_part_tensor_(const torch::Tensor& local, int64_t nb,
                                  torch::Device device) {
    const auto opts =
        torch::TensorOptions().dtype(local.scalar_type()).device(device);
    if (nb == 0) {
      return local.dim() == 0 ? torch::empty({}, opts) : torch::empty({0}, opts);
    }
    const size_t elem = local.element_size();
    if (elem == 0 || (nb % static_cast<int64_t>(elem)) != 0) {
      throw std::runtime_error("SharedPeerGatherSlot: nbytes not divisible by elem");
    }
    const int64_t nelem = nb / static_cast<int64_t>(elem);
    if (local.dim() == 0) {
      if (nelem != 1) {
        throw std::runtime_error("SharedPeerGatherSlot: 0-D payload nelem != 1");
      }
      return torch::empty({}, opts);
    }
    int64_t trailing = 1;
    for (int d = 1; d < local.dim(); ++d) trailing *= local.size(d);
    if (trailing < 1) trailing = 1;
    if (nelem % trailing != 0) {
      return torch::empty({nelem}, opts);
    }
    std::vector<int64_t> shape = local.sizes().vec();
    shape[0] = nelem / trailing;
    return torch::empty(shape, opts);
  }

  std::vector<torch::Tensor> exchange_host_(int rank, const torch::Tensor& local) {
    if (rank < 0 || rank >= shm_->world) {
      throw std::runtime_error("SharedPeerGatherSlot: bad rank");
    }
    auto cpu = local.detach().to(torch::kCPU).contiguous();
    const int64_t nbytes = static_cast<int64_t>(cpu.nbytes());
    if (static_cast<size_t>(nbytes) > kMaxBytesPerRank) {
      throw std::runtime_error("SharedPeerGatherSlot: payload exceeds kMaxBytesPerRank");
    }

    pthread_mutex_lock(&shm_->mu);
    const int gen = shm_->gen;
    *nbytes_ptr(rank) = nbytes;
    if (nbytes > 0) {
      std::memcpy(payload_ptr(rank), cpu.data_ptr(), static_cast<size_t>(nbytes));
    }
    ++shm_->nwrite;
    if (shm_->nwrite == shm_->world) {
      pthread_cond_broadcast(&shm_->cv);
    } else {
      while (shm_->nwrite < shm_->world && shm_->gen == gen) {
        pthread_cond_wait(&shm_->cv, &shm_->mu);
      }
    }
    if (shm_->gen != gen) {
      pthread_mutex_unlock(&shm_->mu);
      throw std::runtime_error("SharedPeerGatherSlot: gen advanced early");
    }

    std::vector<torch::Tensor> parts(static_cast<size_t>(shm_->world));
    try {
      for (int r = 0; r < shm_->world; ++r) {
        const int64_t nb = *nbytes_ptr(r);
        auto t = make_part_tensor_(local, nb, torch::kCPU);
        if (nb > 0) {
          std::memcpy(t.data_ptr(), payload_ptr(r), static_cast<size_t>(nb));
        }
        parts[static_cast<size_t>(r)] = t;
      }
    } catch (...) {
      pthread_mutex_unlock(&shm_->mu);
      throw;
    }

    ++shm_->nread;
    if (shm_->nread == shm_->world) {
      shm_->nwrite = 0;
      shm_->nread = 0;
      ++shm_->gen;
      pthread_cond_broadcast(&shm_->cv);
    } else {
      while (shm_->gen == gen) {
        pthread_cond_wait(&shm_->cv, &shm_->mu);
      }
    }
    pthread_mutex_unlock(&shm_->mu);
    return parts;
  }

  std::vector<torch::Tensor> exchange_cuda_ipc_(int rank,
                                                const torch::Tensor& local) {
#if !defined(UMA_ENGINE_USE_CUDA)
    (void)rank;
    (void)local;
    throw std::runtime_error("SharedPeerGatherSlot: cuda_ipc without CUDA");
#else
    if (!ipc_ready_ || !local_dev_buf_) {
      throw std::runtime_error(
          "SharedPeerGatherSlot: cuda_ipc not initialized (call init_cuda_ipc)");
    }
    if (rank < 0 || rank >= shm_->world) {
      throw std::runtime_error("SharedPeerGatherSlot: bad rank");
    }

    torch::Device dev(torch::kCUDA, 0);
    auto gpu = local.detach();
    if (!gpu.defined()) {
      gpu = torch::empty({0}, torch::dtype(torch::kFloat64).device(dev));
    } else if (!gpu.is_cuda()) {
      gpu = gpu.to(dev);
    }
    gpu = gpu.contiguous();
    const int64_t nbytes = static_cast<int64_t>(gpu.nbytes());
    if (static_cast<size_t>(nbytes) > kMaxBytesPerRank) {
      throw std::runtime_error("SharedPeerGatherSlot: payload exceeds kMaxBytesPerRank");
    }
    if (nbytes > 0) {
      cudaError_t st = cudaMemcpy(local_dev_buf_, gpu.data_ptr(),
                                  static_cast<size_t>(nbytes),
                                  cudaMemcpyDeviceToDevice);
      if (st != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy D2D send: ") +
                                 cudaGetErrorString(st));
      }
    }
    // Publish only after device write is complete.
    cudaDeviceSynchronize();

    pthread_mutex_lock(&shm_->mu);
    const int gen = shm_->gen;
    *nbytes_ptr(rank) = nbytes;
    ++shm_->nwrite;
    if (shm_->nwrite == shm_->world) {
      pthread_cond_broadcast(&shm_->cv);
    } else {
      while (shm_->nwrite < shm_->world && shm_->gen == gen) {
        pthread_cond_wait(&shm_->cv, &shm_->mu);
      }
    }
    if (shm_->gen != gen) {
      pthread_mutex_unlock(&shm_->mu);
      throw std::runtime_error("SharedPeerGatherSlot: gen advanced early");
    }
    // Snapshot nbytes; unlock so ranks can D2D in parallel. Send buffers stay
    // stable until all ranks finish nread and gen advances.
    std::vector<int64_t> nbs(static_cast<size_t>(shm_->world));
    for (int r = 0; r < shm_->world; ++r) nbs[static_cast<size_t>(r)] = *nbytes_ptr(r);
    pthread_mutex_unlock(&shm_->mu);

    std::vector<torch::Tensor> parts(static_cast<size_t>(shm_->world));
    for (int r = 0; r < shm_->world; ++r) {
      const int64_t nb = nbs[static_cast<size_t>(r)];
      auto t = make_part_tensor_(gpu, nb, dev);
      if (nb > 0) {
        void* src = remote_dev_ptrs_[static_cast<size_t>(r)];
        cudaError_t st = cudaMemcpy(t.data_ptr(), src, static_cast<size_t>(nb),
                                    cudaMemcpyDeviceToDevice);
        if (st != cudaSuccess) {
          throw std::runtime_error(std::string("cudaMemcpy D2D recv r=") +
                                   std::to_string(r) + ": " +
                                   cudaGetErrorString(st));
        }
      }
      parts[static_cast<size_t>(r)] = t;
    }
    cudaDeviceSynchronize();

    pthread_mutex_lock(&shm_->mu);
    if (shm_->gen != gen) {
      pthread_mutex_unlock(&shm_->mu);
      throw std::runtime_error("SharedPeerGatherSlot: gen advanced during D2D");
    }
    ++shm_->nread;
    if (shm_->nread == shm_->world) {
      shm_->nwrite = 0;
      shm_->nread = 0;
      ++shm_->gen;
      pthread_cond_broadcast(&shm_->cv);
    } else {
      while (shm_->gen == gen) {
        pthread_cond_wait(&shm_->cv, &shm_->mu);
      }
    }
    pthread_mutex_unlock(&shm_->mu);
    return parts;
#endif
  }

  Shm* shm_ = nullptr;
  size_t map_bytes_ = 0;
  bool owns_ = true;

  // Process-local CUDA IPC state (not in shm).
  void* local_dev_buf_ = nullptr;
  std::vector<void*> remote_dev_ptrs_;
  bool ipc_ready_ = false;
  int my_rank_ = -1;
};

}  // namespace kokkos_peer
}  // namespace uma
