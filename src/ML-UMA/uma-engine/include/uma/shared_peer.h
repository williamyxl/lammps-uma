#pragma once

// Process-shared peer gather/reduce for multi-process LibTorch MP.
// Threads cannot run concurrent jit::Module::forward (deadlock). Each GP rank
// is a process; control sync uses PTHREAD_PROCESS_SHARED.
//
// Transports (UMA_PEER_TRANSPORT=shm|cuda_ipc|nccl):
//   shm      — host-staged payload in mmap (legacy / fallback)
//   cuda_ipc — device payload via cudaIpcMemHandle_t; shm holds only control +
//              handles + nbytes. Explicit fallback when NCCL unavailable / forced.
//   nccl     — raw libnccl AllGather/AllReduce (product default when built with
//              UMA_ENGINE_USE_NCCL + CUDA). shm holds control + ncclUniqueId.

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
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif
#if defined(UMA_ENGINE_USE_NCCL)
#include <nccl.h>
#endif

namespace uma {
namespace kokkos_peer {

class SharedPeerGatherSlot {
 public:
  static constexpr size_t kMaxBytesPerRank = 256ull * 1024ull * 1024ull;
  static constexpr int kTransportShm = 0;
  static constexpr int kTransportCudaIpc = 1;
  static constexpr int kTransportNccl = 2;

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
    int transport;       // shm | cuda_ipc | nccl
    int ipc_init_count;  // ranks ready for cuda_ipc OR nccl bootstrap
    char nccl_id[128];   // ncclUniqueId bytes (rank0 publishes)
    // trailing per rank (see rank_stride):
    //   int64_t nbytes
    //   [cuda_ipc] IpcHandle
    //   [shm]      char payload[kMaxBytesPerRank]
    //   [nccl]     (nbytes unused)
  };

  static int select_transport() {
    if (const char* e = std::getenv("UMA_PEER_TRANSPORT")) {
      if (std::strcmp(e, "shm") == 0) return kTransportShm;
      if (std::strcmp(e, "cuda_ipc") == 0) return kTransportCudaIpc;
      if (std::strcmp(e, "nccl") == 0) {
#if defined(UMA_ENGINE_USE_NCCL)
        return kTransportNccl;
#else
        throw std::runtime_error(
            "UMA_PEER_TRANSPORT=nccl but uma-engine built without NCCL");
#endif
      }
    }
    // Product default: NCCL when built; else CUDA IPC; else host shm.
#if defined(UMA_ENGINE_USE_NCCL) && defined(UMA_ENGINE_USE_CUDA)
    if (torch::cuda::is_available()) return kTransportNccl;
#endif
#if defined(UMA_ENGINE_USE_CUDA)
    if (torch::cuda::is_available()) return kTransportCudaIpc;
#endif
    return kTransportShm;
  }

  static size_t rank_stride_for(int transport) {
    size_t n = sizeof(int64_t);
    if (transport == kTransportCudaIpc) {
      n += sizeof(IpcHandle);
    } else if (transport == kTransportNccl) {
      // control-only trailing; nbytes slots unused
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
    if (transport == kTransportCudaIpc) return "cuda_ipc";
    if (transport == kTransportNccl) return "nccl";
    return "shm";
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
    release_nccl();
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

  // Under the MPI bootstrap there is no shm segment, so fall back to the
  // world recorded at init_nccl_external time.
  int world() const { return shm_ ? shm_->world : external_world_; }

  // Worker-only: allocate CUDA send buffer, publish IPC handle, open remotes.
  // No-op for shm/nccl transport or if already initialized.
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
    if (!nccl_ready_) my_rank_ = -1;
  }

  // Multi-node: bootstrap ncclComm from an EXTERNALLY supplied unique id.
  //
  // The shm path below publishes ncclUniqueId through /dev/shm, which is
  // node-local: a rank on node B cannot read node A's shm file. MPI is the one
  // channel that exists across nodes before NCCL is up, so the caller
  // (MPI_Bcast of the id from rank 0) hands the id in here directly.
  //
  // Everything downstream -- the collectives, the dedicated NCCL stream, the
  // W1-W8 optimisations -- is unchanged. NCCL already spans nodes on its own:
  // ncclCommInitRank picks NVLink intra-node and IB inter-node transparently.
  //
  // device_index: the GPU this rank owns. The shm path assumes cudaSetDevice(0)
  // because each forked worker is exec'd with CUDA_VISIBLE_DEVICES pinned to a
  // single GPU. Under MPI the rank sees all local GPUs, so the index must be
  // passed explicitly or every rank on a node would bind GPU 0.
  void init_nccl_external(int rank, int world, const void *unique_id,
                          int device_index) {
#if !defined(UMA_ENGINE_USE_NCCL)
    (void)rank; (void)world; (void)unique_id; (void)device_index;
    throw std::runtime_error("SharedPeerGatherSlot: nccl requested without NCCL");
#else
    if (nccl_ready_) return;
    if (rank < 0 || world < 1 || rank >= world) {
      throw std::runtime_error("init_nccl_external: bad rank/world");
    }
    if (unique_id == nullptr) {
      throw std::runtime_error("init_nccl_external: null ncclUniqueId");
    }
    my_rank_ = rank;
    external_world_ = world;
#if defined(UMA_ENGINE_USE_CUDA)
    if (device_index >= 0) cudaSetDevice(device_index);
#endif
    ncclUniqueId id;
    std::memcpy(&id, unique_id, sizeof(id));
    ncclResult_t nr = ncclCommInitRank(&comm_, world, id, rank);
    if (nr != ncclSuccess) {
      throw std::runtime_error(std::string("init_nccl_external ncclCommInitRank: ") +
                               ncclGetErrorString(nr));
    }
#if defined(UMA_ENGINE_USE_CUDA)
    // Same W8 setup as the shm path: dedicated NCCL stream + events so the
    // Torch default stream orders against it. Reused verbatim so the W1-W8
    // optimisations behave identically under MPI.
    if (cudaStreamCreateWithFlags(&nccl_stream_, cudaStreamNonBlocking) !=
        cudaSuccess) {
      throw std::runtime_error("cudaStreamCreate(nccl_stream) failed");
    }
    if (cudaEventCreateWithFlags(&nccl_done_evt_, cudaEventDisableTiming) !=
        cudaSuccess) {
      throw std::runtime_error("cudaEventCreate(nccl_done_evt) failed");
    }
    if (cudaEventCreateWithFlags(&torch_done_evt_, cudaEventDisableTiming) !=
        cudaSuccess) {
      throw std::runtime_error("cudaEventCreate(torch_done_evt) failed");
    }
#endif
    nccl_ready_ = true;
    std::cerr << "SharedPeerGatherSlot: nccl ready (mpi bootstrap) rank="
              << rank << " world=" << world << " dev=" << device_index << "\n";
#endif
  }

  // Size of the opaque id the caller must broadcast (ncclUniqueId).
  static size_t unique_id_bytes() {
#if defined(UMA_ENGINE_USE_NCCL)
    return sizeof(ncclUniqueId);
#else
    return 0;
#endif
  }

  // Rank 0 only: generate the id to broadcast. Caller owns the buffer.
  static void make_unique_id(void *out) {
#if !defined(UMA_ENGINE_USE_NCCL)
    (void)out;
    throw std::runtime_error("make_unique_id: built without NCCL");
#else
    ncclUniqueId id;
    ncclResult_t nr = ncclGetUniqueId(&id);
    if (nr != ncclSuccess) {
      throw std::runtime_error(std::string("ncclGetUniqueId: ") +
                               ncclGetErrorString(nr));
    }
    std::memcpy(out, &id, sizeof(id));
#endif
  }

  // Worker-only: bootstrap ncclComm from shared unique id. No-op unless nccl.
  void init_nccl(int rank) {
    if (!shm_ || shm_->transport != kTransportNccl) return;
    if (nccl_ready_) return;
#if !defined(UMA_ENGINE_USE_NCCL)
    throw std::runtime_error("SharedPeerGatherSlot: nccl requested without NCCL");
#else
    if (rank < 0 || rank >= shm_->world) {
      throw std::runtime_error("SharedPeerGatherSlot: bad rank for init_nccl");
    }
    my_rank_ = rank;
#if defined(UMA_ENGINE_USE_CUDA)
    cudaSetDevice(0);
#endif
    ncclUniqueId id;
    std::memset(&id, 0, sizeof(id));
    static_assert(sizeof(ncclUniqueId) <= 128, "ncclUniqueId larger than Shm::nccl_id");

    pthread_mutex_lock(&shm_->mu);
    if (rank == 0) {
      ncclResult_t nr = ncclGetUniqueId(&id);
      if (nr != ncclSuccess) {
        pthread_mutex_unlock(&shm_->mu);
        throw std::runtime_error(std::string("ncclGetUniqueId: ") +
                                 ncclGetErrorString(nr));
      }
      std::memcpy(shm_->nccl_id, &id, sizeof(id));
    }
    ++shm_->ipc_init_count;
    if (shm_->ipc_init_count == shm_->world) {
      pthread_cond_broadcast(&shm_->cv);
    } else {
      while (shm_->ipc_init_count < shm_->world) {
        pthread_cond_wait(&shm_->cv, &shm_->mu);
      }
    }
    std::memcpy(&id, shm_->nccl_id, sizeof(id));
    pthread_mutex_unlock(&shm_->mu);

    ncclResult_t nr =
        ncclCommInitRank(&comm_, shm_->world, id, rank);
    if (nr != ncclSuccess) {
      throw std::runtime_error(std::string("ncclCommInitRank: ") +
                               ncclGetErrorString(nr));
    }
#if defined(UMA_ENGINE_USE_CUDA)
    // Tier2/W8: dedicated NCCL stream; Torch default stream waits via event.
    if (cudaStreamCreateWithFlags(&nccl_stream_, cudaStreamNonBlocking) !=
        cudaSuccess) {
      throw std::runtime_error("cudaStreamCreate(nccl_stream) failed");
    }
    if (cudaEventCreateWithFlags(&nccl_done_evt_, cudaEventDisableTiming) !=
        cudaSuccess) {
      throw std::runtime_error("cudaEventCreate(nccl_done_evt) failed");
    }
    if (cudaEventCreateWithFlags(&torch_done_evt_, cudaEventDisableTiming) !=
        cudaSuccess) {
      throw std::runtime_error("cudaEventCreate(torch_done_evt) failed");
    }
#endif
    nccl_ready_ = true;
    std::cerr << "SharedPeerGatherSlot: nccl ready rank=" << rank
              << " world=" << shm_->world << " stream=dedicated\n";
#endif
  }

  void release_nccl() {
#if defined(UMA_ENGINE_USE_NCCL)
    if (comm_ && shm_) {
      // All ranks must enter destroy together; parent broadcasts shutdown first.
      pthread_mutex_lock(&shm_->mu);
      const int gen = shm_->gen;
      ++shm_->nwrite;
      if (shm_->nwrite == shm_->world) {
        pthread_cond_broadcast(&shm_->cv);
      } else {
        while (shm_->nwrite < shm_->world && shm_->gen == gen) {
          pthread_cond_wait(&shm_->cv, &shm_->mu);
        }
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
      ncclCommDestroy(comm_);
      comm_ = nullptr;
    } else if (comm_) {
      ncclCommDestroy(comm_);
      comm_ = nullptr;
    }
#if defined(UMA_ENGINE_USE_CUDA)
    if (nccl_done_evt_) {
      cudaEventDestroy(nccl_done_evt_);
      nccl_done_evt_ = nullptr;
    }
    if (torch_done_evt_) {
      cudaEventDestroy(torch_done_evt_);
      torch_done_evt_ = nullptr;
    }
    if (nccl_stream_) {
      cudaStreamSynchronize(nccl_stream_);
      cudaStreamDestroy(nccl_stream_);
      nccl_stream_ = nullptr;
    }
#endif
#endif
    nccl_ready_ = false;
    if (!ipc_ready_) my_rank_ = -1;
  }

  torch::Tensor all_gather_concat(int rank, const torch::Tensor& local,
                                  int64_t n_atoms) {
#if defined(UMA_ENGINE_USE_NCCL)
    if (shm_ && shm_->transport == kTransportNccl) {
      return all_gather_nccl_(rank, local, n_atoms);
    }
#endif
    auto parts = exchange_(rank, local);
    return all_gather_nodes(parts, n_atoms, local.device());
  }

  torch::Tensor all_reduce(int rank, const torch::Tensor& local) {
#if defined(UMA_ENGINE_USE_NCCL)
    if (shm_ && shm_->transport == kTransportNccl) {
      return all_reduce_nccl_(rank, local);
    }
#endif
    auto parts = exchange_(rank, local);
    return all_reduce_sum(parts, local.device());
  }

  void barrier(int rank) {
    if (shm_ && shm_->transport == kTransportNccl) {
#if defined(UMA_ENGINE_USE_NCCL) && defined(UMA_ENGINE_USE_CUDA)
      auto t = torch::zeros({1}, torch::dtype(torch::kFloat64)
                                     .device(torch::Device(torch::kCUDA, 0)));
      (void)all_reduce_nccl_(rank, t);
      return;
#else
      (void)rank;
      throw std::runtime_error("SharedPeerGatherSlot: nccl barrier without NCCL/CUDA");
#endif
    }
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

#if defined(UMA_ENGINE_USE_NCCL)
  static ncclDataType_t nccl_dtype_(c10::ScalarType t) {
    switch (t) {
      case torch::kFloat64:
        return ncclDouble;
      case torch::kFloat32:
        return ncclFloat;
      case torch::kFloat16:
        return ncclHalf;
      case torch::kInt64:
        return ncclInt64;
      case torch::kInt32:
        return ncclInt32;
      default:
        throw std::runtime_error("SharedPeerGatherSlot: unsupported NCCL dtype");
    }
  }

  torch::Tensor ensure_cuda_(const torch::Tensor& local) const {
    torch::Device dev(torch::kCUDA, 0);
    auto gpu = local.detach();
    if (!gpu.defined()) {
      return torch::empty({0}, torch::dtype(torch::kFloat64).device(dev));
    }
    if (!gpu.is_cuda()) gpu = gpu.to(dev);
    return gpu.contiguous();
  }

  // W17: when UMA_CUDA_GRAPH=1, NCCL runs on Torch's current stream so capture
  // can include collectives (no side-stream event waits outside the graph).
  static bool nccl_on_current_stream_() {
    const char* e = std::getenv("UMA_CUDA_GRAPH");
    return e && std::string(e) == "1";
  }

  // Tier2/W8: NCCL on dedicated stream. Order: current Torch stream → NCCL →
  // current. Uses getCurrentCUDAStream (not cudaStreamDefault) so CUDAStreamGuard
  // during graph capture/replay stays consistent.
  void nccl_precede_from_default_() {
#if defined(UMA_ENGINE_USE_CUDA)
    if (!nccl_stream_ || !torch_done_evt_) return;
    if (nccl_on_current_stream_()) return;
    const cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    cudaEventRecord(torch_done_evt_, s);
    cudaStreamWaitEvent(nccl_stream_, torch_done_evt_, 0);
#endif
  }

  void nccl_join_default_stream_() {
#if defined(UMA_ENGINE_USE_CUDA)
    if (!nccl_stream_ || !nccl_done_evt_) return;
    if (nccl_on_current_stream_()) return;
    const cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    cudaEventRecord(nccl_done_evt_, nccl_stream_);
    cudaStreamWaitEvent(s, nccl_done_evt_, 0);
#endif
  }

  cudaStream_t nccl_cuda_stream_() const {
#if defined(UMA_ENGINE_USE_CUDA)
    if (nccl_on_current_stream_()) {
      return at::cuda::getCurrentCUDAStream().stream();
    }
    return nccl_stream_ ? nccl_stream_ : cudaStreamDefault;
#else
    return cudaStreamDefault;
#endif
  }

  torch::Tensor all_gather_nccl_(int rank, const torch::Tensor& local,
                                 int64_t n_atoms) {
    (void)rank;
    if (!nccl_ready_ || !comm_) {
      throw std::runtime_error(
          "SharedPeerGatherSlot: nccl not initialized (call init_nccl)");
    }
    auto gpu = ensure_cuda_(local);
    const int world = shm_->world;
    if (world == 1) {
      auto sizes = size_list(n_atoms, 1);
      if (gpu.dim() == 0) return gpu;
      return gpu.narrow(0, 0, sizes[0]).contiguous();
    }
    if (gpu.numel() == 0) {
      barrier(rank);
      auto opts = torch::TensorOptions().dtype(gpu.scalar_type()).device(gpu.device());
      return torch::empty({0}, opts);
    }
    const int64_t n_local0 = gpu.dim() == 0 ? 1 : gpu.size(0);
    std::vector<int64_t> out_sizes = gpu.sizes().vec();
    if (gpu.dim() == 0) {
      out_sizes = {static_cast<int64_t>(world)};
    } else {
      out_sizes[0] = n_local0 * world;
    }
    auto gathered = torch::empty(out_sizes, gpu.options());
    const size_t sendcount = static_cast<size_t>(gpu.numel());
    nccl_precede_from_default_();
    ncclResult_t nr =
        ncclAllGather(gpu.data_ptr(), gathered.data_ptr(), sendcount,
                      nccl_dtype_(gpu.scalar_type()), comm_, nccl_cuda_stream_());
    if (nr != ncclSuccess) {
      throw std::runtime_error(std::string("ncclAllGather: ") +
                               ncclGetErrorString(nr));
    }
    nccl_join_default_stream_();
    if (gpu.dim() == 0) {
      return gathered;
    }
    auto sizes = size_list(n_atoms, world);
    bool equal = true;
    for (int r = 1; r < world; ++r) {
      if (sizes[static_cast<size_t>(r)] != sizes[0]) {
        equal = false;
        break;
      }
    }
    if (equal && sizes[0] == n_local0) {
      return gathered;
    }
    std::vector<torch::Tensor> parts;
    parts.reserve(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r) {
      const auto want = sizes[static_cast<size_t>(r)];
      parts.push_back(
          gathered.narrow(0, r * n_local0, want).contiguous());
    }
    return torch::cat(parts, /*dim=*/0).contiguous();
  }

  torch::Tensor all_reduce_nccl_(int rank, const torch::Tensor& local) {
    (void)rank;
    if (!nccl_ready_ || !comm_) {
      throw std::runtime_error(
          "SharedPeerGatherSlot: nccl not initialized (call init_nccl)");
    }
    auto gpu = ensure_cuda_(local);
    if (gpu.numel() == 0) {
      auto t = torch::zeros({1}, torch::dtype(torch::kFloat64).device(gpu.device()));
      auto out = torch::empty_like(t);
      nccl_precede_from_default_();
      ncclResult_t nr =
          ncclAllReduce(t.data_ptr(), out.data_ptr(), 1, ncclDouble, ncclSum,
                        comm_, nccl_cuda_stream_());
      if (nr != ncclSuccess) {
        throw std::runtime_error(std::string("ncclAllReduce(empty sync): ") +
                                 ncclGetErrorString(nr));
      }
      nccl_join_default_stream_();
      return gpu;
    }
    auto out = torch::empty_like(gpu);
    nccl_precede_from_default_();
    ncclResult_t nr =
        ncclAllReduce(gpu.data_ptr(), out.data_ptr(),
                      static_cast<size_t>(gpu.numel()),
                      nccl_dtype_(gpu.scalar_type()), ncclSum, comm_,
                      nccl_cuda_stream_());
    if (nr != ncclSuccess) {
      throw std::runtime_error(std::string("ncclAllReduce: ") +
                               ncclGetErrorString(nr));
    }
    nccl_join_default_stream_();
    return out;
  }
#endif  // UMA_ENGINE_USE_NCCL

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
  // World size when bootstrapped over MPI (no shm segment to read it from).
  int external_world_ = 0;

  // Process-local NCCL state.
#if defined(UMA_ENGINE_USE_NCCL)
  ncclComm_t comm_ = nullptr;
#if defined(UMA_ENGINE_USE_CUDA)
  cudaStream_t nccl_stream_ = nullptr;
  cudaEvent_t nccl_done_evt_ = nullptr;
  cudaEvent_t torch_done_evt_ = nullptr;
#endif
#endif
  bool nccl_ready_ = false;
};

}  // namespace kokkos_peer
}  // namespace uma
