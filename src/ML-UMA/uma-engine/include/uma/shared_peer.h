#pragma once

// Process-shared peer gather/reduce for multi-process LibTorch MP.
// Threads cannot run concurrent jit::Module::forward (deadlock). Each GP rank
// is a process; collectives use PTHREAD_PROCESS_SHARED + host-staged bytes.

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#include <torch/torch.h>

#include "uma/kokkos_peer.h"

namespace uma {
namespace kokkos_peer {

class SharedPeerGatherSlot {
 public:
  static constexpr size_t kMaxBytesPerRank = 256ull * 1024ull * 1024ull;

  struct Shm {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int gen;
    int nwrite;
    int nread;
    int world;
    // trailing: world * (int64 nbytes + kMaxBytesPerRank bytes)
  };

  static SharedPeerGatherSlot* create(int world) {
    if (world < 1) throw std::runtime_error("SharedPeerGatherSlot: world < 1");
    const size_t trail =
        static_cast<size_t>(world) * (sizeof(int64_t) + kMaxBytesPerRank);
    const size_t bytes = sizeof(Shm) + trail;
    void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
      throw std::runtime_error("SharedPeerGatherSlot: mmap failed");
    }
    auto* shm = static_cast<Shm*>(mem);
    std::memset(shm, 0, bytes);
    shm->world = world;
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
    return new SharedPeerGatherSlot(shm, bytes, /*owns=*/true);
  }

  // Map an existing memfd / inherited fd (worker attach; does not own unmap).
  static SharedPeerGatherSlot* attach(Shm* shm, size_t map_bytes) {
    return new SharedPeerGatherSlot(shm, map_bytes, /*owns=*/false);
  }

  size_t map_bytes() const { return map_bytes_; }
  Shm* raw() { return shm_; }

  void destroy() {
    if (!shm_) return;
    if (owns_) {
      pthread_mutex_destroy(&shm_->mu);
      pthread_cond_destroy(&shm_->cv);
      munmap(shm_, map_bytes_);
    }
    shm_ = nullptr;
    delete this;
  }

  int world() const { return shm_->world; }

  torch::Tensor all_gather_concat(int rank, const torch::Tensor& local,
                                  int64_t n_atoms) {
    auto parts = exchange_host_(rank, local);
    return all_gather_nodes(parts, n_atoms, local.device());
  }

  torch::Tensor all_reduce(int rank, const torch::Tensor& local) {
    auto parts = exchange_host_(rank, local);
    return all_reduce_sum(parts, local.device());
  }

  void barrier(int rank) {
    auto dummy = torch::empty({0}, torch::kFloat64);
    (void)exchange_host_(rank, dummy);
  }

 private:
  SharedPeerGatherSlot(Shm* shm, size_t map_bytes, bool owns)
      : shm_(shm), map_bytes_(map_bytes), owns_(owns) {}

  int64_t* nbytes_ptr(int rank) {
    char* base = reinterpret_cast<char*>(shm_ + 1);
    return reinterpret_cast<int64_t*>(
        base + static_cast<size_t>(rank) * (sizeof(int64_t) + kMaxBytesPerRank));
  }

  char* payload_ptr(int rank) {
    return reinterpret_cast<char*>(nbytes_ptr(rank) + 1);
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

    // Snapshot all ranks' host bytes into tensors matching local dtype/shape.
    std::vector<torch::Tensor> parts(static_cast<size_t>(shm_->world));
    const auto opts = torch::TensorOptions().dtype(local.scalar_type()).device(torch::kCPU);
    const size_t elem = local.element_size();
    for (int r = 0; r < shm_->world; ++r) {
      const int64_t nb = *nbytes_ptr(r);
      if (nb == 0) {
        parts[static_cast<size_t>(r)] =
            local.dim() == 0 ? torch::empty({}, opts) : torch::empty({0}, opts);
        continue;
      }
      if (elem == 0 || (nb % static_cast<int64_t>(elem)) != 0) {
        pthread_mutex_unlock(&shm_->mu);
        throw std::runtime_error("SharedPeerGatherSlot: nbytes not divisible by elem");
      }
      const int64_t nelem = nb / static_cast<int64_t>(elem);
      torch::Tensor t;
      if (local.dim() == 0) {
        // 0-D scalar: nbytes must be one element.
        if (nelem != 1) {
          pthread_mutex_unlock(&shm_->mu);
          throw std::runtime_error("SharedPeerGatherSlot: 0-D payload nelem != 1");
        }
        t = torch::empty({}, opts);
      } else {
        int64_t trailing = 1;
        for (int d = 1; d < local.dim(); ++d) trailing *= local.size(d);
        if (trailing < 1) trailing = 1;
        if (nelem % trailing != 0) {
          t = torch::empty({nelem}, opts);
        } else {
          std::vector<int64_t> shape = local.sizes().vec();
          shape[0] = nelem / trailing;
          t = torch::empty(shape, opts);
        }
      }
      std::memcpy(t.data_ptr(), payload_ptr(r), static_cast<size_t>(nb));
      parts[static_cast<size_t>(r)] = t;
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

  Shm* shm_ = nullptr;
  size_t map_bytes_ = 0;
  bool owns_ = true;
};

}  // namespace kokkos_peer
}  // namespace uma
