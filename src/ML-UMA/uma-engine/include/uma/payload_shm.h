#pragma once

// Shared geometry/edge fan-out + rank0 force fan-in for LibTorch MP workers.
// Avoids multi-MB pipe copies of pos/z/edges/forces each predict.

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

namespace uma {

struct PayloadShm {
  // LIM (perf campaign): single-node product caps. Raising either requires
  // resizing the fixed mmap layout + revalidation. Multi-node is out of scope.
  static constexpr int kMaxWorld = 8;
  static constexpr int64_t kMaxN = 8192;
  static constexpr int64_t kMaxE = 512 * 1024;  // edges per rank

  struct Header {
    pthread_mutex_t mu;
    int32_t gen;
    int32_t n;
    int32_t world;
    int64_t charge;
    int64_t spin;
    double cell[9];
    int32_t pbc[3];
    double energy;       // rank0 result
    int32_t result_gen;  // matches gen when forces valid
    int32_t nedges[kMaxWorld];
  };

  // Layout after Header (world_slots = actual GPU count, ≤ kMaxWorld):
  //   double pos[kMaxN * 3]
  //   int64_t z[kMaxN]
  //   per rank r < world_slots:
  //     int64_t eidx[kMaxE * 2]
  //     double  coff[kMaxE * 3]
  //   double forces[kMaxN * 3]

  static size_t map_bytes(int world_slots) {
    if (world_slots < 1 || world_slots > kMaxWorld) {
      throw std::runtime_error("PayloadShm: world_slots out of range");
    }
    size_t n = sizeof(Header);
    n += sizeof(double) * static_cast<size_t>(kMaxN) * 3;
    n += sizeof(int64_t) * static_cast<size_t>(kMaxN);
    n += static_cast<size_t>(world_slots) * rank_edge_stride();
    n += sizeof(double) * static_cast<size_t>(kMaxN) * 3;
    return n;
  }

  // Back-compat helper (full kMaxWorld footprint).
  static size_t map_bytes() { return map_bytes(kMaxWorld); }

  Header* hdr = nullptr;
  size_t bytes = 0;
  int world_slots = kMaxWorld;
  bool owns = false;

  static PayloadShm* create_file(const char* path, int world_slots) {
    const size_t nbytes = map_bytes(world_slots);
    int fd = ::open(path, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0) throw std::runtime_error("PayloadShm: open failed");
    if (ftruncate(fd, static_cast<off_t>(nbytes)) != 0) {
      close(fd);
      throw std::runtime_error("PayloadShm: ftruncate failed");
    }
    void* mem = mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mem == MAP_FAILED) throw std::runtime_error("PayloadShm: mmap failed");
    std::memset(mem, 0, nbytes);
    auto* h = static_cast<Header*>(mem);
    pthread_mutexattr_t mattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&h->mu, &mattr);
    pthread_mutexattr_destroy(&mattr);
    auto* p = new PayloadShm;
    p->hdr = h;
    p->bytes = nbytes;
    p->world_slots = world_slots;
    p->owns = true;
    return p;
  }

  static PayloadShm* create_file(const char* path) {
    return create_file(path, kMaxWorld);
  }

  static PayloadShm* attach_file(const char* path, size_t expect_bytes,
                                 int world_slots) {
    int fd = ::open(path, O_RDWR);
    if (fd < 0) throw std::runtime_error("PayloadShm: attach open failed");
    void* mem = mmap(nullptr, expect_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mem == MAP_FAILED) throw std::runtime_error("PayloadShm: attach mmap failed");
    auto* p = new PayloadShm;
    p->hdr = static_cast<Header*>(mem);
    p->bytes = expect_bytes;
    p->world_slots = world_slots;
    p->owns = false;
    return p;
  }

  static PayloadShm* attach_file(const char* path, size_t expect_bytes) {
    // Infer world_slots from byte size (V2b sized maps).
    int slots = kMaxWorld;
    for (int w = 1; w <= kMaxWorld; ++w) {
      if (map_bytes(w) == expect_bytes) {
        slots = w;
        break;
      }
    }
    return attach_file(path, expect_bytes, slots);
  }

  void destroy() {
    if (!hdr) {
      delete this;
      return;
    }
    if (owns) {
      pthread_mutex_destroy(&hdr->mu);
    }
    munmap(hdr, bytes);
    hdr = nullptr;
    delete this;
  }

  char* base() { return reinterpret_cast<char*>(hdr + 1); }

  double* pos_ptr() { return reinterpret_cast<double*>(base()); }
  int64_t* z_ptr() {
    return reinterpret_cast<int64_t*>(base() + sizeof(double) * kMaxN * 3);
  }
  char* edges_base() {
    return base() + sizeof(double) * kMaxN * 3 + sizeof(int64_t) * kMaxN;
  }
  static size_t rank_edge_stride() {
    return sizeof(int64_t) * static_cast<size_t>(kMaxE) * 2 +
           sizeof(double) * static_cast<size_t>(kMaxE) * 3;
  }
  int64_t* eidx_ptr(int rank) {
    return reinterpret_cast<int64_t*>(edges_base() +
                                     static_cast<size_t>(rank) * rank_edge_stride());
  }
  double* coff_ptr(int rank) {
    return reinterpret_cast<double*>(reinterpret_cast<char*>(eidx_ptr(rank)) +
                                     sizeof(int64_t) * static_cast<size_t>(kMaxE) * 2);
  }
  double* forces_ptr() {
    return reinterpret_cast<double*>(
        edges_base() + static_cast<size_t>(world_slots) * rank_edge_stride());
  }
};

}  // namespace uma
