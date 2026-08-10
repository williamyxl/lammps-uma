#pragma once

// Shared geometry/edge fan-out + rank0 force fan-in for LibTorch MP workers.
// Avoids multi-MB pipe copies of pos/z/edges/forces each predict.
//
// W5: cell_offsets travel as int32 image indices (cast to FP64 on the worker
// before TorchScript). Edge capacity is chosen at create time (max_e).
//
// W6: publish the *full* edge graph once; each worker shards on GPU. Layout
// stores a single eidx/coff region (not per-rank copies).

#include <algorithm>
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
  // W6: full-graph edge ceiling (was 512K per-rank; full publish needs headroom).
  static constexpr int64_t kMaxE = 2 * 1024 * 1024;

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
    int32_t n_edges_full;  // W6: full-graph edge count
    int32_t nedges[kMaxWorld];  // optional local counts (debug / legacy)
    int64_t max_e;  // W5/W6: full-graph edge capacity used to size this map
  };

  // Layout after Header:
  //   double  pos[kMaxN * 3]
  //   int64_t z[kMaxN]
  //   int64_t eidx_full[max_e * 2]     // W6: one full graph
  //   int32_t coff_full[max_e * 3]
  //   double  forces[kMaxN * 3]

  static int64_t choose_max_e(int64_t n_atoms, int max_neighbors) {
    // Full-graph capacity; clamp to product ceiling.
    const int64_t nn = std::max<int64_t>(1, max_neighbors > 0 ? max_neighbors : 300);
    const int64_t nat = std::max<int64_t>(1, n_atoms);
    const int64_t need = nat * nn * 2;  // directed edges worst-case-ish
    return std::min(kMaxE, std::max<int64_t>(need, 4096));
  }

  static size_t edge_bytes(int64_t max_e) {
    return sizeof(int64_t) * static_cast<size_t>(max_e) * 2 +
           sizeof(int32_t) * static_cast<size_t>(max_e) * 3;
  }

  static size_t map_bytes(int world_slots, int64_t max_e) {
    if (world_slots < 1 || world_slots > kMaxWorld) {
      throw std::runtime_error("PayloadShm: world_slots out of range");
    }
    if (max_e < 1 || max_e > kMaxE) {
      throw std::runtime_error("PayloadShm: max_e out of range");
    }
    size_t n = sizeof(Header);
    n += sizeof(double) * static_cast<size_t>(kMaxN) * 3;
    n += sizeof(int64_t) * static_cast<size_t>(kMaxN);
    n += edge_bytes(max_e);  // W6: single full-graph region
    n += sizeof(double) * static_cast<size_t>(kMaxN) * 3;
    return n;
  }

  // Back-compat helpers (full ceiling).
  static size_t map_bytes(int world_slots) { return map_bytes(world_slots, kMaxE); }
  static size_t map_bytes() { return map_bytes(kMaxWorld, kMaxE); }

  Header* hdr = nullptr;
  size_t bytes = 0;
  int world_slots = kMaxWorld;
  int64_t max_e = kMaxE;
  bool owns = false;

  static PayloadShm* create_file(const char* path, int world_slots, int64_t max_e) {
    const size_t nbytes = map_bytes(world_slots, max_e);
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
    h->max_e = max_e;
    h->world = world_slots;
    auto* p = new PayloadShm;
    p->hdr = h;
    p->bytes = nbytes;
    p->world_slots = world_slots;
    p->max_e = max_e;
    p->owns = true;
    return p;
  }

  static PayloadShm* create_file(const char* path, int world_slots) {
    return create_file(path, world_slots, kMaxE);
  }

  static PayloadShm* create_file(const char* path) {
    return create_file(path, kMaxWorld, kMaxE);
  }

  static PayloadShm* attach_file(const char* path, size_t expect_bytes,
                                 int world_slots, int64_t max_e) {
    int fd = ::open(path, O_RDWR);
    if (fd < 0) throw std::runtime_error("PayloadShm: attach open failed");
    void* mem = mmap(nullptr, expect_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mem == MAP_FAILED) throw std::runtime_error("PayloadShm: attach mmap failed");
    auto* p = new PayloadShm;
    p->hdr = static_cast<Header*>(mem);
    p->bytes = expect_bytes;
    p->world_slots = world_slots;
    p->max_e = (p->hdr->max_e > 0) ? p->hdr->max_e : max_e;
    p->owns = false;
    return p;
  }

  static PayloadShm* attach_file(const char* path, size_t expect_bytes,
                                 int world_slots) {
    return attach_file(path, expect_bytes, world_slots, kMaxE);
  }

  static PayloadShm* attach_file(const char* path, size_t expect_bytes) {
    // Parent passes exact byte size via UMA_MP_PAYLOAD_BYTES. Map that,
    // then trust Header::{world,max_e} written at create time.
    int fd = ::open(path, O_RDWR);
    if (fd < 0) throw std::runtime_error("PayloadShm: attach open failed");
    void* mem = mmap(nullptr, expect_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mem == MAP_FAILED) throw std::runtime_error("PayloadShm: attach mmap failed");
    auto* h = static_cast<Header*>(mem);
    const int world = (h->world >= 1 && h->world <= kMaxWorld) ? h->world : kMaxWorld;
    const int64_t max_e =
        (h->max_e >= 1 && h->max_e <= kMaxE) ? h->max_e : kMaxE;
    if (map_bytes(world, max_e) != expect_bytes) {
      munmap(mem, expect_bytes);
      throw std::runtime_error(
          "PayloadShm: attach size mismatch vs header world/max_e");
    }
    auto* p = new PayloadShm;
    p->hdr = h;
    p->bytes = expect_bytes;
    p->world_slots = world;
    p->max_e = max_e;
    p->owns = false;
    return p;
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
  int64_t* eidx_full_ptr() { return reinterpret_cast<int64_t*>(edges_base()); }
  int32_t* coff_full_ptr() {
    return reinterpret_cast<int32_t*>(edges_base() +
                                     sizeof(int64_t) * static_cast<size_t>(max_e) * 2);
  }
  double* forces_ptr() {
    return reinterpret_cast<double*>(edges_base() + edge_bytes(max_e));
  }
};

}  // namespace uma
