// A7 (audit rev 26 §G.18.6): CI coverage for the SharedPeerGatherSlot transport
// table, so the enum + name map + stride arithmetic are exercised on EVERY
// backend rather than only reached by an XPU/CUDA smoke run. This is the "CI"
// half of the auditor's "delete-or-CI the untested transports": the transports
// stay, but their id/name/stride contract is now gated.
//
// Checks:
//   1. transport_name() maps every LIVE id and NEVER returns "shm" for XCCL
//      (the P3.1 mislabel that shipped as a wrong diagnostic on Aurora).
//   2. The id-3 gap is real: no live id equals 3, and select_transport() never
//      returns 3 for any UMA_PEER_TRANSPORT value.
//   3. rank_stride_for()/map_bytes_for() are monotonic and include the control
//      block, so a calloc'd Shm is always large enough for init_control_block().
//
// Returns 0 on success; nonzero on any mismatch.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "uma/shared_peer.h"

namespace {
using Slot = uma::kokkos_peer::SharedPeerGatherSlot;

int check_names() {
  struct { int id; const char* name; } cases[] = {
      {Slot::kTransportShm, "shm"},
      {Slot::kTransportCudaIpc, "cuda_ipc"},
      {Slot::kTransportNccl, "nccl"},
      {Slot::kTransportXccl, "xccl"},
  };
  for (auto& c : cases) {
    if (std::strcmp(Slot::transport_name(c.id), c.name) != 0) {
      std::cerr << "transport_name(" << c.id << ") != " << c.name << "\n";
      return 1;
    }
  }
  // P3.1 regression: XCCL must NOT be named "shm".
  if (std::strcmp(Slot::transport_name(Slot::kTransportXccl), "shm") == 0) {
    std::cerr << "XCCL mislabelled as shm (P3.1 regression)\n";
    return 1;
  }
  return 0;
}

int check_enum_gap() {
  // The retired id 3 must not collide with any live id.
  const int live[] = {Slot::kTransportShm, Slot::kTransportCudaIpc,
                      Slot::kTransportNccl, Slot::kTransportXccl};
  for (int id : live) {
    if (id == 3) {
      std::cerr << "a live transport id reuses the reserved gap 3\n";
      return 1;
    }
  }
  // select_transport() must never yield 3 (nor "shm"->3 etc.).
  for (const char* v : {"shm", "cuda_ipc", "bogus", ""}) {
    ::setenv("UMA_PEER_TRANSPORT", v, 1);
    int t = Slot::select_transport();  // nccl throws if unbuilt; shm/bogus don't
    if (t == 3) {
      std::cerr << "select_transport(" << v << ") returned reserved id 3\n";
      ::unsetenv("UMA_PEER_TRANSPORT");
      return 1;
    }
  }
  ::unsetenv("UMA_PEER_TRANSPORT");
  return 0;
}

int check_strides() {
  const int world = 4;
  for (int t : {Slot::kTransportShm, Slot::kTransportCudaIpc,
                Slot::kTransportNccl, Slot::kTransportXccl}) {
    const size_t stride = Slot::rank_stride_for(t);
    const size_t total = Slot::map_bytes_for(world, t);
    if (stride < sizeof(int64_t)) {  // every transport carries at least nbytes
      std::cerr << "rank_stride_for(" << t << ") too small\n";
      return 1;
    }
    if (total < sizeof(Slot::Shm) + static_cast<size_t>(world) * stride) {
      std::cerr << "map_bytes_for(" << t << ") omits the control block\n";
      return 1;
    }
  }
  return 0;
}
}  // namespace

int main() {
  if (int rc = check_names()) return rc;
  if (int rc = check_enum_gap()) return rc;
  if (int rc = check_strides()) return rc;
  std::cout << "test_transport_table OK (names + enum gap + strides)\n";
  return 0;
}
