#pragma once

// Native oneCCL (XCCL) on-device peer collectives for same-node XPU tiles.
//
// This header is GCC-safe: it exposes ONLY torch tensors + POD. All SYCL /
// oneCCL types live in xccl_peer.cpp, which is compiled with icpx (the SYCL
// compiler). The rest of the engine (GCC) links against this opaque interface.
//
// Data path: ccl::allreduce / ccl::allgather on XPU device (USM) buffers using
// a ccl::communicator built from torch's current XPU SYCL device+context and a
// ccl::stream from torch's current XPU SYCL queue. No host staging.
//
// Bootstrap: rank 0 creates the oneCCL main KVS and MPI_Bcasts its address;
// other ranks create a KVS from that address. MPI is used ONLY for this
// one-time rendezvous, never for tensor data.

#include <memory>

#include <torch/torch.h>

namespace uma {
namespace kokkos_peer {

class XcclPeer {
 public:
  // Build the SYCL device/context/stream + ccl::communicator (KVS over MPI).
  // device_index is informational; the tile is pinned via ZE_AFFINITY_MASK so
  // torch's current XPU device is the rank's tile.
  static std::shared_ptr<XcclPeer> create(int rank, int world, int device_index);

  virtual ~XcclPeer() = default;

  // Sum-reduce `local` (XPU tensor, any dtype cast to fp64) across all ranks;
  // returns an XPU tensor of the same shape/dtype as `local`.
  virtual torch::Tensor all_reduce_sum(const torch::Tensor& local) = 0;

  // All-gather equal-sized `local` [pad, ...] along dim0 -> [world*pad, ...]
  // on the XPU device. Caller pads/unpads.
  virtual torch::Tensor all_gather(const torch::Tensor& local) = 0;

  virtual void barrier() = 0;

  virtual int rank() const = 0;
  virtual int world() const = 0;
};

// opt6-diag: read + reset per-call collective accumulators (UMA_PEER_PERF).
// Returns via out params: all_gather ms/count/bytes, all_reduce ms/count.
void peer_perf_read_reset(double& ag_ms, int& ag_n, double& ag_bytes,
                          double& ar_ms, int& ar_n);

}  // namespace kokkos_peer
}  // namespace uma
