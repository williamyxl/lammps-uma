// Native oneCCL (XCCL) on-device peer collectives for same-node XPU tiles.
//
// COMPILED WITH icpx (SYCL). All SYCL / oneCCL types are confined to this TU.
// The rest of the engine (GCC) sees only the opaque uma::kokkos_peer::XcclPeer
// interface in xccl_peer.h.
//
// Data path: ccl::allreduce / ccl::allgather on XPU USM device buffers using a
// ccl::communicator built from torch's current XPU SYCL device+context and a
// ccl::stream from torch's current XPU SYCL queue. No host staging.
//
// KVS bootstrap over MPI: rank 0 create_main_kvs -> MPI_Bcast address ->
// others create_kvs(address). MPI touches only the KVS address, never data.

#include "uma/xccl_peer.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>

#include <c10/xpu/XPUStream.h>

namespace uma {
namespace kokkos_peer {

// opt6-diag: per-rank accumulators for collective cost (UMA_PEER_PERF=1). The
// caller (mpi_peer_predictor) reads+resets these each force call to see how much
// of fwd/bwd is all_gather vs all_reduce vs compute. Wall time incl. the .wait().
double g_ag_ms = 0.0; int g_ag_n = 0; double g_ag_bytes = 0.0;
double g_ar_ms = 0.0; int g_ar_n = 0;

namespace {
inline double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
ccl::datatype ccl_dtype(c10::ScalarType t) {
  switch (t) {
    case torch::kFloat64: return ccl::datatype::float64;
    case torch::kFloat32: return ccl::datatype::float32;
    case torch::kInt64:   return ccl::datatype::int64;
    case torch::kInt32:   return ccl::datatype::int32;
    default:
      throw std::runtime_error("XcclPeer: unsupported dtype for oneCCL");
  }
}
}  // namespace

class XcclPeerImpl final : public XcclPeer {
 public:
  XcclPeerImpl(int rank, int world, int device_index)
      : rank_(rank), world_(world) {
    (void)device_index;
    static std::once_flag ccl_init_once;
    std::call_once(ccl_init_once, [] { ccl::init(); });

    // torch's current XPU SYCL queue = this rank's tile (pinned via
    // ZE_AFFINITY_MASK, so device 0 in the masked view).
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    sycl_dev_ = q.get_device();
    sycl_ctx_ = q.get_context();

    auto dev = ccl::create_device(sycl_dev_);
    auto ctx = ccl::create_context(sycl_ctx_);

    // KVS rendezvous over MPI (address only).
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type addr;
    if (rank_ == 0) {
      kvs = ccl::create_main_kvs();
      addr = kvs->get_address();
      MPI_Bcast(addr.data(), static_cast<int>(addr.size()), MPI_BYTE, 0,
                MPI_COMM_WORLD);
    } else {
      MPI_Bcast(addr.data(), static_cast<int>(addr.size()), MPI_BYTE, 0,
                MPI_COMM_WORLD);
      kvs = ccl::create_kvs(addr);
    }

    comm_ = std::make_unique<ccl::communicator>(
        ccl::create_communicator(world_, rank_, dev, ctx, kvs));
    stream_ = std::make_unique<ccl::stream>(ccl::create_stream(q));
    queue_ = &q;
  }

  ~XcclPeerImpl() override = default;

  torch::Tensor all_reduce_sum(const torch::Tensor& local) override {
    auto x = local.contiguous();
    const bool cast = x.scalar_type() != torch::kFloat64;
    auto work = cast ? x.to(torch::kFloat64) : x;
    if (!work.device().is_xpu()) work = work.to(torch::Device(torch::kXPU, 0));
    work = work.contiguous();
    auto out = torch::empty_like(work);
    const size_t count = static_cast<size_t>(work.numel());
    const double _t0 = now_ms();
    ccl::allreduce(work.data_ptr(), out.data_ptr(), count,
                   ccl::datatype::float64, ccl::reduction::sum, *comm_,
                   *stream_)
        .wait();
    g_ar_ms += now_ms() - _t0; ++g_ar_n;
    if (cast) out = out.to(local.scalar_type());
    return out.contiguous();
  }

  torch::Tensor all_gather(const torch::Tensor& local) override {
    auto x = local.contiguous();
    if (!x.device().is_xpu()) x = x.to(torch::Device(torch::kXPU, 0)).contiguous();
    std::vector<int64_t> out_shape = x.sizes().vec();
    if (out_shape.empty()) out_shape = {1};
    out_shape[0] = out_shape[0] * world_;
    auto out = torch::empty(out_shape, x.options());
    const size_t count = static_cast<size_t>(x.numel());  // per-rank element count
    const double _t0 = now_ms();
    ccl::allgather(x.data_ptr(), out.data_ptr(), count, ccl_dtype(x.scalar_type()),
                   *comm_, *stream_)
        .wait();
    g_ag_ms += now_ms() - _t0; ++g_ag_n;
    g_ag_bytes += static_cast<double>(count) * x.element_size();
    return out.contiguous();
  }

  void barrier() override {
    ccl::barrier(*comm_, *stream_);
  }

  int rank() const override { return rank_; }
  int world() const override { return world_; }

 private:
  int rank_;
  int world_;
  sycl::device sycl_dev_;
  sycl::context sycl_ctx_;
  sycl::queue* queue_ = nullptr;
  std::unique_ptr<ccl::communicator> comm_;
  std::unique_ptr<ccl::stream> stream_;
};

std::shared_ptr<XcclPeer> XcclPeer::create(int rank, int world,
                                           int device_index) {
  return std::make_shared<XcclPeerImpl>(rank, world, device_index);
}

void peer_perf_read_reset(double& ag_ms, int& ag_n, double& ag_bytes,
                          double& ar_ms, int& ar_n) {
  ag_ms = g_ag_ms; ag_n = g_ag_n; ag_bytes = g_ag_bytes;
  ar_ms = g_ar_ms; ar_n = g_ar_n;
  g_ag_ms = 0.0; g_ag_n = 0; g_ag_bytes = 0.0;
  g_ar_ms = 0.0; g_ar_n = 0;
}

}  // namespace kokkos_peer
}  // namespace uma
