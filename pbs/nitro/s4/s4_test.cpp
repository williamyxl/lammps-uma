// Step 15 (S4). 12 ranks. Each round calls silu_affine, then uma_peer
// all_reduce_sum of that output, then the op again, then an all-reduce of a
// rank id. The op waits before it returns, and the collective waits before it
// returns, so this checks the interleaved call order on one node.

#include "uma/xccl_peer.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <mpi.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

double host_silu(double z) {
  if (z >= 0.0) {
    const double e = std::exp(-z);
    return z / (1.0 + e);
  }
  const double e = std::exp(z);
  return z * e / (1.0 + e);
}

at::Tensor silu_affine(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("uma_sycl::silu_affine", "")
                       .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
  return op.call(x, w, b);
}

double max_abs(const torch::Tensor& a, const torch::Tensor& b) {
  return (a - b).abs().max().item<double>();
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  std::cerr << "S4_ENTER rank=" << rank << " world=" << world << "\n" << std::flush;

  int rc = 0;
  if (world != 12) {
    if (rank == 0) std::cerr << "S4_FAIL world " << world << " want 12\n";
    rc = 1;
  } else if (!torch::xpu::is_available()) {
    if (rank == 0) std::cerr << "S4_FAIL xpu unavailable\n";
    rc = 1;
  }

  int any = 0;
  MPI_Allreduce(&rc, &any, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (any != 0) {
    MPI_Finalize();
    return 1;
  }

  auto scratch = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kXPU));
  torch::xpu::synchronize(0);
  (void)scratch;

  auto peer = uma::uma_peer::XcclPeer::create(rank, world, 0, static_cast<int>(MPI_Comm_c2f(MPI_COMM_WORLD)));
  if (rank == 0) std::cerr << "S4_PEER_OK\n" << std::flush;

  constexpr int64_t n = 4096;
  constexpr int rounds = 8;
  double op_rel = 0.0;
  double ar_rel = 0.0;
  int local_fail = 0;

  for (int r = 0; r < rounds; ++r) {
    auto hx = torch::empty({n}, torch::kFloat64);
    auto hw = torch::empty({n}, torch::kFloat64);
    auto hb = torch::empty({n}, torch::kFloat64);
    auto href = torch::empty({n}, torch::kFloat64);
    auto* xp = hx.data_ptr<double>();
    auto* wp = hw.data_ptr<double>();
    auto* bp = hb.data_ptr<double>();
    auto* rp = href.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
      xp[i] = std::sin((i + 1) * 0.01 + rank * 0.17 + r * 0.03);
      wp[i] = std::cos((i + 1) * 0.02) * 0.5 + 0.1;
      bp[i] = std::sin((i + 1) * 0.007) * 2.0;
      rp[i] = host_silu(xp[i] * wp[i] + bp[i]);
    }
    auto x = hx.to(torch::kXPU);
    auto w = hw.to(torch::kXPU);
    auto b = hb.to(torch::kXPU);

    auto y = silu_affine(x, w, b);
    auto hy = y.to(torch::kCPU).contiguous();
    const double op_abs = max_abs(hy, href);
    const double op_scale = std::max(1.0, href.abs().max().item<double>());
    op_rel = std::max(op_rel, op_abs / op_scale);
    if (op_abs / op_scale > 1e-15) local_fail = 1;

    auto sum = peer->all_reduce_sum(y);
    auto hsum = sum.to(torch::kCPU).contiguous();
    auto oracle = torch::empty({n}, torch::kFloat64);
    MPI_Allreduce(hy.data_ptr<double>(), oracle.data_ptr<double>(), static_cast<int>(n),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double ar_abs = max_abs(hsum, oracle);
    const double ar_scale = std::max(1.0, oracle.abs().max().item<double>());
    ar_rel = std::max(ar_rel, ar_abs / ar_scale);
    if (ar_abs / ar_scale > 1e-12) local_fail = 1;

    auto y2 = silu_affine(x, w, b);
    auto hy2 = y2.to(torch::kCPU).contiguous();
    if (max_abs(hy2, hy) != 0.0) local_fail = 1;

    auto id = torch::full({8}, static_cast<double>(rank + 1),
                          torch::TensorOptions().dtype(torch::kFloat64).device(torch::kXPU));
    auto idsum = peer->all_reduce_sum(id).to(torch::kCPU);
    const double expect = static_cast<double>(world) * static_cast<double>(world + 1) / 2.0;
    if (max_abs(idsum, torch::full({8}, expect, torch::kFloat64)) != 0.0) local_fail = 1;
  }

  double op_rel_all = 0.0;
  double ar_rel_all = 0.0;
  int fails = 0;
  MPI_Reduce(&op_rel, &op_rel_all, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&ar_rel, &ar_rel_all, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_fail, &fails, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&fails, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cerr << "S4_OP_REL " << op_rel_all << "\n";
    std::cerr << "S4_AR_REL " << ar_rel_all << "\n";
    std::cerr << "S4_FAILS " << fails << "\n";
    if (fails == 0) std::cerr << "S4_PASS\n";
    else std::cerr << "S4_FAIL\n";
  }
  MPI_Finalize();
  return fails == 0 ? 0 : 1;
}
