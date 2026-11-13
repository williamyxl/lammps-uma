// Step 11 (N1-a): device neighbor graph vs the CPU reference builder.
// Runs on one XPU. Exit 0 only when every case matches, including edge order.

#include "uma/neighbor_list.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include <torch/torch.h>

namespace {

int g_fails = 0;

void sync_dev(const torch::Device& dev) {
  if (dev.is_cpu()) return;
  if (dev.is_xpu() && torch::xpu::is_available()) {
    torch::xpu::synchronize(dev.index() < 0 ? 0 : dev.index());
  } else if (dev.is_cuda() && torch::cuda::is_available()) {
    torch::cuda::synchronize();
  }
}

torch::Tensor diag_cell(double lx, double ly, double lz) {
  return torch::tensor({{lx, 0.0, 0.0}, {0.0, ly, 0.0}, {0.0, 0.0, lz}}, torch::kFloat64);
}

torch::Tensor pbc_all() {
  return torch::tensor({true, true, true}, torch::kBool);
}

torch::Tensor rocksalt(int R, double a) {
  const double basis[8][3] = {{0, 0, 0},       {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5},
                              {0.5, 0.5, 0.5}, {0, 0, 0.5},   {0, 0.5, 0},   {0.5, 0, 0}};
  auto out = torch::empty({static_cast<int64_t>(8) * R * R * R, 3}, torch::kFloat64);
  auto acc = out.accessor<double, 2>();
  int64_t n = 0;
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < R; ++j) {
      for (int k = 0; k < R; ++k) {
        for (int b = 0; b < 8; ++b) {
          acc[n][0] = (i + basis[b][0]) * a;
          acc[n][1] = (j + basis[b][1]) * a;
          acc[n][2] = (k + basis[b][2]) * a;
          ++n;
        }
      }
    }
  }
  return out;
}

void explain(const uma::NeighborGraph& got, const uma::NeighborGraph& ref) {
  auto gi = got.edge_index.defined() ? got.edge_index.to(torch::kCPU) : torch::Tensor();
  auto ri = ref.edge_index.defined() ? ref.edge_index.to(torch::kCPU) : torch::Tensor();
  std::cerr << "  edges got=" << (gi.defined() ? gi.size(1) : -1)
            << " ref=" << (ri.defined() ? ri.size(1) : -1) << "\n";
  if (!gi.defined() || !ri.defined() || gi.dim() != 2 || ri.dim() != 2 || gi.size(0) != 2 ||
      ri.size(0) != 2 || gi.size(1) != ri.size(1)) {
    return;
  }
  auto diff = gi.ne(ri).any(0).nonzero();
  const int64_t show = std::min<int64_t>(diff.size(0), 5);
  for (int64_t s = 0; s < show; ++s) {
    const int64_t i = diff[s].item<int64_t>();
    std::cerr << "  col " << i << " got " << gi[0][i].item<int64_t>() << ","
              << gi[1][i].item<int64_t>() << " ref " << ri[0][i].item<int64_t>() << ","
              << ri[1][i].item<int64_t>() << "\n";
  }
}

bool same_graph(const uma::NeighborGraph& got, const uma::NeighborGraph& ref) {
  if (!got.edge_index.defined() || !ref.edge_index.defined()) return false;
  auto gi = got.edge_index.to(torch::kCPU);
  auto ri = ref.edge_index.to(torch::kCPU);
  if (gi.sizes() != ri.sizes() || !torch::equal(gi, ri)) return false;
  auto gc = got.cell_offsets.to(torch::kFloat64).cpu();
  auto rc = ref.cell_offsets.to(torch::kFloat64).cpu();
  return gc.sizes() == rc.sizes() && torch::equal(gc, rc);
}

void check_case(const std::string& name, const torch::Tensor& pos, const torch::Tensor& cell,
                const torch::Tensor& pbc, double cutoff, int max_neighbors, int64_t cap,
                const torch::Device& dev, int expect_overflow) {
  uma::NeighborListConfig cfg;
  cfg.cutoff = cutoff;
  cfg.max_neighbors = max_neighbors;
  auto built = uma::build_neighbor_graph_device(pos.to(dev), cell.to(dev), pbc.to(dev), cfg, cap);
  if (expect_overflow != 0) {
    if (built.overflow != expect_overflow) {
      std::cerr << "FAIL " << name << " overflow=" << built.overflow
                << " expected " << expect_overflow << "\n";
      ++g_fails;
    } else {
      std::cerr << "OK   " << name << " overflow=" << built.overflow << "\n";
    }
    return;
  }
  if (built.overflow != 0) {
    std::cerr << "FAIL " << name << " unexpected overflow "
              << uma::nitro_n1_overflow_reason(built.overflow) << "\n";
    ++g_fails;
    return;
  }
  auto ref = uma::build_neighbor_graph(pos.to(torch::kFloat64).cpu(), cell.to(torch::kFloat64).cpu(),
                                       pbc.to(torch::kCPU), cfg);
  if (!same_graph(built.graph, ref)) {
    std::cerr << "FAIL " << name << " on " << dev << "\n";
    explain(built.graph, ref);
    ++g_fails;
    return;
  }
  std::cerr << "OK   " << name << " on " << dev << " edges=" << ref.edge_index.size(1) << "\n";
}

void check_wrap(const torch::Tensor& pos, const torch::Tensor& cell, const torch::Tensor& pbc,
                const torch::Device& dev) {
  uma::NeighborListConfig cfg;
  cfg.cutoff = 6.0;
  cfg.max_neighbors = 300;
  auto built = uma::build_neighbor_graph_device(pos.to(dev), cell.to(dev), pbc.to(dev), cfg, 64);
  auto ref = uma::wrap_positions_to_cell(pos.cpu(), cell.cpu(), pbc.cpu()).to(torch::kFloat64);
  auto got = built.wrapped_pos.to(torch::kFloat64).cpu();
  const double max_abs = (got - ref).abs().max().item<double>();
  if (built.overflow != 0 || max_abs > 1e-8) {
    std::cerr << "FAIL wrap max|d|=" << max_abs << " overflow=" << built.overflow << "\n";
    ++g_fails;
  } else {
    std::cerr << "OK   wrap max|d|=" << max_abs << "\n";
  }
}

}  // namespace

int main() {
  if (!torch::xpu::is_available() || torch::xpu::device_count() < 1) {
    std::cerr << "N1_SELFTEST_FAIL no XPU\n";
    return 1;
  }
  const torch::Device cpu(torch::kCPU);
  const torch::Device xpu(torch::kXPU, 0);
  std::cerr << "N1_SELFTEST device_count=" << torch::xpu::device_count() << "\n";

  const auto pbc = pbc_all();
  auto pos4 = rocksalt(4, 5.64);
  auto cell4 = diag_cell(5.64 * 4, 5.64 * 4, 5.64 * 4);
  for (torch::Device dev : {cpu, xpu}) {
    check_case("rocksalt 4^3", pos4, cell4, pbc, 6.0, 300, 64, dev, 0);
    check_case("rocksalt outside box", pos4 + (5.64 * 4), cell4, pbc, 6.0, 300, 64, dev, 0);
  }
  check_wrap(pos4 + (5.64 * 4), cell4, pbc, xpu);

  torch::manual_seed(0);
  auto rnd = torch::rand({128, 3}, torch::kFloat64) * 24.0;
  auto cell_r = diag_cell(24, 24, 24);
  for (torch::Device dev : {cpu, xpu}) {
    check_case("random ortho N=128", rnd, cell_r, pbc, 6.0, 300, 64, dev, 0);
  }

  auto tiny = torch::rand({16, 3}, torch::kFloat64) * 4.0;
  auto cell_t = diag_cell(4, 4, 4);
  for (torch::Device dev : {cpu, xpu}) {
    check_case("tiny cell N=16", tiny, cell_t, pbc, 6.0, 300, 64, dev, 0);
  }

  auto dense = torch::rand({40, 3}, torch::kFloat64) * 4.0;
  for (torch::Device dev : {cpu, xpu}) {
    check_case("dense truncate max_neighbors=5", dense, cell_t, pbc, 6.0, 5, 64, dev, 0);
  }

  auto cutoff_pos = torch::tensor({{0.0, 0.0, 0.0},
                                   {6.0, 0.0, 0.0},
                                   {6.0 + 1e-4, 0.0, 0.0},
                                   {1e-10, 0.0, 0.0}},
                                  torch::kFloat64);
  auto cell_c = diag_cell(30, 30, 30);
  for (torch::Device dev : {cpu, xpu}) {
    check_case("cutoff boundary", cutoff_pos, cell_c, pbc, 6.0, 300, 64, dev, 0);
  }

  auto tri_cell = torch::tensor({{6.0, 0.5, 0.0}, {0.2, 6.0, 0.3}, {0.1, 0.4, 6.0}}, torch::kFloat64);
  auto tri_pos = torch::rand({20, 3}, torch::kFloat64).matmul(tri_cell);
  for (torch::Device dev : {cpu, xpu}) {
    check_case("triclinic N=20", tri_pos, tri_cell, pbc, 6.0, 300, 64, dev, 0);
  }

  auto crowd = torch::empty({20, 3}, torch::kFloat64);
  {
    auto acc = crowd.accessor<double, 2>();
    for (int i = 0; i < 20; ++i) {
      acc[i][0] = 0.05 + i * 1e-4;
      acc[i][1] = 0.05;
      acc[i][2] = 0.05;
    }
  }
  check_case("cell cap overflow", crowd, diag_cell(30, 30, 30), pbc, 6.0, 300, /*cap=*/2, xpu, 1);

  // Control: host build + H2D versus the device build, NaCl-sized 14^3.
  auto pos14 = rocksalt(14, 5.64);
  auto cell14 = diag_cell(5.64 * 14, 5.64 * 14, 5.64 * 14);
  uma::NeighborListConfig cfg;
  cfg.cutoff = 6.0;
  cfg.max_neighbors = 300;
  std::cerr << "timing NaCl-sized 14^3 nat=" << pos14.size(0) << "\n";
  for (int rep = 0; rep < 3; ++rep) {
    sync_dev(xpu);
    auto t0 = std::chrono::steady_clock::now();
    auto host = uma::build_neighbor_graph(pos14, cell14, pbc, cfg);
    auto h_idx = host.edge_index.to(xpu);
    auto h_off = host.cell_offsets.to(xpu);
    sync_dev(xpu);
    auto t1 = std::chrono::steady_clock::now();
    auto built = uma::build_neighbor_graph_device(pos14.to(xpu), cell14.to(xpu), pbc.to(xpu), cfg, 64);
    sync_dev(xpu);
    auto t2 = std::chrono::steady_clock::now();
    const double ms_host = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_dev = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cerr << "  rep " << rep << " host+H2D " << ms_host << " ms  device " << ms_dev
              << " ms  overflow=" << built.overflow << " edges_host=" << h_idx.size(1)
              << " edges_dev=" << built.graph.edge_index.size(1) << "\n";
    if (rep == 0) {
      uma::NeighborGraph moved;
      moved.edge_index = h_idx;
      moved.cell_offsets = h_off;
      if (built.overflow != 0 || !same_graph(built.graph, host)) {
        std::cerr << "FAIL 14^3 edge match\n";
        explain(built.graph, host);
        ++g_fails;
      } else {
        std::cerr << "OK   14^3 edge match\n";
      }
    }
    (void)h_off;
  }

  if (g_fails != 0) {
    std::cerr << "N1_SELFTEST_FAIL cases=" << g_fails << "\n";
    return 1;
  }
  std::cerr << "N1_SELFTEST_OK\n";
  return 0;
}
