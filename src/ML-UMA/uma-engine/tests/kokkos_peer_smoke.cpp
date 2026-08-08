#include "uma/kokkos_peer.h"
#include "uma/graph_shard.h"

#include <cmath>
#include <iostream>

int main() {
  using namespace uma;

  // size_list / pad
  auto sizes = kokkos_peer::size_list(10, 4);
  if (sizes != std::vector<int64_t>({3, 3, 2, 2})) {
    std::cerr << "size_list mismatch\n";
    return 1;
  }

  auto a = torch::arange(3, torch::kFloat64).unsqueeze(1);
  auto b = torch::arange(3, 6, torch::kFloat64).unsqueeze(1);
  auto c = torch::arange(6, 8, torch::kFloat64).unsqueeze(1);
  auto d = torch::arange(8, 10, torch::kFloat64).unsqueeze(1);
  // pad last two to 3
  c = kokkos_peer::pad_nodes(c, 3);
  d = kokkos_peer::pad_nodes(d, 3);

  auto full = kokkos_peer::all_gather_nodes({a, b, c, d}, /*n_atoms=*/10, torch::kCPU);
  auto expect = torch::arange(10, torch::kFloat64).unsqueeze(1);
  if (!torch::allclose(full, expect)) {
    std::cerr << "all_gather_nodes mismatch\n" << full << "\n";
    return 1;
  }

  auto s = kokkos_peer::all_reduce_sum(
      {torch::tensor({1.0}), torch::tensor({2.0}), torch::tensor({3.0})}, torch::kCPU);
  if (std::abs(s.item<double>() - 6.0) > 1e-12) {
    std::cerr << "all_reduce_sum mismatch\n";
    return 1;
  }

  // graph_shard coverage still holds for n=10, world=4
  auto edges = torch::stack({torch::arange(10), torch::arange(10)}, 0);
  if (!graph_shard::partitions_cover_all_edges(edges, 10, 4)) {
    std::cerr << "partition coverage failed\n";
    return 1;
  }

  std::cout << "kokkos_peer_smoke OK\n";
  return 0;
}
