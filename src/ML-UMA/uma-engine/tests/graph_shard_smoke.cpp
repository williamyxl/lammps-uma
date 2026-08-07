// Smoke test: FairChem-key partition covers all edges.
// Build: linked with uma_parity_cli-style Torch via uma_engine.
#include "uma/graph_shard.h"

#include <iostream>

int main() {
  const int64_t n_atoms = 8;
  // Toy edges: centers 0..7 each once + one extra to center 3
  auto e0 = torch::tensor({1, 2, 3, 4, 5, 6, 7, 0, 1}, torch::kLong);
  auto e1 = torch::tensor({0, 1, 2, 3, 4, 5, 6, 7, 3}, torch::kLong);
  auto edge_index = torch::stack({e0, e1}, 0);
  const int world = 2;
  if (!uma::graph_shard::partitions_cover_all_edges(edge_index, n_atoms, world)) {
    std::cerr << "FAIL: partitions do not cover all edges\n";
    return 1;
  }
  int64_t total_e = 0;
  for (int r = 0; r < world; ++r) {
    auto s = uma::graph_shard::shard_edges(edge_index, torch::Tensor(), n_atoms,
                                           world, r);
    total_e += s.edge_index.size(1);
    std::cout << "rank " << r << " nodes=" << s.node_ids.numel()
              << " edges=" << s.edge_index.size(1) << "\n";
  }
  if (total_e != edge_index.size(1)) {
    std::cerr << "FAIL: edge count mismatch\n";
    return 1;
  }
  std::cout << "graph_shard_smoke OK\n";
  return 0;
}
