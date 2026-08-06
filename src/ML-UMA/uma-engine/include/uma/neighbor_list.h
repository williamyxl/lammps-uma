#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace uma {

struct NeighborListConfig {
  double cutoff = 6.0;
  int max_neighbors = 300;
  double distance_tolerance = 1e-8;
};

struct NeighborGraph {
  torch::Tensor edge_index;    // [2, E] int64 — row0=neighbor, row1=center
  torch::Tensor cell_offsets;  // [E, 3] float — integer image triples
};

/// Wrap Cartesian positions into the unit cell (ASE/FairChem AtomicData convention).
/// Must be applied before model forward — neighbor offsets are defined in the wrapped frame.
torch::Tensor wrap_positions_to_cell(const torch::Tensor& pos,
                                     const torch::Tensor& cell,
                                     const torch::Tensor& pbc);

/// Single-system neighbor list (pymatgen-external-graph compatible).
NeighborGraph build_neighbor_graph(const torch::Tensor& pos,
                                   const torch::Tensor& cell,
                                   const torch::Tensor& pbc,
                                   const NeighborListConfig& config = {});

}  // namespace uma
