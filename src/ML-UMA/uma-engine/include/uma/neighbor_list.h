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
/// Dispatches to the O(N*neighbors) cell-list for large orthorhombic boxes
/// (floor(L_d/cutoff) >= 3 on every periodic axis) and to the O(N^2) all-pairs
/// path otherwise. Set env UMA_NL_ALLPAIRS=1 to force the all-pairs path.
NeighborGraph build_neighbor_graph(const torch::Tensor& pos,
                                   const torch::Tensor& cell,
                                   const torch::Tensor& pbc,
                                   const NeighborListConfig& config = {});

/// O(N^2) all-pairs implementation (original; always correct). Exposed for A/B.
NeighborGraph build_neighbor_graph_allpairs(const torch::Tensor& pos,
                                            const torch::Tensor& cell,
                                            const torch::Tensor& pbc,
                                            const NeighborListConfig& config = {});

/// O(N*neighbors) linked-cell implementation. Requires an orthorhombic box with
/// >= 3 bins per periodic axis; falls back to all-pairs if that is not met.
NeighborGraph build_neighbor_graph_celllist(const torch::Tensor& pos,
                                            const torch::Tensor& cell,
                                            const torch::Tensor& pbc,
                                            const NeighborListConfig& config = {});

}  // namespace uma
