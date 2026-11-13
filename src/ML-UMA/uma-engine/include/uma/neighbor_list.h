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

/// Device (or CPU-tensor) neighbor graph built from composed torch ops.
/// Edge order matches emit_graph: center ascending, then (distance, neighbor,
/// ox, oy, oz). overflow == 0 means the graph is complete.
///   1 cell occupancy exceeded cell_cap (a closer neighbor may have been dropped)
///   2 all-pairs candidate width exceeds the packed row limit
///   3 atom count does not fit the tie-break key (N >= 2^20)
///   4 an image index is outside ±7
struct DeviceNeighborBuild {
  NeighborGraph graph;
  torch::Tensor wrapped_pos;
  int64_t overflow = 0;
};

bool nitro_n1_requested();
int64_t nitro_n1_cell_cap();
const char* nitro_n1_overflow_reason(int64_t code);

DeviceNeighborBuild build_neighbor_graph_device(const torch::Tensor& pos,
                                               const torch::Tensor& cell,
                                               const torch::Tensor& pbc,
                                               const NeighborListConfig& config,
                                               int64_t cell_cap);

}  // namespace uma
