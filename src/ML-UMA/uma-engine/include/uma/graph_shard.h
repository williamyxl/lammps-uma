#pragma once

// FairChem-compatible atom/edge partition for same-node graph parallel.
// NL must already be a full vesin (or CPU) graph with FairChem orientation:
//   edge_index[0] = neighbor, edge_index[1] = center (target).
// Partition key matches FairChem gp_utils / filter_edges_by_node_partition:
//   node_partition = tensor_split(arange(n_atoms), world_size)[rank]
//   keep edges where edge_index[1] ∈ node_partition
//
// Does NOT run the UMA forward. Used by native Kokkos+LibTorch GP (no Ray).

#include <stdexcept>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace uma {
namespace graph_shard {

inline torch::Tensor node_partition(int64_t n_atoms, int world_size, int rank) {
  if (world_size < 1 || rank < 0 || rank >= world_size) {
    throw std::runtime_error("node_partition: invalid world_size/rank");
  }
  if (n_atoms < 0) {
    throw std::runtime_error("node_partition: n_atoms < 0");
  }
  auto all = torch::arange(n_atoms, torch::TensorOptions().dtype(torch::kLong));
  auto parts = torch::tensor_split(all, world_size);
  return parts[rank].contiguous();
}

struct Shard {
  torch::Tensor node_ids;     // [n_local] int64
  torch::Tensor edge_index;   // [2, E_local] int64 (FairChem orientation)
  torch::Tensor cell_offsets; // [E_local, 3] optional empty
  torch::Tensor edge_keep;    // [E] bool mask on full graph (debug)
};

// edge_index: [2, E] on CPU or CUDA; cell_offsets: [E, 3] or undefined.
inline Shard shard_edges(const torch::Tensor& edge_index,
                         const torch::Tensor& cell_offsets, int64_t n_atoms,
                         int world_size, int rank) {
  if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
    throw std::runtime_error("shard_edges: edge_index must be [2, E]");
  }
  auto nodes = node_partition(n_atoms, world_size, rank).to(edge_index.device());
  // keep if center (row 1) is in this rank's partition
  auto centers = edge_index[1];
  auto keep = torch::isin(centers, nodes);
  auto idx = keep.nonzero().squeeze(-1);
  Shard out;
  out.node_ids = nodes;
  out.edge_keep = keep;
  out.edge_index = edge_index.index_select(/*dim=*/1, idx).contiguous();
  if (cell_offsets.defined() && cell_offsets.numel() > 0) {
    out.cell_offsets = cell_offsets.index_select(/*dim=*/0, idx).contiguous();
  }
  return out;
}

// Coverage check: every edge assigned to exactly one rank (by center).
inline bool partitions_cover_all_edges(const torch::Tensor& edge_index,
                                       int64_t n_atoms, int world_size) {
  auto assigned = torch::zeros({edge_index.size(1)},
                               torch::TensorOptions()
                                   .dtype(torch::kBool)
                                   .device(edge_index.device()));
  for (int r = 0; r < world_size; ++r) {
    auto s = shard_edges(edge_index, torch::Tensor(), n_atoms, world_size, r);
    assigned = assigned.logical_or(s.edge_keep);
  }
  return assigned.all().item<bool>();
}

}  // namespace graph_shard
}  // namespace uma
