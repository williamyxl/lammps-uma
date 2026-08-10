#pragma once

// FairChem-compatible atom/edge partition for same-node graph parallel.
// NL must already be a full vesin (or CPU) graph with FairChem orientation:
//   edge_index[0] = neighbor, edge_index[1] = center (target).
// Partition key matches FairChem gp_utils / filter_edges_by_node_partition:
//   node_partition = tensor_split(arange(n_atoms), world_size)[rank]
//   keep edges where edge_index[1] ∈ node_partition
//
// Does NOT run the UMA forward. Used by native Kokkos+LibTorch GP (no Ray).

#include <cstdint>
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

// W10: pad local edges to a fixed capacity for allocator / CUDA-graph stability.
// Dummy edges are self-edges on atom `pad_atom` with a large lattice shift so
// |r| ≫ cutoff (FairChem add_n_empty_edges spirit: contribution ~0 via envelope).
// edge_index: [2, E] int64; cell_offsets: [E, 3] float (or empty → zeros then pad).
inline void pad_edges_to_capacity(torch::Tensor& edge_index,
                                  torch::Tensor& cell_offsets, int64_t capacity,
                                  int64_t pad_atom) {
  if (capacity < 1) return;
  const int64_t e = edge_index.defined() ? edge_index.size(1) : 0;
  if (e >= capacity) return;
  const auto opts_i =
      torch::TensorOptions().dtype(torch::kLong).device(edge_index.device());
  const auto opts_f =
      cell_offsets.defined() && cell_offsets.numel() > 0
          ? cell_offsets.options()
          : torch::TensorOptions().dtype(torch::kFloat64).device(edge_index.device());
  const int64_t n_pad = capacity - e;
  auto pad_eidx = torch::full({2, n_pad}, pad_atom, opts_i);
  // Large integer-like shift in float coff space (model does coff @ cell).
  auto pad_coff = torch::zeros({n_pad, 3}, opts_f);
  pad_coff.index_put_({torch::indexing::Slice(), 0}, 2.0);
  if (e == 0) {
    edge_index = pad_eidx;
    cell_offsets = pad_coff;
    return;
  }
  edge_index = torch::cat({pad_eidx, edge_index}, /*dim=*/1).contiguous();
  if (!cell_offsets.defined() || cell_offsets.numel() == 0) {
    cell_offsets = torch::zeros({e, 3}, opts_f);
  }
  cell_offsets = torch::cat({pad_coff, cell_offsets}, /*dim=*/0).contiguous();
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

// Rank of atom i under torch::tensor_split(arange(n), world) (FairChem).
inline int rank_of_atom(int64_t i, int64_t n_atoms, int world_size) {
  if (world_size <= 1) return 0;
  const int64_t base = n_atoms / world_size;
  const int64_t rem = n_atoms % world_size;
  const int64_t first = rem * (base + 1);
  if (i < first) return static_cast<int>(i / (base + 1));
  if (base == 0) return world_size - 1;
  return static_cast<int>(rem + (i - first) / base);
}

// Fast CPU pack of all rank shards (no torch::isin).
// Layout per rank: eidx [2, ne] row-major (row0 then row1); coff [ne, 3] int32.
// eidx_out[r] capacity ≥ 2*max_e; coff_out[r] ≥ 3*max_e.
inline bool pack_shards_cpu(const int64_t* eidx_full,  // [2, E] row-major
                            const int32_t* coff_full,  // [E, 3] or nullptr
                            int64_t n_edges, int64_t n_atoms, int world_size,
                            int64_t max_e, int32_t* nedges_out,
                            int64_t** eidx_out, int32_t** coff_out) {
  if (world_size < 1 || n_edges < 0) return false;
  std::vector<int64_t> counts(static_cast<size_t>(world_size), 0);
  const int64_t* neigh = eidx_full;
  const int64_t* center = eidx_full + n_edges;
  for (int64_t e = 0; e < n_edges; ++e) {
    const int r = rank_of_atom(center[e], n_atoms, world_size);
    if (r < 0 || r >= world_size) return false;
    ++counts[static_cast<size_t>(r)];
  }
  for (int r = 0; r < world_size; ++r) {
    if (counts[static_cast<size_t>(r)] > max_e) return false;
    nedges_out[r] = static_cast<int32_t>(counts[static_cast<size_t>(r)]);
    counts[static_cast<size_t>(r)] = 0;
  }
  for (int64_t e = 0; e < n_edges; ++e) {
    const int r = rank_of_atom(center[e], n_atoms, world_size);
    const int64_t ne = nedges_out[r];
    const int64_t j = counts[static_cast<size_t>(r)]++;
    eidx_out[r][j] = neigh[e];
    eidx_out[r][ne + j] = center[e];
    if (coff_full) {
      coff_out[r][3 * j + 0] = coff_full[3 * e + 0];
      coff_out[r][3 * j + 1] = coff_full[3 * e + 1];
      coff_out[r][3 * j + 2] = coff_full[3 * e + 2];
    } else {
      coff_out[r][3 * j + 0] = coff_out[r][3 * j + 1] = coff_out[r][3 * j + 2] = 0;
    }
  }
  return true;
}

// Legacy double-coff overload (unused after W5; kept for compile safety).
inline bool pack_shards_cpu(const int64_t* eidx_full, const double* coff_full,
                            int64_t n_edges, int64_t n_atoms, int world_size,
                            int64_t max_e, int32_t* nedges_out, int64_t** eidx_out,
                            double** coff_out) {
  if (world_size < 1 || n_edges < 0) return false;
  std::vector<int64_t> counts(static_cast<size_t>(world_size), 0);
  const int64_t* neigh = eidx_full;
  const int64_t* center = eidx_full + n_edges;
  for (int64_t e = 0; e < n_edges; ++e) {
    const int r = rank_of_atom(center[e], n_atoms, world_size);
    if (r < 0 || r >= world_size) return false;
    ++counts[static_cast<size_t>(r)];
  }
  for (int r = 0; r < world_size; ++r) {
    if (counts[static_cast<size_t>(r)] > max_e) return false;
    nedges_out[r] = static_cast<int32_t>(counts[static_cast<size_t>(r)]);
    counts[static_cast<size_t>(r)] = 0;
  }
  for (int64_t e = 0; e < n_edges; ++e) {
    const int r = rank_of_atom(center[e], n_atoms, world_size);
    const int64_t ne = nedges_out[r];
    const int64_t j = counts[static_cast<size_t>(r)]++;
    eidx_out[r][j] = neigh[e];
    eidx_out[r][ne + j] = center[e];
    if (coff_full) {
      coff_out[r][3 * j + 0] = coff_full[3 * e + 0];
      coff_out[r][3 * j + 1] = coff_full[3 * e + 1];
      coff_out[r][3 * j + 2] = coff_full[3 * e + 2];
    } else {
      coff_out[r][3 * j + 0] = coff_out[r][3 * j + 1] = coff_out[r][3 * j + 2] = 0.0;
    }
  }
  return true;
}

}  // namespace graph_shard
}  // namespace uma
