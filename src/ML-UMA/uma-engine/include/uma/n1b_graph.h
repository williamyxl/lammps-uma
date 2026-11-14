#pragma once

// Step 16 (N1-b). SYCL candidate pairs for the device neighbor builder.
// Order and max_neighbors truncation stay in emit_graph. This header is
// GCC-safe: the kernels live in n1b_graph.sycl.cpp.

#include <torch/torch.h>

#include <cstdint>

namespace uma {

struct N1bCandidates {
  torch::Tensor wrapped;
  torch::Tensor neighbor;
  torch::Tensor ox;
  torch::Tensor oy;
  torch::Tensor oz;
  torch::Tensor keep;
  int64_t overflow = 0;
};

// Orthorhombic cell list. nc[d] >= 1, len[d] > 0. overflow 1 when a cell
// holds more than cell_cap atoms.
N1bCandidates n1b_cell_candidates(const torch::Tensor& pos64, const double cell[3][3],
                                 const int pbc[3], double cutoff, double tol, int64_t cell_cap,
                                 const int nc[3], const double len[3]);

// All-pairs over the image list. images is nimg triples (ox, oy, oz).
N1bCandidates n1b_pair_candidates(const torch::Tensor& pos64, const double cell[3][3],
                                 const int pbc[3], double cutoff, double tol,
                                 const int64_t* images, int64_t nimg);

}  // namespace uma
