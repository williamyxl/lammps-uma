#pragma once

// Slim CUDA vesin adapter borrowed from graspa-mlip src_clean/vesin_nl.h.
// Builds FairChem-compatible graphs for UMA (edge flip happens in Predictor).

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/script.h>

#if defined(VESIN_ROOT)
#include "vesin_torch.hpp"
#endif

namespace uma {
namespace vesin_nl {

struct VesinGraphDevice {
  torch::Tensor edge_index;   // [2, E] int64 CUDA — row0=center, row1=neighbor (vesin)
  torch::Tensor shifts;       // [E, 3] int32
  torch::Tensor shifts_cart;  // [E, 3] float
  torch::Tensor wrapped_pos;  // [N, 3]
};

inline void require_cuda_tensor(const torch::Tensor& t, const char* name) {
  if (!t.defined() || !t.is_cuda()) {
    throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
  }
}

// Fractional wrap on device — mirrors graspa / neighbor_list_pbc wrap.
// Optional inv_cell skips linalg_inv when the caller already has it (W9).
inline torch::Tensor wrap_positions_cuda(
    const torch::Tensor& pos_cuda, const torch::Tensor& cell_cuda,
    const torch::Tensor& pbc_cuda, torch::ScalarType float_dtype,
    const torch::Tensor& inv_cell_opt = torch::Tensor()) {
  require_cuda_tensor(pos_cuda, "pos_cuda");
  require_cuda_tensor(cell_cuda, "cell_cuda");
  require_cuda_tensor(pbc_cuda, "pbc_cuda");

  auto pos = pos_cuda.to(float_dtype).contiguous();
  auto cell = cell_cuda.to(float_dtype).contiguous();
  if (pos.dim() != 2 || pos.size(1) != 3) {
    throw std::runtime_error("pos_cuda must be [N, 3]");
  }

  auto inv_cell =
      (inv_cell_opt.defined() && inv_cell_opt.scalar_type() == float_dtype &&
       inv_cell_opt.device() == pos.device())
          ? inv_cell_opt
          : torch::linalg_inv(cell);
  auto frac = pos.matmul(inv_cell);

  torch::Tensor pbc_mask;
  if (pbc_cuda.numel() == 1) {
    // opt5-graph: build the mask on-device without .item() (which forces an
    // XPU->host sync every step). Broadcast the scalar bool to [1,3] on device.
    pbc_mask = pbc_cuda.to(float_dtype).view({1, 1}).expand({1, 3}).to(
        pos.device());
  } else if (pbc_cuda.numel() == 3) {
    pbc_mask = pbc_cuda.to(float_dtype).view({1, 3}).to(pos.device());
  } else {
    throw std::runtime_error("pbc_cuda must be scalar bool or length-3 bool tensor");
  }

  frac = frac - torch::floor(frac) * pbc_mask;
  return frac.matmul(cell);
}

inline torch::Tensor shifts_cart_from_unit_shifts(const torch::Tensor& shifts,
                                                  const torch::Tensor& cell_cuda,
                                                  torch::ScalarType float_dtype) {
  auto S = shifts.to(float_dtype);
  auto cell = cell_cuda.to(float_dtype);
  return S.matmul(cell);
}

// UMA max_neighbors cap: per-center distance sort, keep nearest K (FairChem parity).
inline void cap_edges_per_center(torch::Tensor& edge_index, torch::Tensor& shifts,
                                 torch::Tensor& shifts_cart,
                                 const torch::Tensor& edge_dist, int max_neighbors) {
  if (max_neighbors <= 0) return;

  const int64_t n_edges = edge_index.size(1);
  if (n_edges == 0) return;

  auto i_cpu =
      edge_index.index({0}).detach().to(torch::kCPU, torch::kInt64).contiguous();
  auto dist_cpu = edge_dist.detach().to(torch::kCPU, torch::kFloat64).contiguous();
  auto i_acc = i_cpu.accessor<int64_t, 1>();
  auto dist_acc = dist_cpu.accessor<double, 1>();

  std::unordered_map<int64_t, std::vector<std::pair<double, int64_t>>> by_center;
  by_center.reserve(static_cast<size_t>(n_edges));
  for (int64_t e = 0; e < n_edges; ++e) {
    by_center[i_acc[e]].emplace_back(dist_acc[e], e);
  }

  std::vector<int64_t> keep;
  keep.reserve(static_cast<size_t>(n_edges));
  for (auto& kv : by_center) {
    auto& pairs = kv.second;
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<double, int64_t>& a,
                 const std::pair<double, int64_t>& b) { return a.first < b.first; });
    const int n_keep =
        std::min(static_cast<int>(pairs.size()), max_neighbors);
    for (int k = 0; k < n_keep; ++k) {
      keep.push_back(pairs[static_cast<size_t>(k)].second);
    }
  }
  std::sort(keep.begin(), keep.end());

  auto keep_idx =
      torch::empty({static_cast<int64_t>(keep.size())},
                   torch::TensorOptions().dtype(torch::kInt64));
  {
    auto keep_acc = keep_idx.accessor<int64_t, 1>();
    for (size_t k = 0; k < keep.size(); ++k) {
      keep_acc[static_cast<int64_t>(k)] = keep[k];
    }
  }
  keep_idx = keep_idx.to(edge_index.device());

  edge_index = edge_index.index_select(1, keep_idx);
  shifts = shifts.index_select(0, keep_idx);
  shifts_cart = shifts_cart.index_select(0, keep_idx);
}

#if defined(VESIN_ROOT)
inline VesinGraphDevice vesin_build_graph_cuda_impl(torch::Tensor pos_cuda,
                                                    torch::Tensor cell_cuda,
                                                    torch::Tensor pbc_cuda,
                                                    double cutoff, int max_neighbors,
                                                    bool full_directed,
                                                    torch::ScalarType float_dtype) {
  // vesin-torch cell_list requires float64 points/box regardless of MLIP dtype.
  // W9: wrap once; reuse for vesin + wrapped_pos (product path is FP64).
  constexpr torch::ScalarType kVesinComputeDtype = torch::kFloat64;
  auto cell_vesin = cell_cuda.to(kVesinComputeDtype).contiguous();
  auto inv_cell = torch::linalg_inv(cell_vesin);
  auto pos_wrapped = wrap_positions_cuda(pos_cuda, cell_cuda, pbc_cuda,
                                         kVesinComputeDtype, inv_cell);
  auto pos_vesin = pos_wrapped.contiguous();
  auto pbc_vesin = pbc_cuda.contiguous();

  auto nl = c10::make_intrusive<vesin_torch::NeighborListHolder>(
      cutoff, full_directed, /*sorted=*/false, "cell_list");
  auto out = nl->compute(pos_vesin, cell_vesin, pbc_vesin, "ijS", /*copy=*/true);
  if (out.size() != 3) {
    throw std::runtime_error("vesin_build_graph_cuda: expected 3 tensors from ijS");
  }

  auto i = out[0].to(torch::kInt64).contiguous();
  auto j = out[1].to(torch::kInt64).contiguous();
  auto S = out[2].to(torch::kInt32).contiguous();

  if (!i.is_cuda() || !j.is_cuda() || !S.is_cuda()) {
    throw std::runtime_error(
        "vesin_build_graph_cuda: vesin ijS outputs must be CUDA tensors");
  }

  if (i.dim() == 0) i = i.reshape({1});
  if (j.dim() == 0) j = j.reshape({1});
  auto edge_index = torch::stack({i, j}, 0);
  auto shifts = S;
  auto shifts_cart =
      shifts_cart_from_unit_shifts(shifts, cell_cuda, float_dtype);
  // Same wrapped frame; cast only if MLIP dtype differs from vesin FP64.
  auto wrapped_pos = (float_dtype == kVesinComputeDtype)
                         ? pos_wrapped
                         : pos_wrapped.to(float_dtype);

  if (max_neighbors > 0 && edge_index.size(1) > 0) {
    // Re-index i/j after possible dim fix; use current edge_index rows.
    i = edge_index.index({0});
    j = edge_index.index({1});
    // opt5-graph: the cap-check does `counts.max().item()` — a BLOCKING XPU->host
    // sync EVERY MD step. For a fixed lattice (e.g. NaCl at 6Å) the max degree is
    // constant and always <= max_neighbors, so the cap is a no-op; the sync is
    // pure overhead that serializes the XPU queue each step. UMA_SKIP_MAXNBR_CAP=1
    // skips the whole check (and its sync) — bit-identical when the cap wouldn't
    // fire (which the caller asserts by setting the flag). Default: keep the check.
    static const bool skip_cap = [] {
      const char* e = std::getenv("UMA_SKIP_MAXNBR_CAP");
      return e != nullptr && e[0] == '1' && e[1] == '\0';
    }();
    if (!skip_cap) {
      const int64_t n_atoms = wrapped_pos.size(0);
      auto counts = torch::bincount(i, /*weights=*/{}, /*minlength=*/n_atoms);
      const int64_t max_degree = counts.max().item<int64_t>();
      if (max_degree > max_neighbors) {
        auto pi = wrapped_pos.index_select(0, i);
        auto pj = wrapped_pos.index_select(0, j) + shifts_cart;
        auto dist = (pj - pi).norm(2, /*dim=*/1);
        cap_edges_per_center(edge_index, shifts, shifts_cart, dist, max_neighbors);
      }
    }
  }

  VesinGraphDevice graph;
  graph.edge_index = edge_index;
  graph.shifts = shifts;
  graph.shifts_cart = shifts_cart;
  graph.wrapped_pos = wrapped_pos;
  return graph;
}
#endif

inline VesinGraphDevice vesin_build_graph_cuda(torch::Tensor pos_cuda,
                                               torch::Tensor cell_cuda,
                                               torch::Tensor pbc_cuda, double cutoff,
                                               int max_neighbors, bool full_directed,
                                               torch::ScalarType float_dtype) {
  require_cuda_tensor(pos_cuda, "pos_cuda");
  require_cuda_tensor(cell_cuda, "cell_cuda");
  require_cuda_tensor(pbc_cuda, "pbc_cuda");

  if (pos_cuda.dim() != 2 || pos_cuda.size(1) != 3) {
    throw std::runtime_error("pos_cuda must be [N, 3]");
  }
  if (cell_cuda.dim() != 2 || cell_cuda.size(0) != 3 || cell_cuda.size(1) != 3) {
    throw std::runtime_error("cell_cuda must be [3, 3]");
  }

#if defined(VESIN_ROOT)
  return vesin_build_graph_cuda_impl(pos_cuda, cell_cuda, pbc_cuda, cutoff,
                                     max_neighbors, full_directed, float_dtype);
#else
  (void)cutoff;
  (void)max_neighbors;
  (void)full_directed;
  (void)float_dtype;
  throw std::runtime_error(
      "vesin_build_graph_cuda: VESIN_ROOT not defined at compile time");
#endif
}

}  // namespace vesin_nl
}  // namespace uma
