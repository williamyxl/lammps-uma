#include "uma/neighbor_list.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace uma {
namespace {

struct Vec3 {
  double x = 0;
  double y = 0;
  double z = 0;
};

struct NeighborCandidate {
  int64_t neighbor;
  int ox, oy, oz;
  double distance;
};

Vec3 add(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 sub(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

double norm(const Vec3& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}


std::array<std::array<double, 3>, 3> cell_from_tensor(const torch::Tensor& cell) {
  auto c = cell.contiguous().to(torch::kFloat64).cpu();
  if (c.dim() == 3) {
    c = c.squeeze(0);
  }
  if (c.dim() != 2 || c.size(0) != 3 || c.size(1) != 3) {
    throw std::runtime_error("cell must be [3,3] or [1,3,3]");
  }
  auto acc = c.accessor<double, 2>();
  return {{{acc[0][0], acc[0][1], acc[0][2]},
           {acc[1][0], acc[1][1], acc[1][2]},
           {acc[2][0], acc[2][1], acc[2][2]}}};
}

std::array<bool, 3> pbc_from_tensor(const torch::Tensor& pbc) {
  auto p = pbc.contiguous().to(torch::kBool).cpu();
  if (p.dim() == 2) {
    p = p.squeeze(0);
  }
  if (p.dim() != 1 || p.size(0) != 3) {
    throw std::runtime_error("pbc must be [3] or [1,3]");
  }
  auto acc = p.accessor<bool, 1>();
  return {acc[0], acc[1], acc[2]};
}

std::vector<Vec3> positions_from_tensor(const torch::Tensor& pos) {
  auto p = pos.contiguous().to(torch::kFloat64).cpu();
  if (p.dim() != 2 || p.size(1) != 3) {
    throw std::runtime_error("pos must be [N,3]");
  }
  std::vector<Vec3> out(p.size(0));
  auto acc = p.accessor<double, 2>();
  for (int64_t i = 0; i < p.size(0); ++i) {
    out[i] = {acc[i][0], acc[i][1], acc[i][2]};
  }
  return out;
}

void wrap_positions(std::vector<Vec3>& pos,
                    const std::array<std::array<double, 3>, 3>& cell,
                    const std::array<bool, 3>& pbc) {
  const int n = static_cast<int>(pos.size());
  for (int i = 0; i < n; ++i) {
    // frac = solve(cell^T, pos)  (rows are lattice vectors)
    const double det =
        cell[0][0] * (cell[1][1] * cell[2][2] - cell[1][2] * cell[2][1]) -
        cell[0][1] * (cell[1][0] * cell[2][2] - cell[1][2] * cell[2][0]) +
        cell[0][2] * (cell[1][0] * cell[2][1] - cell[1][1] * cell[2][0]);
    if (std::abs(det) < 1e-12) {
      continue;
    }
    const Vec3 r = pos[i];
    // Rows of `cell` are the lattice vectors (ASE/fairchem convention), so the
    // fractional coordinates are frac = r * cell^{-1}, i.e. frac[k] = sum_j r_j *
    // cofactor(k, j) / det. The previous code used cofactor(j, k) (the transpose),
    // which only matches for diagonal/symmetric cells and silently corrupts the
    // wrapping for non-orthogonal cells (e.g. hexagonal Mg-MOF-74): atoms that are
    // actually inside the cell get fractional coords outside [0,1), so floor()
    // shifts them by whole lattice vectors and the neighbor list becomes garbage.
    const double c00 = cell[0][0], c01 = cell[0][1], c02 = cell[0][2];
    const double c10 = cell[1][0], c11 = cell[1][1], c12 = cell[1][2];
    const double c20 = cell[2][0], c21 = cell[2][1], c22 = cell[2][2];
    double frac[3];
    frac[0] = (r.x * (c11 * c22 - c12 * c21) +
               r.y * (c12 * c20 - c10 * c22) +
               r.z * (c10 * c21 - c11 * c20)) /
              det;
    frac[1] = (r.x * (c02 * c21 - c01 * c22) +
               r.y * (c00 * c22 - c02 * c20) +
               r.z * (c01 * c20 - c00 * c21)) /
              det;
    frac[2] = (r.x * (c01 * c12 - c02 * c11) +
               r.y * (c02 * c10 - c00 * c12) +
               r.z * (c00 * c11 - c01 * c10)) /
              det;
    for (int d = 0; d < 3; ++d) {
      if (pbc[d]) {
        frac[d] = frac[d] - std::floor(frac[d]);
      }
    }
    pos[i] = {
        frac[0] * cell[0][0] + frac[1] * cell[1][0] + frac[2] * cell[2][0],
        frac[0] * cell[0][1] + frac[1] * cell[1][1] + frac[2] * cell[2][1],
        frac[0] * cell[0][2] + frac[1] * cell[1][2] + frac[2] * cell[2][2],
    };
  }
}

Vec3 lattice_image_shift(int ix,
                         int iy,
                         int iz,
                         const std::array<std::array<double, 3>, 3>& cell) {
  // ASE convention: cell[i] is the i-th lattice vector (row).
  // Image shift = ix * a + iy * b + iz * c.
  return {
      ix * cell[0][0] + iy * cell[1][0] + iz * cell[2][0],
      ix * cell[0][1] + iy * cell[1][1] + iz * cell[2][1],
      ix * cell[0][2] + iy * cell[1][2] + iz * cell[2][2],
  };
}

std::array<int, 3> image_repeats(const std::array<std::array<double, 3>, 3>& cell,
                                 const std::array<bool, 3>& pbc,
                                 double cutoff) {
  std::array<int, 3> rep = {0, 0, 0};
  for (int d = 0; d < 3; ++d) {
    if (!pbc[d]) {
      rep[d] = 0;
      continue;
    }
    const double len = std::sqrt(cell[d][0] * cell[d][0] + cell[d][1] * cell[d][1] +
                                 cell[d][2] * cell[d][2]);
    rep[d] = static_cast<int>(std::ceil(cutoff / std::max(len, 1e-12)));
  }
  return rep;
}

// Per-center candidate generator (image loop + distance cutoff).
void gen_candidates_for_center(const std::vector<Vec3>& positions,
                               int64_t center,
                               int64_t nbr_begin,
                               int64_t nbr_end,
                               const std::array<std::array<double, 3>, 3>& cell,
                               const std::array<bool, 3>& pbc,
                               const std::array<int, 3>& rep,
                               const NeighborListConfig& config,
                               std::vector<NeighborCandidate>& out) {
  for (int64_t neighbor = nbr_begin; neighbor < nbr_end; ++neighbor) {
    for (int ix = -rep[0]; ix <= rep[0]; ++ix) {
      for (int iy = -rep[1]; iy <= rep[1]; ++iy) {
        for (int iz = -rep[2]; iz <= rep[2]; ++iz) {
          if (!pbc[0] && ix != 0) {
            continue;
          }
          if (!pbc[1] && iy != 0) {
            continue;
          }
          if (!pbc[2] && iz != 0) {
            continue;
          }
          const Vec3 image_shift = lattice_image_shift(ix, iy, iz, cell);
          const Vec3 neighbor_pos = add(positions[neighbor], image_shift);
          const double dist = norm(sub(neighbor_pos, positions[center]));
          if (dist >= config.distance_tolerance && dist <= config.cutoff) {
            out.push_back({neighbor, ix, iy, iz, dist});
          }
        }
      }
    }
  }
}

}  // namespace

torch::Tensor wrap_positions_to_cell(const torch::Tensor& pos,
                                     const torch::Tensor& cell,
                                     const torch::Tensor& pbc) {
  auto cell_mat = cell_from_tensor(cell);
  auto pbc_flags = pbc_from_tensor(pbc);
  auto positions = positions_from_tensor(pos);
  wrap_positions(positions, cell_mat, pbc_flags);
  auto out = torch::empty({static_cast<int64_t>(positions.size()), 3},
                          pos.options().device(torch::kCPU).dtype(torch::kFloat64));
  auto acc = out.accessor<double, 2>();
  for (int64_t i = 0; i < static_cast<int64_t>(positions.size()); ++i) {
    acc[i][0] = positions[i].x;
    acc[i][1] = positions[i].y;
    acc[i][2] = positions[i].z;
  }
  return out.to(pos.device(), pos.scalar_type());
}

NeighborGraph build_neighbor_graph(const torch::Tensor& pos,
                                   const torch::Tensor& cell,
                                   const torch::Tensor& pbc,
                                   const NeighborListConfig& config) {
  if (!pos.defined() || pos.size(0) == 0) {
    throw std::runtime_error("pos must be non-empty");
  }

  auto cell_mat = cell_from_tensor(cell);
  auto pbc_flags = pbc_from_tensor(pbc);
  auto positions = positions_from_tensor(pos);
  wrap_positions(positions, cell_mat, pbc_flags);

  const int64_t n = static_cast<int64_t>(positions.size());
  const auto rep = image_repeats(cell_mat, pbc_flags, config.cutoff);

  std::vector<int64_t> src;
  std::vector<int64_t> dst;
  std::vector<std::array<double, 3>> offsets;

  src.reserve(n * config.max_neighbors);
  dst.reserve(n * config.max_neighbors);
  offsets.reserve(n * config.max_neighbors);

  for (int64_t center = 0; center < n; ++center) {
    std::vector<NeighborCandidate> candidates;
    candidates.reserve(256);

    gen_candidates_for_center(positions, center, /*nbr_begin=*/0, /*nbr_end=*/n,
                              cell_mat, pbc_flags, rep, config, candidates);

    std::sort(candidates.begin(), candidates.end(),
              [](const NeighborCandidate& a, const NeighborCandidate& b) {
                return a.distance < b.distance;
              });
    const int keep = std::min<int>(
        static_cast<int>(candidates.size()), config.max_neighbors);
    for (int k = 0; k < keep; ++k) {
      const auto& c = candidates[k];
      src.push_back(c.neighbor);
      dst.push_back(center);
      offsets.push_back(
          {static_cast<double>(c.ox), static_cast<double>(c.oy),
           static_cast<double>(c.oz)});
    }
  }

  const int64_t e = static_cast<int64_t>(src.size());
  auto edge_index = torch::empty({2, e}, torch::TensorOptions().dtype(torch::kLong));
  auto cell_offsets = torch::empty({e, 3}, pos.options());

  if (e > 0) {
    auto ei_acc = edge_index.accessor<int64_t, 2>();
    auto co_cpu = cell_offsets.to(torch::kFloat64).cpu();
    auto co_acc = co_cpu.accessor<double, 2>();
    for (int64_t i = 0; i < e; ++i) {
      ei_acc[0][i] = src[i];
      ei_acc[1][i] = dst[i];
      co_acc[i][0] = offsets[i][0];
      co_acc[i][1] = offsets[i][1];
      co_acc[i][2] = offsets[i][2];
    }
    cell_offsets = co_cpu.to(pos.dtype());
  }

  return {edge_index, cell_offsets};
}

}  // namespace uma
