#include "uma/neighbor_list.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
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
  // P0.5: the number of periodic images to search along axis d must be based on
  // the INTERPLANAR SPACING (perpendicular distance between the lattice planes
  // normal to that axis), NOT the lattice-vector length |cell[d]|. For a skewed
  // (triclinic) cell |cell[d]| overestimates the spacing, so cutoff/|cell[d]|
  // under-counts the images and edges within the cutoff get silently dropped.
  //
  // Interplanar spacing along axis d = V / |A_d|, where V = |a . (b x c)| is the
  // cell volume and A_d = (the cross product of the OTHER two lattice vectors) is
  // the area vector of the plane spanned by them. For an orthorhombic cell this
  // reduces to |cell[d]|, so the orthorhombic path is unchanged.
  auto cross = [](const std::array<double, 3>& u,
                  const std::array<double, 3>& v) -> std::array<double, 3> {
    return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0]};
  };
  auto dot = [](const std::array<double, 3>& u, const std::array<double, 3>& v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  };
  auto vnorm = [&](const std::array<double, 3>& u) { return std::sqrt(dot(u, u)); };

  const std::array<double, 3> a = {cell[0][0], cell[0][1], cell[0][2]};
  const std::array<double, 3> b = {cell[1][0], cell[1][1], cell[1][2]};
  const std::array<double, 3> c = {cell[2][0], cell[2][1], cell[2][2]};
  const double volume = std::abs(dot(a, cross(b, c)));

  std::array<int, 3> rep = {0, 0, 0};
  for (int d = 0; d < 3; ++d) {
    if (!pbc[d]) {
      rep[d] = 0;
      continue;
    }
    // Area vector of the plane spanned by the other two lattice vectors.
    std::array<double, 3> area_vec;
    if (d == 0)
      area_vec = cross(b, c);
    else if (d == 1)
      area_vec = cross(c, a);
    else
      area_vec = cross(a, b);
    const double area = vnorm(area_vec);
    const double spacing = (area > 1e-12) ? (volume / area) : 1e-12;
    rep[d] = static_cast<int>(std::ceil(cutoff / std::max(spacing, 1e-12)));
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

// True if the cell is (numerically) orthorhombic: off-diagonals ~0 and
// diagonal lengths strictly positive.
bool is_orthorhombic(const std::array<std::array<double, 3>, 3>& cell) {
  double diag_scale = 0.0;
  for (int d = 0; d < 3; ++d) {
    diag_scale = std::max(diag_scale, std::abs(cell[d][d]));
  }
  const double tol = 1e-9 * std::max(diag_scale, 1.0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        if (std::abs(cell[i][j]) <= tol) {
          return false;  // degenerate axis
        }
      } else if (std::abs(cell[i][j]) > tol) {
        return false;  // non-zero off-diagonal
      }
    }
  }
  return true;
}

// Cell-list (linked-cell) per-center candidate generator for orthorhombic
// boxes with >= 3 bins along every periodic axis. Reproduces EXACTLY the same
// (neighbor, image-offset, distance) candidate set as
// gen_candidates_for_center: same cutoff test, same self/image exclusion via
// distance_tolerance, same integer image triples.
//
// Grid: ncell[d] = floor(L_d / cutoff) bins on periodic axes (>= 3 required by
// the caller so the 27-cell stencil covers the full cutoff sphere and any
// periodic wrap is at most +/-1 image). Non-periodic axes use a single bin and
// contribute no image offset. A neighbor found in stencil cell (ci + dc) whose
// index wraps by w full periods contributes image-offset component w on that
// axis, matching the all-pairs `ix/iy/iz` image indices.
struct CellGrid {
  std::array<int, 3> ncell;
  std::array<double, 3> len;  // box length along each axis (L_d for periodic)
  std::array<bool, 3> pbc;
  std::vector<std::vector<int64_t>> buckets;  // flattened ncell[0]*ncell[1]*ncell[2]
  std::array<int, 3> cell_of(const Vec3& p) const {
    std::array<int, 3> c{0, 0, 0};
    for (int d = 0; d < 3; ++d) {
      if (ncell[d] <= 1) {
        c[d] = 0;
        continue;
      }
      const double coord = (d == 0 ? p.x : (d == 1 ? p.y : p.z));
      double f = coord / len[d];  // wrapped positions => f in [0,1)
      int idx = static_cast<int>(std::floor(f * ncell[d]));
      // Guard against FP rounding at the upper boundary.
      if (idx < 0) idx = 0;
      if (idx >= ncell[d]) idx = ncell[d] - 1;
      c[d] = idx;
    }
    return c;
  }
  int64_t flat(int cx, int cy, int cz) const {
    return (static_cast<int64_t>(cx) * ncell[1] + cy) * ncell[2] + cz;
  }
};

bool build_cell_grid(const std::vector<Vec3>& positions,
                     const std::array<std::array<double, 3>, 3>& cell,
                     const std::array<bool, 3>& pbc,
                     const NeighborListConfig& config,
                     CellGrid& grid) {
  for (int d = 0; d < 3; ++d) {
    grid.pbc[d] = pbc[d];
    grid.len[d] = std::abs(cell[d][d]);
    if (pbc[d]) {
      const int nc = static_cast<int>(std::floor(grid.len[d] / config.cutoff));
      if (nc < 3) {
        return false;  // stencil would not cover the cutoff sphere / images
      }
      grid.ncell[d] = nc;
    } else {
      grid.ncell[d] = 1;
    }
  }
  const int64_t total =
      static_cast<int64_t>(grid.ncell[0]) * grid.ncell[1] * grid.ncell[2];
  grid.buckets.assign(static_cast<size_t>(total), {});
  for (int64_t i = 0; i < static_cast<int64_t>(positions.size()); ++i) {
    const auto c = grid.cell_of(positions[i]);
    grid.buckets[static_cast<size_t>(grid.flat(c[0], c[1], c[2]))].push_back(i);
  }
  return true;
}

void gen_candidates_for_center_celllist(const std::vector<Vec3>& positions,
                                        int64_t center,
                                        const CellGrid& grid,
                                        const std::array<std::array<double, 3>, 3>& cell,
                                        const std::array<bool, 3>& pbc,
                                        const NeighborListConfig& config,
                                        std::vector<NeighborCandidate>& out) {
  const auto cc = grid.cell_of(positions[center]);
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        const int dc[3] = {dx, dy, dz};
        int nb[3];
        int wrap[3];  // integer image offset contributed by this stencil step
        bool skip = false;
        for (int d = 0; d < 3; ++d) {
          const int raw = (d == 0 ? cc[0] : (d == 1 ? cc[1] : cc[2])) + dc[d];
          const int nd = grid.ncell[d];
          if (!pbc[d]) {
            if (raw < 0 || raw >= nd) {
              skip = true;  // no periodic image across a non-periodic axis
              break;
            }
            nb[d] = raw;
            wrap[d] = 0;
          } else {
            // Floor-divide to fold raw into [0, nd) and record the wrap count.
            int w = 0;
            int idx = raw;
            while (idx < 0) {
              idx += nd;
              w -= 1;
            }
            while (idx >= nd) {
              idx -= nd;
              w += 1;
            }
            nb[d] = idx;
            wrap[d] = w;
          }
        }
        if (skip) {
          continue;
        }
        const int ox = wrap[0];
        const int oy = wrap[1];
        const int oz = wrap[2];
        const Vec3 image_shift = lattice_image_shift(ox, oy, oz, cell);
        const auto& bucket =
            grid.buckets[static_cast<size_t>(grid.flat(nb[0], nb[1], nb[2]))];
        for (int64_t neighbor : bucket) {
          const Vec3 neighbor_pos = add(positions[neighbor], image_shift);
          const double dist = norm(sub(neighbor_pos, positions[center]));
          if (dist >= config.distance_tolerance && dist <= config.cutoff) {
            out.push_back({neighbor, ox, oy, oz, dist});
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

namespace {

// Shared sort-by-distance + max_neighbors truncation + tensor emission. This is
// the UNCHANGED tail of the original build_neighbor_graph; both the all-pairs
// and cell-list paths funnel through here so their output is bit-identical given
// the same candidate sets.
NeighborGraph emit_graph(
    const torch::Tensor& pos,
    const std::vector<std::vector<NeighborCandidate>>& per_center,
    const NeighborListConfig& config) {
  const int64_t n = static_cast<int64_t>(per_center.size());
  std::vector<int64_t> src;
  std::vector<int64_t> dst;
  std::vector<std::array<double, 3>> offsets;
  src.reserve(n * config.max_neighbors);
  dst.reserve(n * config.max_neighbors);
  offsets.reserve(n * config.max_neighbors);

  for (int64_t center = 0; center < n; ++center) {
    std::vector<NeighborCandidate> candidates = per_center[center];
    // Sort by distance, with a total-order tie-break on (neighbor, ox, oy, oz).
    // The tie-break is REQUIRED for identical output: with plain distance-only
    // sorting the original std::sort is unstable, so when max_neighbors cuts
    // through a shell of exactly-equal distances the surviving neighbors depend
    // on generation order — which differs between the all-pairs and cell-list
    // paths. A deterministic total order makes truncation order-independent, so
    // both paths keep the SAME edges (and makes the all-pairs path itself
    // reproducible run-to-run).
    std::sort(candidates.begin(), candidates.end(),
              [](const NeighborCandidate& a, const NeighborCandidate& b) {
                if (a.distance != b.distance) return a.distance < b.distance;
                if (a.neighbor != b.neighbor) return a.neighbor < b.neighbor;
                if (a.ox != b.ox) return a.ox < b.ox;
                if (a.oy != b.oy) return a.oy < b.oy;
                return a.oz < b.oz;
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

bool allpairs_forced() {
  const char* v = std::getenv("UMA_NL_ALLPAIRS");
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

}  // namespace

// O(N^2) all-pairs neighbor graph (original implementation, always correct).
NeighborGraph build_neighbor_graph_allpairs(const torch::Tensor& pos,
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

  std::vector<std::vector<NeighborCandidate>> per_center(n);
  for (int64_t center = 0; center < n; ++center) {
    per_center[center].reserve(256);
    gen_candidates_for_center(positions, center, /*nbr_begin=*/0, /*nbr_end=*/n,
                              cell_mat, pbc_flags, rep, config,
                              per_center[center]);
  }
  return emit_graph(pos, per_center, config);
}

// O(N * neighbors) cell-list neighbor graph. Requires an orthorhombic box with
// >= 3 bins along every periodic axis; returns false-equivalent (throws) only
// if misused. Produces candidate sets identical to the all-pairs path.
NeighborGraph build_neighbor_graph_celllist(const torch::Tensor& pos,
                                            const torch::Tensor& cell,
                                            const torch::Tensor& pbc,
                                            const NeighborListConfig& config) {
  auto cell_mat = cell_from_tensor(cell);
  auto pbc_flags = pbc_from_tensor(pbc);
  auto positions = positions_from_tensor(pos);
  wrap_positions(positions, cell_mat, pbc_flags);

  const int64_t n = static_cast<int64_t>(positions.size());
  CellGrid grid;
  if (!build_cell_grid(positions, cell_mat, pbc_flags, config, grid)) {
    // Not enough bins — caller should have gated on this; fall back.
    return build_neighbor_graph_allpairs(pos, cell, pbc, config);
  }

  std::vector<std::vector<NeighborCandidate>> per_center(n);
  for (int64_t center = 0; center < n; ++center) {
    per_center[center].reserve(256);
    gen_candidates_for_center_celllist(positions, center, grid, cell_mat,
                                       pbc_flags, config, per_center[center]);
  }
  return emit_graph(pos, per_center, config);
}

NeighborGraph build_neighbor_graph(const torch::Tensor& pos,
                                   const torch::Tensor& cell,
                                   const torch::Tensor& pbc,
                                   const NeighborListConfig& config) {
  if (!pos.defined() || pos.size(0) == 0) {
    throw std::runtime_error("pos must be non-empty");
  }

  // A/B switch: force the original O(N^2) path for validation.
  if (allpairs_forced()) {
    return build_neighbor_graph_allpairs(pos, cell, pbc, config);
  }

  // Dispatch to the cell-list only when it is provably safe: orthorhombic box
  // with >= 3 bins along every periodic axis (floor(L_d/cutoff) >= 3). Small or
  // triclinic systems use the all-pairs path (fast for small N, always correct).
  auto cell_mat = cell_from_tensor(cell);
  auto pbc_flags = pbc_from_tensor(pbc);
  if (is_orthorhombic(cell_mat)) {
    bool ok = true;
    for (int d = 0; d < 3; ++d) {
      if (pbc_flags[d]) {
        const double len = std::abs(cell_mat[d][d]);
        if (static_cast<int>(std::floor(len / config.cutoff)) < 3) {
          ok = false;
          break;
        }
      }
    }
    if (ok) {
      return build_neighbor_graph_celllist(pos, cell, pbc, config);
    }
  }
  return build_neighbor_graph_allpairs(pos, cell, pbc, config);
}

}  // namespace uma
