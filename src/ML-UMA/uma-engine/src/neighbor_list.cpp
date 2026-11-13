#include "uma/neighbor_list.h"
#include "uma/runtime_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
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
  const char* v = uma::runtime_get("UMA_NL_ALLPAIRS");
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

bool nitro_n1_requested() {
  const char* v = runtime_get("UMA_NITRO_N1");
  return v != nullptr && v[0] == '1';
}

int64_t nitro_n1_cell_cap() {
  const char* v = runtime_get("UMA_NITRO_N1_CELL_CAP");
  if (v == nullptr || v[0] == '\0') {
    return 64;
  }
  char* end = nullptr;
  const long x = std::strtol(v, &end, 10);
  if (end == v || x < 1 || x > 100000) {
    return 64;
  }
  return static_cast<int64_t>(x);
}

const char* nitro_n1_overflow_reason(int64_t code) {
  switch (code) {
    case 1:
      return "cell occupancy exceeded UMA_NITRO_N1_CELL_CAP";
    case 2:
      return "all-pairs candidate row exceeds the packed limit";
    case 3:
      return "atom count >= 2^20 does not fit the tie-break key";
    case 4:
      return "image index outside +/-7";
    default:
      return "unknown overflow";
  }
}

namespace {

constexpr int64_t kTieNeighborBits = 20;
constexpr int64_t kMaxPackAtoms = 1LL << kTieNeighborBits;
constexpr int64_t kMaxImage = 7;
constexpr int64_t kMaxRow = 65536;

torch::Tensor wrap_torch(const torch::Tensor& pos64,
                         const std::array<std::array<double, 3>, 3>& cell,
                         const std::array<bool, 3>& pbc) {
  const double c00 = cell[0][0], c01 = cell[0][1], c02 = cell[0][2];
  const double c10 = cell[1][0], c11 = cell[1][1], c12 = cell[1][2];
  const double c20 = cell[2][0], c21 = cell[2][1], c22 = cell[2][2];
  const double det = c00 * (c11 * c22 - c12 * c21) - c01 * (c10 * c22 - c12 * c20) +
                     c02 * (c10 * c21 - c11 * c20);
  if (std::abs(det) < 1e-12) {
    throw std::runtime_error("N1 wrap: degenerate cell");
  }
  const double inv_h[3][3] = {
      {(c11 * c22 - c12 * c21) / det, (c02 * c21 - c01 * c22) / det,
       (c01 * c12 - c02 * c11) / det},
      {(c12 * c20 - c10 * c22) / det, (c00 * c22 - c02 * c20) / det,
       (c02 * c10 - c00 * c12) / det},
      {(c10 * c21 - c11 * c20) / det, (c01 * c20 - c00 * c21) / det,
       (c00 * c11 - c01 * c10) / det},
  };
  const double cell_h[3][3] = {
      {c00, c01, c02},
      {c10, c11, c12},
      {c20, c21, c22},
  };
  auto dev = pos64.device();
  auto inv = torch::from_blob(const_cast<double*>(inv_h[0]), {3, 3}, torch::kFloat64)
                 .clone()
                 .to(dev);
  auto cell_t = torch::from_blob(const_cast<double*>(cell_h[0]), {3, 3}, torch::kFloat64)
                    .clone()
                    .to(dev);
  auto frac = pos64.matmul(inv);
  std::vector<torch::Tensor> folded;
  folded.reserve(3);
  for (int d = 0; d < 3; ++d) {
    auto fd = frac.select(1, d);
    if (pbc[d]) {
      fd = fd - torch::floor(fd);
    }
    folded.push_back(fd);
  }
  return torch::stack(folded, 1).matmul(cell_t);
}

DeviceNeighborBuild emit_with_reference_order(
    const torch::Tensor& pos_orig,
    const torch::Tensor& wrapped,
    const std::array<std::array<double, 3>, 3>& cell,
    const std::array<bool, 3>& pbc,
    const NeighborListConfig& config,
    const torch::Tensor& neighbor,
    const torch::Tensor& ox,
    const torch::Tensor& oy,
    const torch::Tensor& oz,
    const torch::Tensor& keep) {
  // The candidate set is chosen on the tensor's device. Order and truncation
  // then use the reference builder's comparator and its scalar distances, so a
  // 1 ulp difference in a torch reduction cannot reorder an equal shell.
  const int64_t n = neighbor.size(0);
  DeviceNeighborBuild out;
  out.overflow = 0;
  out.wrapped_pos = wrapped;
  std::vector<std::vector<NeighborCandidate>> per_center(static_cast<size_t>(n));
  auto nz = keep.nonzero();
  if (nz.size(0) > 0) {
    auto rows_d = nz.select(1, 0);
    auto cols_d = nz.select(1, 1);
    auto rows = rows_d.to(torch::kCPU);
    auto nb = neighbor.index({rows_d, cols_d}).to(torch::kCPU);
    auto oxc = ox.index({rows_d, cols_d}).to(torch::kCPU);
    auto oyc = oy.index({rows_d, cols_d}).to(torch::kCPU);
    auto ozc = oz.index({rows_d, cols_d}).to(torch::kCPU);
    auto positions = positions_from_tensor(pos_orig);
    wrap_positions(positions, cell, pbc);
    const int64_t e = rows.size(0);
    auto rows_a = rows.accessor<int64_t, 1>();
    auto nb_a = nb.accessor<int64_t, 1>();
    auto ox_a = oxc.accessor<int64_t, 1>();
    auto oy_a = oyc.accessor<int64_t, 1>();
    auto oz_a = ozc.accessor<int64_t, 1>();
    for (int64_t i = 0; i < e; ++i) {
      NeighborCandidate c;
      c.neighbor = nb_a[i];
      c.ox = static_cast<int>(ox_a[i]);
      c.oy = static_cast<int>(oy_a[i]);
      c.oz = static_cast<int>(oz_a[i]);
      const Vec3 shift = lattice_image_shift(c.ox, c.oy, c.oz, cell);
      const Vec3 nbr_pos = add(positions[static_cast<size_t>(c.neighbor)], shift);
      c.distance = norm(sub(nbr_pos, positions[static_cast<size_t>(rows_a[i])]));
      per_center[static_cast<size_t>(rows_a[i])].push_back(c);
    }
  }
  out.graph = emit_graph(pos_orig, per_center, config);
  out.graph.edge_index = out.graph.edge_index.to(wrapped.device());
  out.graph.cell_offsets = out.graph.cell_offsets.to(wrapped.device(), wrapped.scalar_type());
  return out;
}

DeviceNeighborBuild build_allpairs_device(
    const torch::Tensor& pos_orig,
    const torch::Tensor& wrapped,
    const std::array<std::array<double, 3>, 3>& cell,
    const std::array<bool, 3>& pbc,
    const NeighborListConfig& config) {
  const int64_t n = wrapped.size(0);
  const auto rep = image_repeats(cell, pbc, config.cutoff);
  for (int d = 0; d < 3; ++d) {
    if (rep[d] > kMaxImage) {
      DeviceNeighborBuild bad;
      bad.overflow = 4;
      bad.wrapped_pos = wrapped;
      return bad;
    }
  }
  std::vector<std::array<int64_t, 3>> images;
  for (int ix = -rep[0]; ix <= rep[0]; ++ix) {
    for (int iy = -rep[1]; iy <= rep[1]; ++iy) {
      for (int iz = -rep[2]; iz <= rep[2]; ++iz) {
        if (!pbc[0] && ix != 0) continue;
        if (!pbc[1] && iy != 0) continue;
        if (!pbc[2] && iz != 0) continue;
        images.push_back({ix, iy, iz});
      }
    }
  }
  const int64_t nimg = static_cast<int64_t>(images.size());
  const int64_t width = n * nimg;
  if (width > kMaxRow || width <= 0) {
    DeviceNeighborBuild bad;
    bad.overflow = 2;
    bad.wrapped_pos = wrapped;
    return bad;
  }
  const auto dev = wrapped.device();
  auto iopt = torch::TensorOptions().dtype(torch::kLong).device(dev);
  std::vector<int64_t> ox_h, oy_h, oz_h;
  ox_h.reserve(static_cast<size_t>(nimg));
  for (const auto& im : images) {
    ox_h.push_back(im[0]);
    oy_h.push_back(im[1]);
    oz_h.push_back(im[2]);
  }
  auto ox_i = torch::from_blob(ox_h.data(), {nimg}, torch::kLong).clone().to(dev);
  auto oy_i = torch::from_blob(oy_h.data(), {nimg}, torch::kLong).clone().to(dev);
  auto oz_i = torch::from_blob(oz_h.data(), {nimg}, torch::kLong).clone().to(dev);
  auto cell_t = torch::from_blob(const_cast<double*>(&cell[0][0]), {3, 3}, torch::kFloat64)
                    .clone()
                    .to(dev);
  auto img = torch::stack({ox_i, oy_i, oz_i}, 1).to(wrapped.dtype());
  auto shift = img.matmul(cell_t);

  // [N center, N neighbor, nimg]
  auto dvec = wrapped.view({1, n, 1, 3}) + shift.view({1, 1, nimg, 3}) -
              wrapped.view({n, 1, 1, 3});
  auto dist = dvec.square().sum(-1).sqrt().reshape({n, width});
  auto nbr = torch::arange(n, iopt).view({1, n, 1}).expand({n, n, nimg}).reshape({n, width});
  auto ox = ox_i.view({1, 1, nimg}).expand({n, n, nimg}).reshape({n, width});
  auto oy = oy_i.view({1, 1, nimg}).expand({n, n, nimg}).reshape({n, width});
  auto oz = oz_i.view({1, 1, nimg}).expand({n, n, nimg}).reshape({n, width});
  auto keep = dist.ge(config.distance_tolerance) & dist.le(config.cutoff);
  return emit_with_reference_order(pos_orig, wrapped, cell, pbc, config, nbr, ox, oy, oz, keep);
}

DeviceNeighborBuild build_celllist_device(
    const torch::Tensor& pos_orig,
    const torch::Tensor& wrapped,
    const std::array<std::array<double, 3>, 3>& cell,
    const std::array<bool, 3>& pbc,
    const NeighborListConfig& config,
    int64_t cell_cap) {
  const int64_t n = wrapped.size(0);
  const auto dev = wrapped.device();
  int64_t nc[3];
  double len[3];
  for (int d = 0; d < 3; ++d) {
    len[d] = std::abs(cell[d][d]);
    nc[d] = pbc[d] ? static_cast<int64_t>(std::floor(len[d] / config.cutoff)) : 1;
  }
  auto iopt = torch::TensorOptions().dtype(torch::kLong).device(dev);
  std::vector<torch::Tensor> ci;
  for (int d = 0; d < 3; ++d) {
    auto coord = wrapped.select(1, d);
    auto idx = torch::floor(coord / len[d] * static_cast<double>(nc[d])).to(torch::kLong);
    ci.push_back(idx.clamp(0, nc[d] - 1));
  }
  auto flat = (ci[0] * nc[1] + ci[1]) * nc[2] + ci[2];
  const int64_t ncells = nc[0] * nc[1] * nc[2];
  auto counts = torch::bincount(flat, torch::Tensor(), ncells);
  const int64_t maxc = counts.max().item<int64_t>();
  if (maxc > cell_cap) {
    DeviceNeighborBuild bad;
    bad.overflow = 1;
    bad.wrapped_pos = wrapped;
    return bad;
  }
  auto order = torch::argsort(flat, /*stable=*/true, /*dim=*/0, /*descending=*/false);
  auto flat_s = flat.index_select(0, order);
  auto starts = torch::cumsum(counts, 0) - counts;
  auto slot = torch::arange(n, iopt) - starts.index_select(0, flat_s);
  auto packed = torch::full({ncells, cell_cap}, static_cast<int64_t>(-1), iopt);
  packed.index_put_({flat_s, slot}, order);

  int64_t dc_h[27][3];
  int ndc = 0;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      for (int z = -1; z <= 1; ++z) {
        dc_h[ndc][0] = x;
        dc_h[ndc][1] = y;
        dc_h[ndc][2] = z;
        ++ndc;
      }
    }
  }
  auto dc = torch::from_blob(dc_h, {27, 3}, torch::kLong).clone().to(dev);
  std::vector<torch::Tensor> idx_d, w_d;
  torch::Tensor skip;
  for (int d = 0; d < 3; ++d) {
    auto rd = ci[d].unsqueeze(1) + dc.select(1, d).unsqueeze(0);
    torch::Tensor idx;
    torch::Tensor w;
    if (!pbc[d]) {
      auto outside = rd.lt(0) | rd.gt(nc[d] - 1);
      skip = skip.defined() ? (skip | outside) : outside;
      idx = rd.clamp(0, nc[d] - 1);
      w = torch::zeros_like(rd);
    } else {
      auto neg = rd.lt(0);
      auto hi = rd.gt(nc[d] - 1);
      idx = torch::where(neg, rd + nc[d], torch::where(hi, rd - nc[d], rd));
      w = torch::where(neg, torch::full_like(rd, static_cast<int64_t>(-1)),
                       torch::where(hi, torch::full_like(rd, static_cast<int64_t>(1)),
                                    torch::zeros_like(rd)));
    }
    idx_d.push_back(idx);
    w_d.push_back(w);
  }
  auto pf = (idx_d[0] * nc[1] + idx_d[1]) * nc[2] + idx_d[2];
  auto gathered = packed.index_select(0, pf.reshape(-1)).view({n, 27, cell_cap});
  auto wstack = torch::stack(w_d, -1);
  auto cell_t = torch::from_blob(const_cast<double*>(&cell[0][0]), {3, 3}, torch::kFloat64)
                    .clone()
                    .to(dev);
  auto shift = wstack.to(wrapped.dtype()).matmul(cell_t);
  auto nbr_pos = wrapped.index_select(0, gathered.clamp_min(0).reshape(-1))
                     .view({n, 27, cell_cap, 3});
  auto dvec = nbr_pos + shift.unsqueeze(2) - wrapped.view({n, 1, 1, 3});
  auto dist = dvec.square().sum(-1).sqrt();
  auto keep = gathered.ge(0) & dist.ge(config.distance_tolerance) & dist.le(config.cutoff);
  if (skip.defined()) {
    keep = keep & skip.logical_not().unsqueeze(-1);
  }
  const int64_t width = 27 * cell_cap;
  auto nbr = gathered.reshape({n, width});
  auto dist_f = dist.reshape({n, width});
  auto keep_f = keep.reshape({n, width});
  auto ox = w_d[0].unsqueeze(-1).expand({n, 27, cell_cap}).reshape({n, width});
  auto oy = w_d[1].unsqueeze(-1).expand({n, 27, cell_cap}).reshape({n, width});
  auto oz = w_d[2].unsqueeze(-1).expand({n, 27, cell_cap}).reshape({n, width});
  return emit_with_reference_order(pos_orig, wrapped, cell, pbc, config, nbr, ox, oy, oz,
                                   keep_f);
}

}  // namespace

DeviceNeighborBuild build_neighbor_graph_device(const torch::Tensor& pos,
                                               const torch::Tensor& cell,
                                               const torch::Tensor& pbc,
                                               const NeighborListConfig& config,
                                               int64_t cell_cap) {
  if (!pos.defined() || pos.dim() != 2 || pos.size(1) != 3 || pos.size(0) == 0) {
    throw std::runtime_error("N1 pos must be non-empty [N,3]");
  }
  if (config.cutoff <= 0 || cell_cap < 1) {
    throw std::runtime_error("N1 cutoff and cell cap must be positive");
  }
  const int64_t n = pos.size(0);
  if (n >= kMaxPackAtoms) {
    DeviceNeighborBuild bad;
    bad.overflow = 3;
    bad.wrapped_pos = pos;
    return bad;
  }
  const auto cell_mat = cell_from_tensor(cell);
  const auto pbc_flags = pbc_from_tensor(pbc);
  auto wrapped = wrap_torch(pos.to(torch::kFloat64).contiguous(), cell_mat, pbc_flags);

  bool celllist = is_orthorhombic(cell_mat);
  if (celllist) {
    for (int d = 0; d < 3; ++d) {
      if (!pbc_flags[d]) continue;
      const double len = std::abs(cell_mat[d][d]);
      if (static_cast<int>(std::floor(len / config.cutoff)) < 3) {
        celllist = false;
        break;
      }
    }
  }
  DeviceNeighborBuild built =
      celllist ? build_celllist_device(pos, wrapped, cell_mat, pbc_flags, config, cell_cap)
               : build_allpairs_device(pos, wrapped, cell_mat, pbc_flags, config);
  if (built.overflow != 0 && !built.wrapped_pos.defined()) {
    built.wrapped_pos = wrapped;
  }
  if (built.wrapped_pos.defined()) {
    built.wrapped_pos = built.wrapped_pos.to(pos.device(), pos.scalar_type()).contiguous();
  }
  return built;
}

}  // namespace uma
