// Step 16 (N1-b). FP64 SYCL candidate builder. The cutoff test is the same
// comparison the host builder uses. Edge order is not decided here.

#include "uma/n1b_graph.h"

#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace uma {
namespace {

struct WrapCell {
  double c[3][3];
  double det;
  int pbc[3];
};

WrapCell make_wrap(const double cell[3][3], const int pbc[3]) {
  WrapCell w{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) w.c[i][j] = cell[i][j];
    w.pbc[i] = pbc[i];
  }
  const double c00 = cell[0][0], c01 = cell[0][1], c02 = cell[0][2];
  const double c10 = cell[1][0], c11 = cell[1][1], c12 = cell[1][2];
  const double c20 = cell[2][0], c21 = cell[2][1], c22 = cell[2][2];
  w.det = c00 * (c11 * c22 - c12 * c21) - c01 * (c10 * c22 - c12 * c20) +
          c02 * (c10 * c21 - c11 * c20);
  return w;
}

void wrap_positions_sycl(sycl::queue& q, const double* pos, double* out, int64_t n, WrapCell w) {
  if (std::abs(w.det) < 1e-12) {
    throw std::runtime_error("N1-b wrap: degenerate cell");
  }
  q.parallel_for<class N1bWrap>(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> id) {
      const int64_t i = static_cast<int64_t>(id[0]);
      const double x = pos[i * 3 + 0];
      const double y = pos[i * 3 + 1];
      const double z = pos[i * 3 + 2];
      const double c00 = w.c[0][0], c01 = w.c[0][1], c02 = w.c[0][2];
      const double c10 = w.c[1][0], c11 = w.c[1][1], c12 = w.c[1][2];
      const double c20 = w.c[2][0], c21 = w.c[2][1], c22 = w.c[2][2];
      double frac[3];
      frac[0] = (x * (c11 * c22 - c12 * c21) + y * (c12 * c20 - c10 * c22) +
                 z * (c10 * c21 - c11 * c20)) /
                w.det;
      frac[1] = (x * (c02 * c21 - c01 * c22) + y * (c00 * c22 - c02 * c20) +
                 z * (c01 * c20 - c00 * c21)) /
                w.det;
      frac[2] = (x * (c01 * c12 - c02 * c11) + y * (c02 * c10 - c00 * c12) +
                 z * (c00 * c11 - c01 * c10)) /
                w.det;
      for (int d = 0; d < 3; ++d) {
        if (w.pbc[d]) frac[d] = frac[d] - sycl::floor(frac[d]);
      }
      out[i * 3 + 0] = frac[0] * c00 + frac[1] * c10 + frac[2] * c20;
      out[i * 3 + 1] = frac[0] * c01 + frac[1] * c11 + frac[2] * c21;
      out[i * 3 + 2] = frac[0] * c02 + frac[1] * c12 + frac[2] * c22;
    }).wait();
}

sycl::queue& xpu_queue() { return c10::xpu::getCurrentXPUStream().queue(); }

}  // namespace

N1bCandidates n1b_cell_candidates(const torch::Tensor& pos64, const double cell[3][3],
                                 const int pbc[3], double cutoff, double tol, int64_t cell_cap,
                                 const int nc[3], const double len[3]) {
  const int64_t n = pos64.size(0);
  const int64_t ncells = static_cast<int64_t>(nc[0]) * nc[1] * nc[2];
  const int64_t width = 27 * cell_cap;
  std::cerr << "N1B_CELL n=" << n << " ncells=" << ncells << " cap=" << cell_cap << "\n" << std::flush;
  auto dev = pos64.device();
  auto wrapped = torch::empty({n, 3}, pos64.options());
  auto flat = torch::empty({n}, torch::TensorOptions().dtype(torch::kLong).device(dev));
  auto counts = torch::zeros({ncells}, torch::TensorOptions().dtype(torch::kInt).device(dev));

  sycl::queue& q = xpu_queue();
  const WrapCell w = make_wrap(cell, pbc);
  wrap_positions_sycl(q, pos64.data_ptr<double>(), wrapped.data_ptr<double>(), n, w);

  const int nc0 = nc[0], nc1 = nc[1], nc2 = nc[2];
  const double len0 = len[0], len1 = len[1], len2 = len[2];
  auto* wp = wrapped.data_ptr<double>();
  auto* fp = flat.data_ptr<int64_t>();
  auto* cp = counts.data_ptr<int>();
  q.parallel_for<class N1bBin>(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> id) {
      const int64_t i = static_cast<int64_t>(id[0]);
      const double coord[3] = {wp[i * 3], wp[i * 3 + 1], wp[i * 3 + 2]};
      const double lens[3] = {len0, len1, len2};
      const int ncs[3] = {nc0, nc1, nc2};
      int idx[3];
      for (int d = 0; d < 3; ++d) {
        int b = 0;
        if (ncs[d] > 1) {
          b = static_cast<int>(sycl::floor(coord[d] / lens[d] * static_cast<double>(ncs[d])));
          if (b < 0) b = 0;
          if (b >= ncs[d]) b = ncs[d] - 1;
        }
        idx[d] = b;
      }
      const int64_t f = (static_cast<int64_t>(idx[0]) * nc1 + idx[1]) * nc2 + idx[2];
      fp[i] = f;
      sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(cp[f])
          .fetch_add(1);
    }).wait();

  const int maxc = counts.to(torch::kCPU).max().item<int>();
  std::cerr << "N1B_BIN maxc=" << maxc << "\n" << std::flush;
  N1bCandidates out;
  out.wrapped = wrapped;
  if (static_cast<int64_t>(maxc) > cell_cap) {
    out.overflow = 1;
    return out;
  }

  auto counters = torch::zeros({ncells}, counts.options());
  auto packed = torch::full({ncells * cell_cap}, static_cast<int64_t>(-1), flat.options());
  auto* ctr = counters.data_ptr<int>();
  auto* pk = packed.data_ptr<int64_t>();
  const int64_t cap = cell_cap;
  q.parallel_for<class N1bPack>(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> id) {
      const int64_t i = static_cast<int64_t>(id[0]);
      const int64_t f = fp[i];
      const int slot =
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(ctr[f])
              .fetch_add(1);
      if (slot >= 0 && static_cast<int64_t>(slot) < cap) pk[f * cap + slot] = i;
    }).wait();

  auto nbr = torch::empty({n, width}, flat.options());
  auto ox = torch::empty({n, width}, flat.options());
  auto oy = torch::empty({n, width}, flat.options());
  auto oz = torch::empty({n, width}, flat.options());
  auto keep = torch::zeros({n, width}, torch::TensorOptions().dtype(torch::kByte).device(dev));
  auto* np = nbr.data_ptr<int64_t>();
  auto* oxp = ox.data_ptr<int64_t>();
  auto* oyp = oy.data_ptr<int64_t>();
  auto* ozp = oz.data_ptr<int64_t>();
  auto* kp = keep.data_ptr<uint8_t>();
  const double c00 = cell[0][0], c01 = cell[0][1], c02 = cell[0][2];
  const double c10 = cell[1][0], c11 = cell[1][1], c12 = cell[1][2];
  const double c20 = cell[2][0], c21 = cell[2][1], c22 = cell[2][2];
  const int p0 = pbc[0], p1 = pbc[1], p2 = pbc[2];
  q.parallel_for<class N1bStencil>(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> id) {
      const int64_t i = static_cast<int64_t>(id[0]);
      const double xi = wp[i * 3], yi = wp[i * 3 + 1], zi = wp[i * 3 + 2];
      const double lens[3] = {len0, len1, len2};
      const int ncs[3] = {nc0, nc1, nc2};
      const int pb[3] = {p0, p1, p2};
      int ci[3];
      for (int d = 0; d < 3; ++d) {
        const double coord = d == 0 ? xi : (d == 1 ? yi : zi);
        int b = 0;
        if (ncs[d] > 1) {
          b = static_cast<int>(sycl::floor(coord / lens[d] * static_cast<double>(ncs[d])));
          if (b < 0) b = 0;
          if (b >= ncs[d]) b = ncs[d] - 1;
        }
        ci[d] = b;
      }
      for (int s = 0; s < 27; ++s) {
        const int dc[3] = {s / 9 - 1, (s / 3) % 3 - 1, s % 3 - 1};
        int idx[3];
        int img[3];
        bool skip = false;
        for (int d = 0; d < 3; ++d) {
          const int rd = ci[d] + dc[d];
          if (!pb[d]) {
            img[d] = 0;
            if (rd < 0 || rd > ncs[d] - 1) {
              skip = true;
              idx[d] = 0;
            } else {
              idx[d] = rd;
            }
          } else if (rd < 0) {
            idx[d] = rd + ncs[d];
            img[d] = -1;
          } else if (rd > ncs[d] - 1) {
            idx[d] = rd - ncs[d];
            img[d] = 1;
          } else {
            idx[d] = rd;
            img[d] = 0;
          }
        }
        const int64_t pf =
            skip ? 0 : (static_cast<int64_t>(idx[0]) * nc1 + idx[1]) * nc2 + idx[2];
        const double sx = img[0] * c00 + img[1] * c10 + img[2] * c20;
        const double sy = img[0] * c01 + img[1] * c11 + img[2] * c21;
        const double sz = img[0] * c02 + img[1] * c12 + img[2] * c22;
        for (int64_t slot = 0; slot < cap; ++slot) {
          const int64_t col = static_cast<int64_t>(s) * cap + slot;
          const int64_t row = i * (27 * cap) + col;
          const int64_t nb = skip ? -1 : pk[pf * cap + slot];
          oxp[row] = img[0];
          oyp[row] = img[1];
          ozp[row] = img[2];
          np[row] = nb;
          if (nb < 0) {
            kp[row] = 0;
            continue;
          }
          const double dx = wp[nb * 3] + sx - xi;
          const double dy = wp[nb * 3 + 1] + sy - yi;
          const double dz = wp[nb * 3 + 2] + sz - zi;
          const double dist = sycl::sqrt(dx * dx + dy * dy + dz * dz);
          kp[row] = (dist >= tol && dist <= cutoff) ? 1 : 0;
        }
      }
    }).wait();

  out.neighbor = nbr;
  out.ox = ox;
  out.oy = oy;
  out.oz = oz;
  out.keep = keep.to(torch::kBool);
  std::cerr << "N1B_CELL_DONE\n" << std::flush;
  return out;
}

N1bCandidates n1b_pair_candidates(const torch::Tensor& pos64, const double cell[3][3],
                                 const int pbc[3], double cutoff, double tol,
                                 const int64_t* images, int64_t nimg) {
  const int64_t n = pos64.size(0);
  const int64_t width = n * nimg;
  std::cerr << "N1B_PAIRS n=" << n << " nimg=" << nimg << "\n" << std::flush;
  auto dev = pos64.device();
  auto wrapped = torch::empty({n, 3}, pos64.options());
  sycl::queue& q = xpu_queue();
  wrap_positions_sycl(q, pos64.data_ptr<double>(), wrapped.data_ptr<double>(), n, make_wrap(cell, pbc));

  auto img = torch::from_blob(const_cast<int64_t*>(images), {nimg, 3}, torch::kLong).clone().to(dev);
  auto nbr = torch::empty({n, width}, torch::TensorOptions().dtype(torch::kLong).device(dev));
  auto ox = torch::empty({n, width}, nbr.options());
  auto oy = torch::empty({n, width}, nbr.options());
  auto oz = torch::empty({n, width}, nbr.options());
  auto keep = torch::empty({n, width}, torch::TensorOptions().dtype(torch::kByte).device(dev));
  auto* wp = wrapped.data_ptr<double>();
  auto* ip = img.data_ptr<int64_t>();
  auto* np = nbr.data_ptr<int64_t>();
  auto* oxp = ox.data_ptr<int64_t>();
  auto* oyp = oy.data_ptr<int64_t>();
  auto* ozp = oz.data_ptr<int64_t>();
  auto* kp = keep.data_ptr<uint8_t>();
  const double c00 = cell[0][0], c01 = cell[0][1], c02 = cell[0][2];
  const double c10 = cell[1][0], c11 = cell[1][1], c12 = cell[1][2];
  const double c20 = cell[2][0], c21 = cell[2][1], c22 = cell[2][2];
  q.parallel_for<class N1bPairs>(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> id) {
      const int64_t i = static_cast<int64_t>(id[0]);
      const double xi = wp[i * 3], yi = wp[i * 3 + 1], zi = wp[i * 3 + 2];
      for (int64_t j = 0; j < n; ++j) {
        for (int64_t im = 0; im < nimg; ++im) {
          const int64_t oxi = ip[im * 3], oyi = ip[im * 3 + 1], ozi = ip[im * 3 + 2];
          const double sx = oxi * c00 + oyi * c10 + ozi * c20;
          const double sy = oxi * c01 + oyi * c11 + ozi * c21;
          const double sz = oxi * c02 + oyi * c12 + ozi * c22;
          const double dx = wp[j * 3] + sx - xi;
          const double dy = wp[j * 3 + 1] + sy - yi;
          const double dz = wp[j * 3 + 2] + sz - zi;
          const double dist = sycl::sqrt(dx * dx + dy * dy + dz * dz);
          const int64_t col = j * nimg + im;
          const int64_t row = i * (n * nimg) + col;
          np[row] = j;
          oxp[row] = oxi;
          oyp[row] = oyi;
          ozp[row] = ozi;
          kp[row] = (dist >= tol && dist <= cutoff) ? 1 : 0;
        }
      }
    }).wait();

  N1bCandidates out;
  out.wrapped = wrapped;
  out.neighbor = nbr;
  out.ox = ox;
  out.oy = oy;
  out.oz = oz;
  out.keep = keep.to(torch::kBool);
  std::cerr << "N1B_PAIRS_DONE\n" << std::flush;
  return out;
}

}  // namespace uma
