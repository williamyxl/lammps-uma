// Step 25 (N5.6). In-model fused inverse rotation and scatter, work-group 64.
// Linked as a shared library so the device image survives oneCCL.

#include "uma/n5_ops.h"

#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>
#include <sycl/sycl.hpp>

namespace {

constexpr int kWg = 64;

struct Dim {
  int64_t E;
  int64_t L;
  int64_t M;
  int64_t C;
  int64_t offset;
};

Dim check_dims(const at::Tensor& x, const at::Tensor& w, const at::Tensor& edge, int64_t offset) {
  TORCH_CHECK(x.dtype() == at::kDouble && w.dtype() == at::kDouble, "f5: FP64 required");
  TORCH_CHECK(x.device().is_xpu() && w.device().is_xpu() && edge.device().is_xpu(), "f5: XPU required");
  TORCH_CHECK(edge.dtype() == at::kLong && x.dim() == 3 && w.dim() == 3 && edge.dim() == 2, "f5: shapes");
  Dim d;
  d.E = x.size(0);
  d.M = x.size(1);
  d.C = x.size(2);
  d.L = w.size(1);
  d.offset = offset;
  TORCH_CHECK(w.size(0) == d.E && w.size(2) == d.M && edge.size(0) == 2 && edge.size(1) == d.E,
              "f5: wigner or edge shape");
  TORCH_CHECK(d.L <= 16 && d.M <= 16 && d.C > 0, "f5: L and M must be <= 16");
  return d;
}

using Atomic = sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                sycl::access::address_space::global_space>;

void scatter_forward(const at::Tensor& x, const at::Tensor& w, const at::Tensor& edge, at::Tensor& y,
                     int64_t offset) {
  const Dim d = check_dims(x, w, edge, offset);
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto ec = edge.contiguous();
  y.zero_();
  const double* xp = xc.data_ptr<double>();
  const double* wp = wc.data_ptr<double>();
  const int64_t* ep = ec.data_ptr<int64_t>();
  double* yp = y.data_ptr<double>();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * kWg)), sycl::range<1>(kWg)),
                 [=](sycl::nd_item<1> it) {
                   const int64_t e = static_cast<int64_t>(it.get_group(0));
                   const int tid = static_cast<int>(it.get_local_id(0));
                   const int64_t center = ep[d.E + e] - d.offset;
                   const double* xe = xp + (e * d.M) * d.C;
                   const double* we = wp + (e * d.L) * d.M;
                   double* yn = yp + (center * d.L) * d.C;
                   for (int64_t l = 0; l < d.L; ++l) {
                     const double* wl = we + l * d.M;
                     for (int64_t c = tid; c < d.C; c += kWg) {
                       double acc = 0.0;
                       for (int64_t m = 0; m < d.M; ++m) acc += wl[m] * xe[m * d.C + c];
                       Atomic(yn[l * d.C + c]).fetch_add(acc);
                     }
                   }
                 })
      .wait();
}

void scatter_backward(const at::Tensor& x, const at::Tensor& w, const at::Tensor& edge, const at::Tensor& gy,
                      at::Tensor& gx, at::Tensor& gw, int64_t offset) {
  const Dim d = check_dims(x, w, edge, offset);
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto ec = edge.contiguous();
  auto gyc = gy.contiguous();
  const double* xp = xc.data_ptr<double>();
  const double* wp = wc.data_ptr<double>();
  const int64_t* ep = ec.data_ptr<int64_t>();
  const double* gyp = gyc.data_ptr<double>();
  double* gxp = gx.data_ptr<double>();
  double* gwp = gw.data_ptr<double>();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(static_cast<size_t>(kWg)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * kWg)), sycl::range<1>(kWg)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         const int tid = static_cast<int>(it.get_local_id(0));
                         const int64_t center = ep[d.E + e] - d.offset;
                         const double* xe = xp + (e * d.M) * d.C;
                         const double* we = wp + (e * d.L) * d.M;
                         const double* yn = gyp + (center * d.L) * d.C;
                         double* gxe = gxp + (e * d.M) * d.C;
                         double* gwe = gwp + (e * d.L) * d.M;
                         double* red = &slm[0];
                         for (int64_t m = 0; m < d.M; ++m) {
                           for (int64_t c = tid; c < d.C; c += kWg) {
                             double acc = 0.0;
                             for (int64_t l = 0; l < d.L; ++l) acc += we[l * d.M + m] * yn[l * d.C + c];
                             gxe[m * d.C + c] = acc;
                           }
                         }
                         for (int64_t l = 0; l < d.L; ++l) {
                           for (int64_t m = 0; m < d.M; ++m) {
                             double partial = 0.0;
                             for (int64_t c = tid; c < d.C; c += kWg) {
                               partial += yn[l * d.C + c] * xe[m * d.C + c];
                             }
                             red[tid] = partial;
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int off = kWg / 2; off > 0; off >>= 1) {
                               if (tid < off) red[tid] += red[tid + off];
                               it.barrier(sycl::access::fence_space::local_space);
                             }
                             if (tid == 0) gwe[l * d.M + m] = red[0];
                             it.barrier(sycl::access::fence_space::local_space);
                           }
                         }
                       });
      })
      .wait();
}

class F5Autograd : public torch::autograd::Function<F5Autograd> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x, torch::Tensor w,
                               torch::Tensor edge, int64_t n_nodes, int64_t offset) {
    ctx->save_for_backward({x, w, edge});
    ctx->saved_data["off"] = offset;
    auto y = torch::empty({n_nodes, w.size(1), x.size(2)}, x.options());
    scatter_forward(x, w, edge, y, offset);
    return y;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    const int64_t offset = ctx->saved_data["off"].toInt();
    auto gx = torch::empty_like(saved[0]);
    auto gw = torch::empty_like(saved[1]);
    scatter_backward(saved[0], saved[1], saved[2], grad_outputs[0], gx, gw, offset);
    return {gx, gw, torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor f5_autograd(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& edge,
                          int64_t n_nodes, int64_t offset) {
  return F5Autograd::apply(x, w, edge, n_nodes, offset);
}

torch::Tensor f5_xpu(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& edge,
                     int64_t n_nodes, int64_t offset) {
  auto y = torch::empty({n_nodes, w.size(1), x.size(2)}, x.options());
  scatter_forward(x, w, edge, y, offset);
  return y;
}

}  // namespace

extern "C" void uma_f5_scatter_anchor() {}

TORCH_LIBRARY_IMPL(uma_n5, XPU, m) { m.impl("f5", &f5_xpu); }

TORCH_LIBRARY_IMPL(uma_n5, Autograd, m) { m.impl("f5", &f5_autograd); }
