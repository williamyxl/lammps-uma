// Step 24 (N5.5). In-model fused gather + Wigner rotation, work-group 128.
// Linked as a shared library so the device image survives oneCCL.
// Schema is uma_n5::f4. The GEMMs stay in the traced graph.

#include "uma/n5_ops.h"

#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>
#include <sycl/sycl.hpp>

namespace {

constexpr int kWg = 128;

struct Dim {
  int64_t E;
  int64_t L;
  int64_t M;
  int64_t C;
};

Dim check_dims(const at::Tensor& x, const at::Tensor& edge, const at::Tensor& wigner) {
  TORCH_CHECK(x.dtype() == at::kDouble && wigner.dtype() == at::kDouble, "f4: FP64 required");
  TORCH_CHECK(x.device().is_xpu() && edge.device().is_xpu() && wigner.device().is_xpu(), "f4: XPU required");
  TORCH_CHECK(edge.dtype() == at::kLong && x.dim() == 3 && edge.dim() == 2 && wigner.dim() == 3, "f4: shapes");
  Dim d;
  d.L = x.size(1);
  d.C = x.size(2);
  d.E = edge.size(1);
  d.M = wigner.size(1);
  TORCH_CHECK(edge.size(0) == 2 && wigner.size(0) == d.E && wigner.size(2) == d.L, "f4: edge or wigner shape");
  TORCH_CHECK(d.L <= 16 && d.M <= 16 && d.C > 0, "f4: L and M must be <= 16");
  return d;
}

using Atomic = sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                sycl::access::address_space::global_space>;

void permute_forward(const at::Tensor& x, const at::Tensor& edge, const at::Tensor& wigner, at::Tensor& y) {
  const Dim d = check_dims(x, edge, wigner);
  auto xc = x.contiguous();
  auto ec = edge.contiguous();
  auto wc = wigner.contiguous();
  const double* xp = xc.data_ptr<double>();
  const int64_t* ep = ec.data_ptr<int64_t>();
  const double* wp = wc.data_ptr<double>();
  double* yp = y.data_ptr<double>();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * kWg)), sycl::range<1>(kWg)),
                 [=](sycl::nd_item<1> it) {
                   const int64_t e = static_cast<int64_t>(it.get_group(0));
                   const int tid = static_cast<int>(it.get_local_id(0));
                   const int64_t src = ep[e];
                   const int64_t dst = ep[d.E + e];
                   const double* xs = xp + (src * d.L) * d.C;
                   const double* xt = xp + (dst * d.L) * d.C;
                   const double* we = wp + (e * d.M) * d.L;
                   double* ye = yp + (e * d.M) * (2 * d.C);
                   for (int64_t m = 0; m < d.M; ++m) {
                     const double* wm = we + m * d.L;
                     for (int64_t c = tid; c < d.C; c += kWg) {
                       double acc_s = 0.0;
                       double acc_t = 0.0;
                       for (int64_t l = 0; l < d.L; ++l) {
                         const double wv = wm[l];
                         acc_s += wv * xs[l * d.C + c];
                         acc_t += wv * xt[l * d.C + c];
                       }
                       ye[m * (2 * d.C) + c] = acc_s;
                       ye[m * (2 * d.C) + d.C + c] = acc_t;
                     }
                   }
                 })
      .wait();
}

void permute_backward(const at::Tensor& x, const at::Tensor& edge, const at::Tensor& wigner,
                      const at::Tensor& gy, at::Tensor& gx, at::Tensor& gw) {
  const Dim d = check_dims(x, edge, wigner);
  auto xc = x.contiguous();
  auto ec = edge.contiguous();
  auto wc = wigner.contiguous();
  auto gyc = gy.contiguous();
  gx.zero_();
  const double* xp = xc.data_ptr<double>();
  const int64_t* ep = ec.data_ptr<int64_t>();
  const double* wp = wc.data_ptr<double>();
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
                         const int64_t src = ep[e];
                         const int64_t dst = ep[d.E + e];
                         const double* xs = xp + (src * d.L) * d.C;
                         const double* xt = xp + (dst * d.L) * d.C;
                         const double* gye = gyp + (e * d.M) * (2 * d.C);
                         const double* we = wp + (e * d.M) * d.L;
                         double* gwe = gwp + (e * d.M) * d.L;
                         double* red = &slm[0];
                         for (int64_t l = 0; l < d.L; ++l) {
                           for (int64_t c = tid; c < d.C; c += kWg) {
                             double gs = 0.0;
                             double gt = 0.0;
                             for (int64_t m = 0; m < d.M; ++m) {
                               const double wv = we[m * d.L + l];
                               gs += wv * gye[m * (2 * d.C) + c];
                               gt += wv * gye[m * (2 * d.C) + d.C + c];
                             }
                             Atomic(gxp[((src * d.L + l) * d.C) + c]).fetch_add(gs);
                             Atomic(gxp[((dst * d.L + l) * d.C) + c]).fetch_add(gt);
                           }
                         }
                         for (int64_t m = 0; m < d.M; ++m) {
                           for (int64_t l = 0; l < d.L; ++l) {
                             double partial = 0.0;
                             for (int64_t c = tid; c < d.C; c += kWg) {
                               partial += gye[m * (2 * d.C) + c] * xs[l * d.C + c];
                               partial += gye[m * (2 * d.C) + d.C + c] * xt[l * d.C + c];
                             }
                             red[tid] = partial;
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int offset = kWg / 2; offset > 0; offset >>= 1) {
                               if (tid < offset) red[tid] += red[tid + offset];
                               it.barrier(sycl::access::fence_space::local_space);
                             }
                             if (tid == 0) gwe[m * d.L + l] = red[0];
                             it.barrier(sycl::access::fence_space::local_space);
                           }
                         }
                       });
      })
      .wait();
}

class F4Autograd : public torch::autograd::Function<F4Autograd> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x, torch::Tensor edge,
                               torch::Tensor wigner) {
    ctx->save_for_backward({x, edge, wigner});
    auto y = torch::empty({edge.size(1), wigner.size(1), x.size(2) * 2}, x.options());
    permute_forward(x, edge, wigner, y);
    return y;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto gx = torch::empty_like(saved[0]);
    auto gw = torch::empty_like(saved[2]);
    permute_backward(saved[0], saved[1], saved[2], grad_outputs[0], gx, gw);
    return {gx, torch::Tensor(), gw};
  }
};

torch::Tensor f4_autograd(const torch::Tensor& x, const torch::Tensor& edge, const torch::Tensor& wigner) {
  return F4Autograd::apply(x, edge, wigner);
}

torch::Tensor f4_xpu(const torch::Tensor& x, const torch::Tensor& edge, const torch::Tensor& wigner) {
  auto y = torch::empty({edge.size(1), wigner.size(1), x.size(2) * 2}, x.options());
  permute_forward(x, edge, wigner, y);
  return y;
}

}  // namespace

extern "C" void uma_f4_permute_anchor() {}

TORCH_LIBRARY_IMPL(uma_n5, XPU, m) { m.impl("f4", &f4_xpu); }

TORCH_LIBRARY_IMPL(uma_n5, Autograd, m) { m.impl("f4", &f4_autograd); }
