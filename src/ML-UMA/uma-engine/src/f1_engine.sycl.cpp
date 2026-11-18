// Step 26 (N5.7). In-model fused LayerNorm + SiLU, work-group 32.
// Linked as a shared library so the device image survives oneCCL.
// Schema is uma_n5::f1. Weights do not require grad.

#include "uma/n5_ops.h"

#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>
#include <sycl/sycl.hpp>

namespace {

constexpr int kWg = 32;

double sigmoid_stable(double z) {
  if (z >= 0.0) {
    const double e = sycl::exp(-z);
    return 1.0 / (1.0 + e);
  }
  const double e = sycl::exp(z);
  return e / (1.0 + e);
}

void reduce_pair(sycl::nd_item<1> it, double* a, double* b) {
  const int tid = static_cast<int>(it.get_local_id(0));
  it.barrier(sycl::access::fence_space::local_space);
  for (int off = kWg / 2; off > 0; off >>= 1) {
    if (tid < off) {
      a[tid] += a[tid + off];
      b[tid] += b[tid + off];
    }
    it.barrier(sycl::access::fence_space::local_space);
  }
}

struct Dim {
  int64_t E;
  int64_t C;
  double eps;
};

Dim dims_of(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, double eps) {
  TORCH_CHECK(x.dtype() == at::kDouble && w.dtype() == at::kDouble && b.dtype() == at::kDouble,
              "f1: FP64 required");
  TORCH_CHECK(x.device().is_xpu() && w.device().is_xpu() && b.device().is_xpu(), "f1: XPU required");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 1 && b.dim() == 1, "f1: x [E,C], weight/bias [C]");
  Dim d;
  d.E = x.size(0);
  d.C = x.size(1);
  d.eps = eps;
  TORCH_CHECK(w.size(0) == d.C && b.size(0) == d.C, "f1: weight/bias length");
  TORCH_CHECK(d.C > 0 && d.C <= 512 && d.E > 0, "f1: C must be 1..512");
  TORCH_CHECK(d.eps > 0.0, "f1: eps must be positive");
  return d;
}

void fwd_edge(sycl::nd_item<1> it, const double* xe, const double* w, const double* b, double* ye,
              double* slm, int64_t C, double eps) {
  const int tid = static_cast<int>(it.get_local_id(0));
  double* sa = slm;
  double* sb = slm + kWg;
  double sum = 0.0;
  for (int64_t c = tid; c < C; c += kWg) sum += xe[c];
  sa[tid] = sum;
  sb[tid] = 0.0;
  reduce_pair(it, sa, sb);
  const double mean = sa[0] / static_cast<double>(C);
  double sumsq = 0.0;
  for (int64_t c = tid; c < C; c += kWg) {
    const double d = xe[c] - mean;
    sumsq += d * d;
  }
  sa[tid] = sumsq;
  sb[tid] = 0.0;
  reduce_pair(it, sa, sb);
  const double var = sa[0] / static_cast<double>(C);
  const double inv_std = sycl::rsqrt(var + eps);
  for (int64_t c = tid; c < C; c += kWg) {
    const double u = (xe[c] - mean) * inv_std * w[c] + b[c];
    ye[c] = u * sigmoid_stable(u);
  }
}

void bwd_edge(sycl::nd_item<1> it, const double* xe, const double* w, const double* b, const double* gye,
              double* gxe, double* slm, int64_t C, double eps) {
  const int tid = static_cast<int>(it.get_local_id(0));
  double* sa = slm;
  double* sb = slm + kWg;
  double sum = 0.0;
  for (int64_t c = tid; c < C; c += kWg) sum += xe[c];
  sa[tid] = sum;
  sb[tid] = 0.0;
  reduce_pair(it, sa, sb);
  const double mean = sa[0] / static_cast<double>(C);
  double sumsq = 0.0;
  for (int64_t c = tid; c < C; c += kWg) {
    const double d = xe[c] - mean;
    sumsq += d * d;
  }
  sa[tid] = sumsq;
  sb[tid] = 0.0;
  reduce_pair(it, sa, sb);
  const double var = sa[0] / static_cast<double>(C);
  const double inv_std = sycl::rsqrt(var + eps);
  const double inv_C = 1.0 / static_cast<double>(C);
  double s_dxh = 0.0;
  double s_dxh_xh = 0.0;
  for (int64_t c = tid; c < C; c += kWg) {
    const double xhat = (xe[c] - mean) * inv_std;
    const double u = xhat * w[c] + b[c];
    const double s = sigmoid_stable(u);
    const double dsilu = s * (1.0 + u * (1.0 - s));
    const double dxhat = gye[c] * dsilu * w[c];
    s_dxh += dxhat;
    s_dxh_xh += dxhat * xhat;
  }
  sa[tid] = s_dxh;
  sb[tid] = s_dxh_xh;
  reduce_pair(it, sa, sb);
  const double mean_dxh = sa[0] * inv_C;
  const double mean_dxh_xh = sb[0] * inv_C;
  for (int64_t c = tid; c < C; c += kWg) {
    const double xhat = (xe[c] - mean) * inv_std;
    const double u = xhat * w[c] + b[c];
    const double s = sigmoid_stable(u);
    const double dsilu = s * (1.0 + u * (1.0 - s));
    const double dxhat = gye[c] * dsilu * w[c];
    gxe[c] = inv_std * (dxhat - mean_dxh - xhat * mean_dxh_xh);
  }
}

void ln_silu_forward(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, double eps,
                     at::Tensor& y) {
  const Dim d = dims_of(x, w, b, eps);
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto bc = b.contiguous();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  const double* xp = xc.data_ptr<double>();
  const double* wp = wc.data_ptr<double>();
  const double* bp = bc.data_ptr<double>();
  double* yp = y.data_ptr<double>();
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(static_cast<size_t>(2 * kWg)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * kWg)),
                                         sycl::range<1>(kWg)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         fwd_edge(it, xp + e * d.C, wp, bp, yp + e * d.C, &slm[0], d.C, d.eps);
                       });
      })
      .wait();
}

void ln_silu_backward(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, double eps,
                      const at::Tensor& gy, at::Tensor& gx) {
  const Dim d = dims_of(x, w, b, eps);
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto bc = b.contiguous();
  auto gyc = gy.contiguous();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  const double* xp = xc.data_ptr<double>();
  const double* wp = wc.data_ptr<double>();
  const double* bp = bc.data_ptr<double>();
  const double* gyp = gyc.data_ptr<double>();
  double* gxp = gx.data_ptr<double>();
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(static_cast<size_t>(2 * kWg)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * kWg)),
                                         sycl::range<1>(kWg)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         bwd_edge(it, xp + e * d.C, wp, bp, gyp + e * d.C, gxp + e * d.C, &slm[0], d.C,
                                  d.eps);
                       });
      })
      .wait();
}

class F1Autograd : public torch::autograd::Function<F1Autograd> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x, torch::Tensor w,
                               torch::Tensor b, double eps) {
    ctx->save_for_backward({x, w, b});
    ctx->saved_data["eps"] = eps;
    auto y = torch::empty_like(x);
    ln_silu_forward(x, w, b, eps, y);
    return y;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    const double eps = ctx->saved_data["eps"].toDouble();
    auto gx = torch::empty_like(saved[0]);
    ln_silu_backward(saved[0], saved[1], saved[2], eps, grad_outputs[0], gx);
    return {gx, torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor f1_autograd(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b,
                          double eps) {
  return F1Autograd::apply(x, w, b, eps);
}

torch::Tensor f1_xpu(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b, double eps) {
  auto y = torch::empty_like(x);
  ln_silu_forward(x, w, b, eps, y);
  return y;
}

}  // namespace

extern "C" void uma_f1_ln_silu_anchor() {}

TORCH_LIBRARY_IMPL(uma_n5, XPU, m) { m.impl("f1", &f1_xpu); }

TORCH_LIBRARY_IMPL(uma_n5, Autograd, m) { m.impl("f1", &f1_autograd); }
