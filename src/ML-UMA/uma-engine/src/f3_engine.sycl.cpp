// Step 27 (N5.8). In-model fused radial MLP chain, work-group 32 LN+SiLU.
// GEMMs stay on vendor linear. Linked as a shared library so the device
// image survives oneCCL. Schema is uma_n5::f3. Weights do not require grad.

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

struct Chain {
  int64_t N;
  int64_t E;
  int64_t H;
  int64_t in;
  int64_t out;
  double eps;
};

Chain chain_dims(const at::Tensor& x, const at::Tensor& W1, const at::Tensor& b1, const at::Tensor& ln1_w,
                 const at::Tensor& ln1_b, const at::Tensor& W2, const at::Tensor& b2,
                 const at::Tensor& ln2_w, const at::Tensor& ln2_b, const at::Tensor& W3,
                 const at::Tensor& b3, double eps) {
  TORCH_CHECK(x.dtype() == at::kDouble && W1.dtype() == at::kDouble, "f3: FP64 required");
  TORCH_CHECK(x.device().is_xpu() && W1.device().is_xpu(), "f3: XPU required");
  TORCH_CHECK(x.dim() == 2 && W1.dim() == 2 && ln1_w.dim() == 2 && W2.dim() == 3 && W3.dim() == 3,
              "f3: shapes");
  Chain d;
  d.E = x.size(0);
  d.in = x.size(1);
  d.N = ln1_w.size(0);
  d.H = ln1_w.size(1);
  d.out = W3.size(1);
  d.eps = eps;
  TORCH_CHECK(d.N >= 1 && d.N <= 8 && d.H > 0 && d.H <= 512, "f3: N or H out of range");
  TORCH_CHECK(W1.size(0) == d.N * d.H && W1.size(1) == d.in && b1.numel() == d.N * d.H, "f3: W1");
  TORCH_CHECK(ln1_b.sizes() == ln1_w.sizes() && ln2_w.sizes() == ln1_w.sizes() &&
                  ln2_b.sizes() == ln1_w.sizes(),
              "f3: LN weights");
  TORCH_CHECK(W2.size(0) == d.N && W2.size(1) == d.H && W2.size(2) == d.H && b2.size(0) == d.N &&
                  b2.size(1) == d.H,
              "f3: W2");
  TORCH_CHECK(W3.size(0) == d.N && W3.size(2) == d.H && b3.size(0) == d.N && b3.size(1) == d.out,
              "f3: W3");
  TORCH_CHECK(d.eps > 0.0, "f3: eps must be positive");
  return d;
}

at::Tensor chain_forward(const at::Tensor& x, const at::Tensor& W1, const at::Tensor& b1,
                         const at::Tensor& ln1_w, const at::Tensor& ln1_b, const at::Tensor& W2,
                         const at::Tensor& b2, const at::Tensor& ln2_w, const at::Tensor& ln2_b,
                         const at::Tensor& W3, const at::Tensor& b3, double eps) {
  const Chain d = chain_dims(x, W1, b1, ln1_w, ln1_b, W2, b2, ln2_w, ln2_b, W3, b3, eps);
  auto xc = x.contiguous();
  auto h_all = at::linear(xc, W1.contiguous(), b1.contiguous());
  auto y = at::empty({d.N, d.E, d.out}, xc.options());
  for (int64_t i = 0; i < d.N; ++i) {
    auto h0 = h_all.narrow(1, i * d.H, d.H).contiguous();
    auto s1 = at::empty_like(h0);
    ln_silu_forward(h0, ln1_w[i], ln1_b[i], eps, s1);
    auto h1 = at::linear(s1, W2[i], b2[i]);
    auto s2 = at::empty_like(h1);
    ln_silu_forward(h1, ln2_w[i], ln2_b[i], eps, s2);
    y[i] = at::linear(s2, W3[i], b3[i]);
  }
  return y;
}

at::Tensor chain_backward(const at::Tensor& x, const at::Tensor& W1, const at::Tensor& b1,
                          const at::Tensor& ln1_w, const at::Tensor& ln1_b, const at::Tensor& W2,
                          const at::Tensor& b2, const at::Tensor& ln2_w, const at::Tensor& ln2_b,
                          const at::Tensor& W3, const at::Tensor& b3, const at::Tensor& gy, double eps) {
  const Chain d = chain_dims(x, W1, b1, ln1_w, ln1_b, W2, b2, ln2_w, ln2_b, W3, b3, eps);
  auto xc = x.contiguous();
  auto h_all = at::linear(xc, W1.contiguous(), b1.contiguous());
  auto gh_all = at::zeros_like(h_all);
  auto gyc = gy.contiguous();
  for (int64_t i = 0; i < d.N; ++i) {
    auto h0 = h_all.narrow(1, i * d.H, d.H).contiguous();
    auto s1 = at::empty_like(h0);
    ln_silu_forward(h0, ln1_w[i], ln1_b[i], eps, s1);
    auto h1 = at::linear(s1, W2[i], b2[i]);
    auto gyi = gyc[i].contiguous();
    auto gs2 = at::matmul(gyi, W3[i]);
    auto gh1 = at::empty_like(h1);
    ln_silu_backward(h1, ln2_w[i], ln2_b[i], eps, gs2, gh1);
    auto gs1 = at::matmul(gh1, W2[i]);
    auto gh0 = at::empty_like(h0);
    ln_silu_backward(h0, ln1_w[i], ln1_b[i], eps, gs1, gh0);
    gh_all.narrow(1, i * d.H, d.H).copy_(gh0);
  }
  return at::matmul(gh_all, W1);
}

class F3Autograd : public torch::autograd::Function<F3Autograd> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x, torch::Tensor W1,
                               torch::Tensor b1, torch::Tensor ln1_w, torch::Tensor ln1_b,
                               torch::Tensor W2, torch::Tensor b2, torch::Tensor ln2_w,
                               torch::Tensor ln2_b, torch::Tensor W3, torch::Tensor b3, double eps) {
    ctx->save_for_backward({x, W1, b1, ln1_w, ln1_b, W2, b2, ln2_w, ln2_b, W3, b3});
    ctx->saved_data["eps"] = eps;
    return chain_forward(x, W1, b1, ln1_w, ln1_b, W2, b2, ln2_w, ln2_b, W3, b3, eps);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    auto s = ctx->get_saved_variables();
    const double eps = ctx->saved_data["eps"].toDouble();
    auto gx = chain_backward(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10],
                             grad_outputs[0], eps);
    torch::autograd::variable_list out(13);
    out[0] = gx;
    return out;
  }
};

torch::Tensor f3_autograd(const torch::Tensor& x, const torch::Tensor& W1, const torch::Tensor& b1,
                          const torch::Tensor& ln1_w, const torch::Tensor& ln1_b, const torch::Tensor& W2,
                          const torch::Tensor& b2, const torch::Tensor& ln2_w, const torch::Tensor& ln2_b,
                          const torch::Tensor& W3, const torch::Tensor& b3, double eps) {
  return F3Autograd::apply(x, W1, b1, ln1_w, ln1_b, W2, b2, ln2_w, ln2_b, W3, b3, eps);
}

torch::Tensor f3_xpu(const torch::Tensor& x, const torch::Tensor& W1, const torch::Tensor& b1,
                     const torch::Tensor& ln1_w, const torch::Tensor& ln1_b, const torch::Tensor& W2,
                     const torch::Tensor& b2, const torch::Tensor& ln2_w, const torch::Tensor& ln2_b,
                     const torch::Tensor& W3, const torch::Tensor& b3, double eps) {
  return chain_forward(x, W1, b1, ln1_w, ln1_b, W2, b2, ln2_w, ln2_b, W3, b3, eps);
}

}  // namespace

extern "C" void uma_f3_radial_anchor() {}

TORCH_LIBRARY_IMPL(uma_n5, XPU, m) { m.impl("f3", &f3_xpu); }

TORCH_LIBRARY_IMPL(uma_n5, Autograd, m) { m.impl("f3", &f3_autograd); }
