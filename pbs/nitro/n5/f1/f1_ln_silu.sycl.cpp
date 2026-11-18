// Step 26 (N5.7). Fused LayerNorm + SiLU on [E, C].
// y = silu(LN(x; weight, bias, eps)). Matches torch layer_norm then silu.
// Not wired into the model. The timed backward is the SYCL kernel.
// ln_silu_scaffold's backward is torch autograd on the stock math and is not timed.
// No GEMM in this block.

#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>
#include <sycl/sycl.hpp>

#include <tuple>

namespace {

double sigmoid_stable(double z) {
  if (z >= 0.0) {
    const double e = sycl::exp(-z);
    return 1.0 / (1.0 + e);
  }
  const double e = sycl::exp(z);
  return e / (1.0 + e);
}

template <int WG>
void reduce_pair(sycl::nd_item<1> it, double* a, double* b) {
  const int tid = static_cast<int>(it.get_local_id(0));
  it.barrier(sycl::access::fence_space::local_space);
  for (int off = WG / 2; off > 0; off >>= 1) {
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

template <int WG>
void fwd_edge(sycl::nd_item<1> it, const double* xe, const double* w, const double* b, double* ye,
              double* slm, int64_t C, double eps) {
  const int tid = static_cast<int>(it.get_local_id(0));
  double* sa = slm;
  double* sb = slm + WG;
  double sum = 0.0;
  for (int64_t c = tid; c < C; c += WG) sum += xe[c];
  sa[tid] = sum;
  sb[tid] = 0.0;
  reduce_pair<WG>(it, sa, sb);
  const double mean = sa[0] / static_cast<double>(C);
  double sumsq = 0.0;
  for (int64_t c = tid; c < C; c += WG) {
    const double d = xe[c] - mean;
    sumsq += d * d;
  }
  sa[tid] = sumsq;
  sb[tid] = 0.0;
  reduce_pair<WG>(it, sa, sb);
  const double var = sa[0] / static_cast<double>(C);
  const double inv_std = sycl::rsqrt(var + eps);
  for (int64_t c = tid; c < C; c += WG) {
    const double u = (xe[c] - mean) * inv_std * w[c] + b[c];
    ye[c] = u * sigmoid_stable(u);
  }
}

template <int WG>
void bwd_edge(sycl::nd_item<1> it, const double* xe, const double* w, const double* b, const double* gye,
              double* gxe, double* slm, int64_t C, double eps) {
  const int tid = static_cast<int>(it.get_local_id(0));
  double* sa = slm;
  double* sb = slm + WG;
  double sum = 0.0;
  for (int64_t c = tid; c < C; c += WG) sum += xe[c];
  sa[tid] = sum;
  sb[tid] = 0.0;
  reduce_pair<WG>(it, sa, sb);
  const double mean = sa[0] / static_cast<double>(C);
  double sumsq = 0.0;
  for (int64_t c = tid; c < C; c += WG) {
    const double d = xe[c] - mean;
    sumsq += d * d;
  }
  sa[tid] = sumsq;
  sb[tid] = 0.0;
  reduce_pair<WG>(it, sa, sb);
  const double var = sa[0] / static_cast<double>(C);
  const double inv_std = sycl::rsqrt(var + eps);
  const double inv_C = 1.0 / static_cast<double>(C);
  double s_dxh = 0.0;
  double s_dxh_xh = 0.0;
  for (int64_t c = tid; c < C; c += WG) {
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
  reduce_pair<WG>(it, sa, sb);
  const double mean_dxh = sa[0] * inv_C;
  const double mean_dxh_xh = sb[0] * inv_C;
  for (int64_t c = tid; c < C; c += WG) {
    const double xhat = (xe[c] - mean) * inv_std;
    const double u = xhat * w[c] + b[c];
    const double s = sigmoid_stable(u);
    const double dsilu = s * (1.0 + u * (1.0 - s));
    const double dxhat = gye[c] * dsilu * w[c];
    gxe[c] = inv_std * (dxhat - mean_dxh - xhat * mean_dxh_xh);
  }
}

template <int WG>
void launch_fwd(sycl::queue& q, const Dim& d, const double* x, const double* w, const double* b,
                double* y) {
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(static_cast<size_t>(2 * WG)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)),
                                         sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         fwd_edge<WG>(it, x + e * d.C, w, b, y + e * d.C, &slm[0], d.C, d.eps);
                       });
      })
      .wait();
}

template <int WG, int SG>
void launch_fwd_sg(sycl::queue& q, const Dim& d, const double* x, const double* w, const double* b,
                   double* y) {
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(static_cast<size_t>(2 * WG)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)),
                                         sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         fwd_edge<WG>(it, x + e * d.C, w, b, y + e * d.C, &slm[0], d.C, d.eps);
                       });
      })
      .wait();
}

template <int WG>
void launch_fwd_local(sycl::queue& q, const Dim& d, const double* x, const double* w, const double* b,
                      double* y) {
  const size_t slm_n = static_cast<size_t>(2 * WG + d.C);
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(slm_n), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)),
                                         sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         const int tid = static_cast<int>(it.get_local_id(0));
                         double* red = &slm[0];
                         double* lx = red + 2 * WG;
                         const double* xe = x + e * d.C;
                         for (int64_t c = tid; c < d.C; c += WG) lx[c] = xe[c];
                         it.barrier(sycl::access::fence_space::local_space);
                         fwd_edge<WG>(it, lx, w, b, y + e * d.C, red, d.C, d.eps);
                       });
      })
      .wait();
}

template <int WG>
void launch_bwd(sycl::queue& q, const Dim& d, const double* x, const double* w, const double* b,
                const double* gy, double* gx) {
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(static_cast<size_t>(2 * WG)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)),
                                         sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         bwd_edge<WG>(it, x + e * d.C, w, b, gy + e * d.C, gx + e * d.C, &slm[0], d.C,
                                      d.eps);
                       });
      })
      .wait();
}

template <int WG, int SG>
void launch_bwd_sg(sycl::queue& q, const Dim& d, const double* x, const double* w, const double* b,
                   const double* gy, double* gx) {
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(static_cast<size_t>(2 * WG)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)),
                                         sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         bwd_edge<WG>(it, x + e * d.C, w, b, gy + e * d.C, gx + e * d.C, &slm[0], d.C,
                                      d.eps);
                       });
      })
      .wait();
}

template <int WG>
void launch_bwd_local(sycl::queue& q, const Dim& d, const double* x, const double* w, const double* b,
                      const double* gy, double* gx) {
  const size_t slm_n = static_cast<size_t>(2 * WG + 2 * d.C);
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(slm_n), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)),
                                         sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         const int tid = static_cast<int>(it.get_local_id(0));
                         double* red = &slm[0];
                         double* lx = red + 2 * WG;
                         double* lgy = lx + d.C;
                         const int64_t off = e * d.C;
                         for (int64_t c = tid; c < d.C; c += WG) {
                           lx[c] = x[off + c];
                           lgy[c] = gy[off + c];
                         }
                         it.barrier(sycl::access::fence_space::local_space);
                         bwd_edge<WG>(it, lx, w, b, lgy, gx + off, red, d.C, d.eps);
                       });
      })
      .wait();
}

void ln_silu_forward(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, double eps,
                     at::Tensor& y, int64_t variant) {
  const Dim d = dims_of(x, w, b, eps);
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto bc = b.contiguous();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  const double* xp = xc.data_ptr<double>();
  const double* wp = wc.data_ptr<double>();
  const double* bp = bc.data_ptr<double>();
  double* yp = y.data_ptr<double>();
  switch (variant) {
    case 0: launch_fwd<128>(q, d, xp, wp, bp, yp); break;
    case 1: launch_fwd<64>(q, d, xp, wp, bp, yp); break;
    case 2: launch_fwd<32>(q, d, xp, wp, bp, yp); break;
    case 3: launch_fwd_sg<128, 16>(q, d, xp, wp, bp, yp); break;
    case 4: launch_fwd_sg<128, 32>(q, d, xp, wp, bp, yp); break;
    case 5: launch_fwd_local<128>(q, d, xp, wp, bp, yp); break;
    case 6: launch_fwd_local<64>(q, d, xp, wp, bp, yp); break;
    case 7: launch_fwd_local<32>(q, d, xp, wp, bp, yp); break;
    default: TORCH_CHECK(false, "f1: unknown variant ", variant);
  }
}

void ln_silu_backward(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, double eps,
                      const at::Tensor& gy, at::Tensor& gx, int64_t variant) {
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
  switch (variant) {
    case 0: launch_bwd<128>(q, d, xp, wp, bp, gyp, gxp); break;
    case 1: launch_bwd<64>(q, d, xp, wp, bp, gyp, gxp); break;
    case 2: launch_bwd<32>(q, d, xp, wp, bp, gyp, gxp); break;
    case 3: launch_bwd_sg<128, 16>(q, d, xp, wp, bp, gyp, gxp); break;
    case 4: launch_bwd_sg<128, 32>(q, d, xp, wp, bp, gyp, gxp); break;
    case 5: launch_bwd_local<128>(q, d, xp, wp, bp, gyp, gxp); break;
    case 6: launch_bwd_local<64>(q, d, xp, wp, bp, gyp, gxp); break;
    case 7: launch_bwd_local<32>(q, d, xp, wp, bp, gyp, gxp); break;
    default: TORCH_CHECK(false, "f1: unknown variant ", variant);
  }
}

at::Tensor aten_ln_silu(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, double eps) {
  const auto C = x.size(1);
  return at::silu(at::layer_norm(x, {C}, w, b, eps));
}

class LnSiluAutograd : public torch::autograd::Function<LnSiluAutograd> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x, torch::Tensor w,
                               torch::Tensor b, double eps, int64_t variant) {
    ctx->save_for_backward({x, w, b});
    ctx->saved_data["eps"] = eps;
    ctx->saved_data["variant"] = variant;
    auto y = torch::empty_like(x);
    ln_silu_forward(x, w, b, eps, y, variant);
    return y;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    const double eps = ctx->saved_data["eps"].toDouble();
    const int64_t variant = ctx->saved_data["variant"].toInt();
    auto gx = torch::empty_like(saved[0]);
    ln_silu_backward(saved[0], saved[1], saved[2], eps, grad_outputs[0], gx, variant);
    return {gx, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor ln_silu_autograd(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b,
                               double eps, int64_t variant) {
  return LnSiluAutograd::apply(x, w, b, eps, variant);
}

at::Tensor ln_silu_scaffold(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b,
                            double eps) {
  auto xv = x.detach().requires_grad_(true);
  auto y = aten_ln_silu(xv, w, b, eps);
  torch::autograd::variable_list outs;
  outs.push_back(y.sum());
  torch::autograd::variable_list ins;
  ins.push_back(xv);
  auto grads = torch::autograd::grad(outs, ins);
  return grads[0];
}

}  // namespace

extern "C" void uma_n5f1_anchor() {}

TORCH_LIBRARY(uma_n5f1, m) {
  m.def("ln_silu(Tensor x, Tensor weight, Tensor bias, float eps, int variant) -> Tensor");
  m.def("ln_silu_scaffold(Tensor x, Tensor weight, Tensor bias, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(uma_n5f1, XPU, m) {
  m.impl("ln_silu", [](const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, double eps,
                       int64_t variant) {
    auto y = torch::empty_like(x);
    ln_silu_forward(x, w, b, eps, y, variant);
    return y;
  });
}

TORCH_LIBRARY_IMPL(uma_n5f1, Autograd, m) {
  m.impl("ln_silu", &ln_silu_autograd);
}

TORCH_LIBRARY_IMPL(uma_n5f1, CompositeExplicitAutograd, m) {
  m.impl("ln_silu_scaffold", &ln_silu_scaffold);
}
