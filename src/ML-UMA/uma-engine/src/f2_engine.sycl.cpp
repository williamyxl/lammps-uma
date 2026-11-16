// Step 23 (N5.4). In-model fused gate, work-group 128. Linked as a shared
// library so the device image survives oneCCL. Schema is uma_n5::f2.

#include "uma/n5_ops.h"

#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>
#include <sycl/sycl.hpp>

namespace {

constexpr int kWg = 128;

double sigmoid_stable(double z) {
  if (z >= 0.0) {
    const double e = sycl::exp(-z);
    return 1.0 / (1.0 + e);
  }
  const double e = sycl::exp(z);
  return e / (1.0 + e);
}

struct Dim {
  int64_t E;
  int64_t ncomp;
  int64_t C;
  int64_t lmax;
  int64_t nvec;
};

Dim check_dims(const at::Tensor& g, const at::Tensor& x, const at::Tensor& expand) {
  TORCH_CHECK(g.dtype() == at::kDouble && x.dtype() == at::kDouble, "f2: FP64 required");
  TORCH_CHECK(g.device().is_xpu() && x.device().is_xpu() && expand.device().is_xpu(), "f2: XPU required");
  TORCH_CHECK(expand.dtype() == at::kLong && g.dim() == 2 && x.dim() == 3, "f2: shapes");
  Dim d;
  d.E = x.size(0);
  d.ncomp = x.size(1);
  d.C = x.size(2);
  d.nvec = expand.size(0);
  TORCH_CHECK(d.ncomp == d.nvec + 1 && g.size(0) == d.E && g.size(1) % d.C == 0, "f2: shape mismatch");
  d.lmax = g.size(1) / d.C;
  TORCH_CHECK(d.lmax >= 1 && d.lmax <= 8 && d.nvec <= 64, "f2: lmax or expand out of range");
  return d;
}

void fwd_channel(const double* g, const double* x, double* y, const int64_t* expand, const Dim& d,
                 int64_t c) {
  double sig[8];
  for (int64_t l = 0; l < d.lmax; ++l) sig[l] = sigmoid_stable(g[l * d.C + c]);
  const double xs = x[c];
  y[c] = xs * sigmoid_stable(xs);
  for (int64_t k = 0; k < d.nvec; ++k) {
    y[(k + 1) * d.C + c] = x[(k + 1) * d.C + c] * sig[expand[k]];
  }
}

void bwd_channel(const double* g, const double* x, const double* gy, double* gx, double* gg,
                 const int64_t* expand, const Dim& d, int64_t c) {
  double sig[8];
  double ds[8];
  for (int64_t l = 0; l < d.lmax; ++l) {
    const double s = sigmoid_stable(g[l * d.C + c]);
    sig[l] = s;
    ds[l] = s * (1.0 - s);
  }
  const double xs = x[c];
  const double s0 = sigmoid_stable(xs);
  gx[c] = gy[c] * (s0 * (1.0 + xs * (1.0 - s0)));
  double acc[8] = {};
  for (int64_t k = 0; k < d.nvec; ++k) {
    const int64_t l = expand[k];
    const double xv = x[(k + 1) * d.C + c];
    const double gyv = gy[(k + 1) * d.C + c];
    gx[(k + 1) * d.C + c] = gyv * sig[l];
    acc[l] += gyv * xv * ds[l];
  }
  for (int64_t l = 0; l < d.lmax; ++l) gg[l * d.C + c] = acc[l];
}

void gate_forward(const at::Tensor& g, const at::Tensor& x, const at::Tensor& expand, at::Tensor& y) {
  const Dim d = check_dims(g, x, expand);
  auto gc = g.contiguous();
  auto xc = x.contiguous();
  auto ec = expand.contiguous();
  const double* gp = gc.data_ptr<double>();
  const double* xp = xc.data_ptr<double>();
  double* yp = y.data_ptr<double>();
  const int64_t* ep = ec.data_ptr<int64_t>();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * kWg)), sycl::range<1>(kWg)),
                 [=](sycl::nd_item<1> it) {
                   const int64_t e = static_cast<int64_t>(it.get_group(0));
                   const int tid = static_cast<int>(it.get_local_id(0));
                   const double* ge = gp + e * d.lmax * d.C;
                   const double* xe = xp + e * d.ncomp * d.C;
                   double* ye = yp + e * d.ncomp * d.C;
                   for (int64_t c = tid; c < d.C; c += kWg) fwd_channel(ge, xe, ye, ep, d, c);
                 })
      .wait();
}

void gate_backward(const at::Tensor& g, const at::Tensor& x, const at::Tensor& gy, at::Tensor& gx,
                   at::Tensor& gg, const at::Tensor& expand) {
  const Dim d = check_dims(g, x, expand);
  auto gc = g.contiguous();
  auto xc = x.contiguous();
  auto gyc = gy.contiguous();
  auto ec = expand.contiguous();
  const double* gp = gc.data_ptr<double>();
  const double* xp = xc.data_ptr<double>();
  const double* gyp = gyc.data_ptr<double>();
  double* gxp = gx.data_ptr<double>();
  double* ggp = gg.data_ptr<double>();
  const int64_t* ep = ec.data_ptr<int64_t>();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * kWg)), sycl::range<1>(kWg)),
                 [=](sycl::nd_item<1> it) {
                   const int64_t e = static_cast<int64_t>(it.get_group(0));
                   const int tid = static_cast<int>(it.get_local_id(0));
                   const int64_t go = e * d.lmax * d.C;
                   const int64_t xo = e * d.ncomp * d.C;
                   for (int64_t c = tid; c < d.C; c += kWg) {
                     bwd_channel(gp + go, xp + xo, gyp + xo, gxp + xo, ggp + go, ep, d, c);
                   }
                 })
      .wait();
}

class F2Autograd : public torch::autograd::Function<F2Autograd> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor g, torch::Tensor x,
                               torch::Tensor expand) {
    ctx->save_for_backward({g, x, expand});
    auto y = torch::empty_like(x);
    gate_forward(g, x, expand, y);
    return y;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto gx = torch::empty_like(saved[1]);
    auto gg = torch::empty_like(saved[0]);
    gate_backward(saved[0], saved[1], grad_outputs[0], gx, gg, saved[2]);
    return {gg, gx, torch::Tensor()};
  }
};

torch::Tensor f2_autograd(const torch::Tensor& g, const torch::Tensor& x, const torch::Tensor& expand) {
  return F2Autograd::apply(g, x, expand);
}

torch::Tensor f2_xpu(const torch::Tensor& g, const torch::Tensor& x, const torch::Tensor& expand) {
  auto y = torch::empty_like(x);
  gate_forward(g, x, expand, y);
  return y;
}

}  // namespace

extern "C" void uma_f2_gate_anchor() {}

TORCH_LIBRARY_IMPL(uma_n5, XPU, m) { m.impl("f2", &f2_xpu); }

TORCH_LIBRARY_IMPL(uma_n5, Autograd, m) { m.impl("f2", &f2_autograd); }
