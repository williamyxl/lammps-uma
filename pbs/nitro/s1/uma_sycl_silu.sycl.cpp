// Step 12 (S1). Prototype compute op, not part of the UMA model.
// y = silu(x * w + b), elementwise, FP64, on the current XPU stream.
// Backward is a SYCL kernel. Autograd is registered in C++.

#include <torch/torch.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

namespace {

void check_fp64_xpu(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.dtype() == at::kDouble, name, ": FP64 required");
  TORCH_CHECK(t.device().is_xpu(), name, ": XPU required");
  TORCH_CHECK(t.numel() > 0, name, ": empty tensor");
}

at::Tensor silu_affine_xpu(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b) {
  check_fp64_xpu(x, "x");
  check_fp64_xpu(w, "w");
  check_fp64_xpu(b, "b");
  TORCH_CHECK(x.sizes() == w.sizes() && x.sizes() == b.sizes(), "silu_affine: size mismatch");
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto bc = b.contiguous();
  auto y = torch::empty(xc.sizes(), xc.options());
  const auto n = static_cast<size_t>(xc.numel());
  auto* xp = xc.data_ptr<double>();
  auto* wp = wc.data_ptr<double>();
  auto* bp = bc.data_ptr<double>();
  auto* yp = y.data_ptr<double>();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  constexpr size_t kWg = 256;
  const size_t global = ((n + kWg - 1) / kWg) * kWg;
  q.parallel_for<class SiluAffineFwd>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kWg)),
      [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        if (i >= n) return;
        const double z = xp[i] * wp[i] + bp[i];
        double s;
        if (z >= 0.0) {
          const double e = sycl::exp(-z);
          s = 1.0 / (1.0 + e);
          yp[i] = z * s;
        } else {
          const double e = sycl::exp(z);
          s = e / (1.0 + e);
          yp[i] = z * s;
        }
      });
  return y;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> silu_affine_backward_xpu(
    const at::Tensor& gy, const at::Tensor& x, const at::Tensor& w, const at::Tensor& b) {
  check_fp64_xpu(gy, "gy");
  check_fp64_xpu(x, "x");
  check_fp64_xpu(w, "w");
  check_fp64_xpu(b, "b");
  TORCH_CHECK(gy.sizes() == x.sizes() && x.sizes() == w.sizes() && x.sizes() == b.sizes(),
              "silu_affine_backward: size mismatch");
  auto gyc = gy.contiguous();
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto bc = b.contiguous();
  auto gx = torch::empty(xc.sizes(), xc.options());
  auto gw = torch::empty(xc.sizes(), xc.options());
  auto gb = torch::empty(xc.sizes(), xc.options());
  const auto n = static_cast<size_t>(xc.numel());
  auto* gyp = gyc.data_ptr<double>();
  auto* xp = xc.data_ptr<double>();
  auto* wp = wc.data_ptr<double>();
  auto* bp = bc.data_ptr<double>();
  auto* gxp = gx.data_ptr<double>();
  auto* gwp = gw.data_ptr<double>();
  auto* gbp = gb.data_ptr<double>();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  constexpr size_t kWg = 256;
  const size_t global = ((n + kWg - 1) / kWg) * kWg;
  q.parallel_for<class SiluAffineBwd>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kWg)),
      [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        if (i >= n) return;
        const double z = xp[i] * wp[i] + bp[i];
        double sig;
        if (z >= 0.0) {
          const double e = sycl::exp(-z);
          sig = 1.0 / (1.0 + e);
        } else {
          const double e = sycl::exp(z);
          sig = e / (1.0 + e);
        }
        // d(silu)/dz = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
        const double dsilu = sig * (1.0 + z * (1.0 - sig));
        const double g = gyp[i] * dsilu;
        gxp[i] = g * wp[i];
        gwp[i] = g * xp[i];
        gbp[i] = g;
      });
  return {gx, gw, gb};
}

class SiluAffineAutograd : public torch::autograd::Function<SiluAffineAutograd> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x,
                               torch::Tensor w, torch::Tensor b) {
    ctx->save_for_backward({x, w, b});
    return silu_affine_xpu(x, w, b);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto grads = silu_affine_backward_xpu(grad_outputs[0], saved[0], saved[1], saved[2]);
    return {std::get<0>(grads), std::get<1>(grads), std::get<2>(grads)};
  }
};

torch::Tensor silu_affine_autograd(const torch::Tensor& x, const torch::Tensor& w,
                                   const torch::Tensor& b) {
  return SiluAffineAutograd::apply(x, w, b);
}

}  // namespace

TORCH_LIBRARY(uma_sycl, m) {
  m.def("silu_affine(Tensor x, Tensor w, Tensor b) -> Tensor");
  m.def("silu_affine_backward(Tensor gy, Tensor x, Tensor w, Tensor b) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(uma_sycl, XPU, m) {
  m.impl("silu_affine", &silu_affine_xpu);
  m.impl("silu_affine_backward", &silu_affine_backward_xpu);
}

TORCH_LIBRARY_IMPL(uma_sycl, Autograd, m) {
  m.impl("silu_affine", &silu_affine_autograd);
}
