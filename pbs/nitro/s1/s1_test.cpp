// Step 12 (S1). No Python. Calls the SYCL op and checks forward, backward,
// a central finite difference, bitwise repeats, and a no-grad then recompute.

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>

namespace {

void sync() { torch::xpu::synchronize(0); }

at::Tensor silu_affine(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b) {
  std::cerr << "S1_FIND\n" << std::flush;
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("uma_sycl::silu_affine", "")
                       .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
  std::cerr << "S1_CALL_OP\n" << std::flush;
  return op.call(x, w, b);
}

double rel_err(const torch::Tensor& got, const torch::Tensor& ref) {
  auto d = (got - ref).abs();
  const double scale = std::max(1.0, ref.abs().max().item<double>());
  return d.max().item<double>() / scale;
}

torch::Tensor torch_silu_affine(const torch::Tensor& x, const torch::Tensor& w,
                                const torch::Tensor& b) {
  auto z = x * w + b;
  return z * torch::sigmoid(z);
}

int g_fails = 0;

void expect(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL " << msg << "\n";
    ++g_fails;
  } else {
    std::cerr << "OK   " << msg << "\n";
  }
}

}  // namespace

int main() {
  std::cerr << "S1_ENTER\n" << std::flush;
  const bool avail = torch::xpu::is_available();
  std::cerr << "S1_AVAIL " << avail << "\n" << std::flush;
  const auto ndev = torch::xpu::device_count();
  std::cerr << "S1_COUNT " << ndev << "\n" << std::flush;
  if (!avail || ndev < 1) {
    std::cerr << "S1_FAIL no XPU\n";
    return 1;
  }
  const torch::Device dev(torch::kXPU, 0);
  std::cerr << "S1_ONES\n" << std::flush;
  auto ones = torch::ones({4}, torch::TensorOptions().dtype(torch::kFloat64).device(dev));
  sync();
  std::cerr << "S1_ONES_OK " << ones[0].item<double>() << "\n" << std::flush;
  constexpr int64_t n = 65536 + 3;  // not a multiple of the work-group size
  auto idx = torch::arange(n, torch::TensorOptions().dtype(torch::kFloat64).device(dev));
  auto x = torch::sin(idx * 0.01);
  auto w = torch::cos(idx * 0.02) * 0.5 + 0.1;
  auto b = torch::sin(idx * 0.007) * 2.0;
  x.index_put_({0}, 40.0);
  w.index_put_({0}, 1.0);
  b.index_put_({0}, 0.0);
  x.index_put_({1}, -40.0);
  w.index_put_({1}, 1.0);
  b.index_put_({1}, 0.0);

  // Non-contiguous inputs. The kernel must compact them.
  auto x_nc = x.slice(0, 0, n, 2);
  auto w_nc = w.slice(0, 0, n, 2);
  auto b_nc = b.slice(0, 0, n, 2);
  TORCH_CHECK(!x_nc.is_contiguous(), "slice should be non-contiguous");

  std::cerr << "S1_TOUCH\n" << std::flush;
  {
    static auto touch = c10::Dispatcher::singleton()
                            .findSchemaOrThrow("uma_sycl::touch", "")
                            .typed<at::Tensor(const at::Tensor&)>();
    auto yt = touch.call(x_nc);
    sync();
    std::cerr << "S1_TOUCH_OK " << (yt - x_nc).abs().max().item<double>() << "\n" << std::flush;
  }
  std::cerr << "S1_OP\n" << std::flush;
  auto y = silu_affine(x_nc, w_nc, b_nc);
  std::cerr << "S1_OP_BACK\n" << std::flush;
  sync();
  auto y_ref = torch_silu_affine(x_nc, w_nc, b_nc);
  sync();
  const double fwd_rel = rel_err(y, y_ref);
  expect(fwd_rel <= 1e-15, "forward rel " + std::to_string(fwd_rel));

  // Bitwise repeats.
  bool bitwise = true;
  auto y0 = y.to(torch::kCPU);
  for (int rep = 0; rep < 10; ++rep) {
    auto yr = silu_affine(x_nc, w_nc, b_nc);
    sync();
    if (!torch::equal(yr.to(torch::kCPU), y0)) bitwise = false;
  }
  expect(bitwise, "bitwise across 10 repeats");

  // Autograd versus the analytic torch formula, gy = 1.
  auto xg = x_nc.detach().clone().requires_grad_(true);
  auto wg = w_nc.detach().clone().requires_grad_(true);
  auto bg = b_nc.detach().clone().requires_grad_(true);
  auto yg = silu_affine(xg, wg, bg);
  torch::autograd::variable_list outs{yg.sum()};
  torch::autograd::variable_list ins{xg, wg, bg};
  auto grads = torch::autograd::grad(outs, ins);
  sync();
  auto z = xg.detach() * wg.detach() + bg.detach();
  auto sig = torch::sigmoid(z);
  auto dsilu = sig * (1.0 + z * (1.0 - sig));
  const double gx_rel = rel_err(grads[0], dsilu * wg.detach());
  const double gw_rel = rel_err(grads[1], dsilu * xg.detach());
  const double gb_rel = rel_err(grads[2], dsilu);
  expect(gx_rel <= 1e-15 && gw_rel <= 1e-15 && gb_rel <= 1e-15,
         "analytic grad rel gx " + std::to_string(gx_rel) + " gw " + std::to_string(gw_rel) +
             " gb " + std::to_string(gb_rel));

  // Central finite difference on a few elements.
  const double eps = 1e-6;
  const int64_t probes[] = {0, 1, 2, n / 4, n / 2};
  double fd_max = 0.0;
  auto probe = [&](torch::Tensor& var, const torch::Tensor& grad, const char* name) {
    for (int64_t i : probes) {
      if (i >= var.size(0)) continue;
      auto plus = var.detach().clone();
      auto minus = var.detach().clone();
      plus.index_put_({i}, plus.index({i}).item<double>() + eps);
      minus.index_put_({i}, minus.index({i}).item<double>() - eps);
      torch::Tensor yp, ym;
      if (name[0] == 'x') {
        yp = silu_affine(plus, wg.detach(), bg.detach());
        ym = silu_affine(minus, wg.detach(), bg.detach());
      } else if (name[0] == 'w') {
        yp = silu_affine(xg.detach(), plus, bg.detach());
        ym = silu_affine(xg.detach(), minus, bg.detach());
      } else {
        yp = silu_affine(xg.detach(), wg.detach(), plus);
        ym = silu_affine(xg.detach(), wg.detach(), minus);
      }
      sync();
      const double num = (yp.index({i}) - ym.index({i})).item<double>() / (2.0 * eps);
      const double ana = grad.index({i}).item<double>();
      fd_max = std::max(fd_max, std::abs(num - ana));
    }
  };
  probe(xg, grads[0], "x");
  probe(wg, grads[1], "w");
  probe(bg, grads[2], "b");
  expect(fd_max <= 1e-9, "finite difference max abs " + std::to_string(fd_max));

  // Checkpoint-style: no-grad forward, then a grad forward. Outputs must match.
  torch::Tensor y_ng;
  {
    torch::NoGradGuard guard;
    y_ng = silu_affine(xg.detach(), wg.detach(), bg.detach());
  }
  auto x2 = xg.detach().clone().requires_grad_(true);
  auto w2 = wg.detach().clone().requires_grad_(true);
  auto b2 = bg.detach().clone().requires_grad_(true);
  auto y_g = silu_affine(x2, w2, b2);
  sync();
  expect(torch::equal(y_ng.to(torch::kCPU), y_g.detach().to(torch::kCPU)),
         "no-grad forward matches the recompute");
  torch::autograd::variable_list outs2{y_g.sum()};
  torch::autograd::variable_list ins2{x2, w2, b2};
  auto g2 = torch::autograd::grad(outs2, ins2);
  sync();
  expect(rel_err(g2[0], grads[0]) <= 1e-15, "recompute grad matches");

  if (g_fails != 0) {
    std::cerr << "S1_FAIL cases=" << g_fails << "\n";
    return 1;
  }
  std::cerr << "S1_PASS\n";
  return 0;
}
