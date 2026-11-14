// Step 13 (S2). Load a traced module that calls uma_sycl::silu_affine.
// The op is linked into this process. No Python at runtime.

#include <torch/script.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <string>

namespace {

void sync() { torch::xpu::synchronize(0); }

double rel_err(const torch::Tensor& got, const torch::Tensor& ref) {
  auto d = (got - ref).abs();
  const double scale = std::max(1.0, ref.abs().max().item<double>());
  return d.max().item<double>() / scale;
}

struct Inputs {
  torch::Tensor x;
  torch::Tensor w;
  torch::Tensor b;
};

Inputs make_inputs(int64_t n, bool grad) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kXPU);
  auto idx = torch::arange(n, opt);
  Inputs in;
  in.x = torch::sin(idx * 0.01);
  in.w = torch::cos(idx * 0.02) * 0.5 + 0.1;
  in.b = torch::sin(idx * 0.007) * 2.0;
  if (grad) {
    in.x.requires_grad_(true);
    in.w.requires_grad_(true);
    in.b.requires_grad_(true);
  }
  return in;
}

torch::Tensor eager(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b) {
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

void check_shape(torch::jit::script::Module& m, int64_t n) {
  auto in = make_inputs(n, true);
  torch::jit::IValue out = m.forward({in.x, in.w, in.b});
  auto y = out.toTensor();
  sync();
  auto y_ref = eager(in.x.detach(), in.w.detach(), in.b.detach());
  sync();
  const double fwd = rel_err(y, y_ref);
  expect(fwd <= 1e-15, "n=" + std::to_string(n) + " forward rel " + std::to_string(fwd));

  torch::autograd::variable_list outs{y.sum()};
  torch::autograd::variable_list ins{in.x, in.w, in.b};
  auto grads = torch::autograd::grad(outs, ins);
  sync();
  auto z = in.x.detach() * in.w.detach() + in.b.detach();
  auto sig = torch::sigmoid(z);
  auto dsilu = sig * (1.0 + z * (1.0 - sig));
  const double gx = rel_err(grads[0], dsilu * in.w.detach());
  const double gw = rel_err(grads[1], dsilu * in.x.detach());
  const double gb = rel_err(grads[2], dsilu);
  expect(gx <= 1e-9 && gw <= 1e-9 && gb <= 1e-9,
         "n=" + std::to_string(n) + " grad rel gx " + std::to_string(gx) + " gw " +
             std::to_string(gw) + " gb " + std::to_string(gb));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: s2_test TRACED.pt\n";
    return 2;
  }
  std::cerr << "S2_ENTER\n" << std::flush;
  if (!torch::xpu::is_available()) {
    std::cerr << "S2_FAIL no XPU\n";
    return 1;
  }
  torch::jit::script::Module m;
  try {
    m = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "S2_LOAD_FAIL " << e.what() << "\n";
    return 1;
  }
  std::cerr << "S2_LOAD_OK\n" << std::flush;
  m.to(torch::kXPU);
  check_shape(m, 128);
  check_shape(m, 1000);
  if (g_fails != 0) {
    std::cerr << "S2_FAIL cases=" << g_fails << "\n";
    return 1;
  }
  std::cerr << "S2_PASS\n";
  return 0;
}
