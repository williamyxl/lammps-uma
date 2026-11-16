// Step 23 (N5.4). Fused GateActivation for lmax=mmax=2, m_prime=True.
// y[0] = silu(x[0]); y[1+k] = x[1+k] * sigmoid(g[expand[k]]).
// Not wired into the model. The timed backward is the SYCL kernel.
// gate_scaffold's backward is torch autograd on the stock math and is not timed.

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

void fwd_channel(const double* g, const double* x, double* y, const int64_t* expand,
                 int64_t lmax, int64_t ncomp, int64_t nvec, int64_t c, int64_t C) {
  double sig[8];
  for (int64_t l = 0; l < lmax; ++l) {
    sig[l] = sigmoid_stable(g[l * C + c]);
  }
  const double xs = x[c];
  y[c] = xs * sigmoid_stable(xs);
  for (int64_t k = 0; k < nvec; ++k) {
    const int64_t l = expand[k];
    y[(k + 1) * C + c] = x[(k + 1) * C + c] * sig[l];
  }
  (void)ncomp;
}

void bwd_channel(const double* g, const double* x, const double* gy, double* gx, double* gg,
                 const int64_t* expand, int64_t lmax, int64_t nvec, int64_t c, int64_t C) {
  double sig[8];
  double ds[8];
  for (int64_t l = 0; l < lmax; ++l) {
    const double s = sigmoid_stable(g[l * C + c]);
    sig[l] = s;
    ds[l] = s * (1.0 - s);
  }
  const double xs = x[c];
  const double s0 = sigmoid_stable(xs);
  const double silu_d = s0 * (1.0 + xs * (1.0 - s0));
  gx[c] = gy[c] * silu_d;
  double acc[8] = {};
  for (int64_t k = 0; k < nvec; ++k) {
    const int64_t l = expand[k];
    const double xv = x[(k + 1) * C + c];
    const double gyv = gy[(k + 1) * C + c];
    gx[(k + 1) * C + c] = gyv * sig[l];
    acc[l] += gyv * xv * ds[l];
  }
  for (int64_t l = 0; l < lmax; ++l) {
    gg[l * C + c] = acc[l];
  }
}

struct Dim {
  int64_t E;
  int64_t ncomp;
  int64_t C;
  int64_t lmax;
  int64_t nvec;
};

Dim dims_of(const at::Tensor& g, const at::Tensor& x, const at::Tensor& expand) {
  TORCH_CHECK(g.dtype() == at::kDouble && x.dtype() == at::kDouble, "f2: FP64 required");
  TORCH_CHECK(g.device().is_xpu() && x.device().is_xpu() && expand.device().is_xpu(), "f2: XPU required");
  TORCH_CHECK(expand.dtype() == at::kLong, "f2: expand is int64");
  TORCH_CHECK(g.dim() == 2 && x.dim() == 3, "f2: g [E, lmax*C], x [E, ncomp, C]");
  Dim d;
  d.E = x.size(0);
  d.ncomp = x.size(1);
  d.C = x.size(2);
  d.nvec = expand.size(0);
  TORCH_CHECK(d.ncomp == d.nvec + 1, "f2: x components must be expand + 1");
  TORCH_CHECK(g.size(0) == d.E && g.size(1) % d.C == 0, "f2: g shape");
  d.lmax = g.size(1) / d.C;
  TORCH_CHECK(d.lmax >= 1 && d.lmax <= 8, "f2: lmax must be 1..8");
  TORCH_CHECK(d.nvec <= 64, "f2: expand longer than 64");
  return d;
}

template <int WG, int SG>
struct FwdSG {
  const double* g;
  const double* x;
  double* y;
  const int64_t* expand;
  Dim d;
  [[sycl::reqd_sub_group_size(SG)]] void operator()(sycl::nd_item<1> it) const {
    const int64_t e = static_cast<int64_t>(it.get_group(0));
    const int tid = static_cast<int>(it.get_local_id(0));
    const double* ge = g + e * d.lmax * d.C;
    const double* xe = x + e * d.ncomp * d.C;
    double* ye = y + e * d.ncomp * d.C;
    for (int64_t c = tid; c < d.C; c += WG) {
      fwd_channel(ge, xe, ye, expand, d.lmax, d.ncomp, d.nvec, c, d.C);
    }
  }
};

template <int WG>
struct Fwd {
  const double* g;
  const double* x;
  double* y;
  const int64_t* expand;
  Dim d;
  void operator()(sycl::nd_item<1> it) const {
    const int64_t e = static_cast<int64_t>(it.get_group(0));
    const int tid = static_cast<int>(it.get_local_id(0));
    const double* ge = g + e * d.lmax * d.C;
    const double* xe = x + e * d.ncomp * d.C;
    double* ye = y + e * d.ncomp * d.C;
    for (int64_t c = tid; c < d.C; c += WG) {
      fwd_channel(ge, xe, ye, expand, d.lmax, d.ncomp, d.nvec, c, d.C);
    }
  }
};

template <int WG, int SG>
struct BwdSG {
  const double* g;
  const double* x;
  const double* gy;
  double* gx;
  double* gg;
  const int64_t* expand;
  Dim d;
  [[sycl::reqd_sub_group_size(SG)]] void operator()(sycl::nd_item<1> it) const {
    const int64_t e = static_cast<int64_t>(it.get_group(0));
    const int tid = static_cast<int>(it.get_local_id(0));
    const int64_t go = e * d.lmax * d.C;
    const int64_t xo = e * d.ncomp * d.C;
    for (int64_t c = tid; c < d.C; c += WG) {
      bwd_channel(g + go, x + xo, gy + xo, gx + xo, gg + go, expand, d.lmax, d.nvec, c, d.C);
    }
  }
};

template <int WG>
struct Bwd {
  const double* g;
  const double* x;
  const double* gy;
  double* gx;
  double* gg;
  const int64_t* expand;
  Dim d;
  void operator()(sycl::nd_item<1> it) const {
    const int64_t e = static_cast<int64_t>(it.get_group(0));
    const int tid = static_cast<int>(it.get_local_id(0));
    const int64_t go = e * d.lmax * d.C;
    const int64_t xo = e * d.ncomp * d.C;
    for (int64_t c = tid; c < d.C; c += WG) {
      bwd_channel(g + go, x + xo, gy + xo, gx + xo, gg + go, expand, d.lmax, d.nvec, c, d.C);
    }
  }
};

template <int WG>
void launch_fwd(sycl::queue& q, const Dim& d, const double* g, const double* x, double* y,
                const int64_t* expand) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                 Fwd<WG>{g, x, y, expand, d})
      .wait();
}

template <int WG, int SG>
void launch_fwd_sg(sycl::queue& q, const Dim& d, const double* g, const double* x, double* y,
                   const int64_t* expand) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                 FwdSG<WG, SG>{g, x, y, expand, d})
      .wait();
}

template <int WG>
void launch_bwd(sycl::queue& q, const Dim& d, const double* g, const double* x, const double* gy,
                double* gx, double* gg, const int64_t* expand) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                 Bwd<WG>{g, x, gy, gx, gg, expand, d})
      .wait();
}

template <int WG, int SG>
void launch_bwd_sg(sycl::queue& q, const Dim& d, const double* g, const double* x, const double* gy,
                   double* gx, double* gg, const int64_t* expand) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                 BwdSG<WG, SG>{g, x, gy, gx, gg, expand, d})
      .wait();
}

template <int WG>
void launch_fwd_local(sycl::queue& q, const Dim& d, const double* g, const double* x, double* y,
                      const int64_t* expand) {
  const size_t slm_n = static_cast<size_t>(d.lmax * d.C + d.ncomp * d.C);
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(slm_n), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         const int tid = static_cast<int>(it.get_local_id(0));
                         double* lg = &slm[0];
                         double* lx = lg + d.lmax * d.C;
                         const double* ge = g + e * d.lmax * d.C;
                         const double* xe = x + e * d.ncomp * d.C;
                         for (int64_t i = tid; i < d.lmax * d.C; i += WG) lg[i] = ge[i];
                         for (int64_t i = tid; i < d.ncomp * d.C; i += WG) lx[i] = xe[i];
                         it.barrier(sycl::access::fence_space::local_space);
                         double* ye = y + e * d.ncomp * d.C;
                         for (int64_t c = tid; c < d.C; c += WG) {
                           fwd_channel(lg, lx, ye, expand, d.lmax, d.ncomp, d.nvec, c, d.C);
                         }
                       });
      })
      .wait();
}

template <int WG>
void launch_bwd_local(sycl::queue& q, const Dim& d, const double* g, const double* x, const double* gy,
                      double* gx, double* gg, const int64_t* expand) {
  const size_t slm_n = static_cast<size_t>(d.lmax * d.C + 2 * d.ncomp * d.C);
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(slm_n), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         const int tid = static_cast<int>(it.get_local_id(0));
                         double* lg = &slm[0];
                         double* lx = lg + d.lmax * d.C;
                         double* lgy = lx + d.ncomp * d.C;
                         const int64_t go = e * d.lmax * d.C;
                         const int64_t xo = e * d.ncomp * d.C;
                         for (int64_t i = tid; i < d.lmax * d.C; i += WG) lg[i] = g[go + i];
                         for (int64_t i = tid; i < d.ncomp * d.C; i += WG) lx[i] = x[xo + i];
                         for (int64_t i = tid; i < d.ncomp * d.C; i += WG) lgy[i] = gy[xo + i];
                         it.barrier(sycl::access::fence_space::local_space);
                         for (int64_t c = tid; c < d.C; c += WG) {
                           bwd_channel(lg, lx, lgy, gx + xo, gg + go, expand, d.lmax, d.nvec, c, d.C);
                         }
                       });
      })
      .wait();
}

void gate_forward(const at::Tensor& g, const at::Tensor& x, const at::Tensor& expand, at::Tensor& y,
                  int64_t variant) {
  const Dim d = dims_of(g, x, expand);
  auto gc = g.contiguous();
  auto xc = x.contiguous();
  auto ec = expand.contiguous();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  const double* gp = gc.data_ptr<double>();
  const double* xp = xc.data_ptr<double>();
  double* yp = y.data_ptr<double>();
  const int64_t* ep = ec.data_ptr<int64_t>();
  switch (variant) {
    case 0: launch_fwd<128>(q, d, gp, xp, yp, ep); break;
    case 1: launch_fwd<64>(q, d, gp, xp, yp, ep); break;
    case 2: launch_fwd<32>(q, d, gp, xp, yp, ep); break;
    case 3: launch_fwd_sg<128, 16>(q, d, gp, xp, yp, ep); break;
    case 4: launch_fwd_sg<128, 32>(q, d, gp, xp, yp, ep); break;
    case 5: launch_fwd_local<128>(q, d, gp, xp, yp, ep); break;
    case 6: launch_fwd_local<64>(q, d, gp, xp, yp, ep); break;
    case 7: launch_fwd_local<32>(q, d, gp, xp, yp, ep); break;
    default: TORCH_CHECK(false, "f2: unknown variant ", variant);
  }
}

void gate_backward(const at::Tensor& g, const at::Tensor& x, const at::Tensor& gy, at::Tensor& gx,
                   at::Tensor& gg, const at::Tensor& expand, int64_t variant) {
  const Dim d = dims_of(g, x, expand);
  auto gc = g.contiguous();
  auto xc = x.contiguous();
  auto gyc = gy.contiguous();
  auto ec = expand.contiguous();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  const double* gp = gc.data_ptr<double>();
  const double* xp = xc.data_ptr<double>();
  const double* gyp = gyc.data_ptr<double>();
  double* gxp = gx.data_ptr<double>();
  double* ggp = gg.data_ptr<double>();
  const int64_t* ep = ec.data_ptr<int64_t>();
  switch (variant) {
    case 0: launch_bwd<128>(q, d, gp, xp, gyp, gxp, ggp, ep); break;
    case 1: launch_bwd<64>(q, d, gp, xp, gyp, gxp, ggp, ep); break;
    case 2: launch_bwd<32>(q, d, gp, xp, gyp, gxp, ggp, ep); break;
    case 3: launch_bwd_sg<128, 16>(q, d, gp, xp, gyp, gxp, ggp, ep); break;
    case 4: launch_bwd_sg<128, 32>(q, d, gp, xp, gyp, gxp, ggp, ep); break;
    case 5: launch_bwd_local<128>(q, d, gp, xp, gyp, gxp, ggp, ep); break;
    case 6: launch_bwd_local<64>(q, d, gp, xp, gyp, gxp, ggp, ep); break;
    case 7: launch_bwd_local<32>(q, d, gp, xp, gyp, gxp, ggp, ep); break;
    default: TORCH_CHECK(false, "f2: unknown variant ", variant);
  }
}

at::Tensor aten_gate(const at::Tensor& g, const at::Tensor& x, const at::Tensor& expand) {
  const auto E = x.size(0);
  const auto ncomp = x.size(1);
  const auto C = x.size(2);
  const auto lmax = g.size(1) / C;
  auto gs = at::sigmoid(g).view({E, lmax, C});
  gs = at::index_select(gs, 1, expand);
  auto scalar = at::silu(x.slice(1, 0, 1));
  auto vec = x.slice(1, 1, ncomp) * gs;
  return at::cat({scalar, vec}, 1);
}

class GateAutograd : public torch::autograd::Function<GateAutograd> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor g, torch::Tensor x,
                               torch::Tensor expand, int64_t variant) {
    ctx->save_for_backward({g, x, expand});
    ctx->saved_data["variant"] = variant;
    auto y = torch::empty_like(x);
    gate_forward(g, x, expand, y, variant);
    return y;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    const int64_t variant = ctx->saved_data["variant"].toInt();
    auto gx = torch::empty_like(saved[1]);
    auto gg = torch::empty_like(saved[0]);
    gate_backward(saved[0], saved[1], grad_outputs[0], gx, gg, saved[2], variant);
    return {gg, gx, torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor gate_autograd(const torch::Tensor& g, const torch::Tensor& x, const torch::Tensor& expand,
                            int64_t variant) {
  return GateAutograd::apply(g, x, expand, variant);
}

std::tuple<at::Tensor, at::Tensor> gate_scaffold(const torch::Tensor& g, const torch::Tensor& x,
                                                 const torch::Tensor& expand) {
  auto gv = g.detach().requires_grad_(true);
  auto xv = x.detach().requires_grad_(true);
  auto y = aten_gate(gv, xv, expand);
  torch::autograd::variable_list outs;
  outs.push_back(y.sum());
  torch::autograd::variable_list ins;
  ins.push_back(gv);
  ins.push_back(xv);
  auto grads = torch::autograd::grad(outs, ins);
  return {grads[0], grads[1]};
}

}  // namespace

extern "C" void uma_n5f2_anchor() {}

TORCH_LIBRARY(uma_n5f2, m) {
  m.def("gate(Tensor g, Tensor x, Tensor expand, int variant) -> Tensor");
  m.def("gate_scaffold(Tensor g, Tensor x, Tensor expand) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(uma_n5f2, XPU, m) {
  m.impl("gate", [](const at::Tensor& g, const at::Tensor& x, const at::Tensor& expand, int64_t variant) {
    auto y = torch::empty_like(x);
    gate_forward(g, x, expand, y, variant);
    return y;
  });
}

TORCH_LIBRARY_IMPL(uma_n5f2, Autograd, m) {
  m.impl("gate", &gate_autograd);
}

TORCH_LIBRARY_IMPL(uma_n5f2, CompositeExplicitAutograd, m) {
  m.impl("gate_scaffold", &gate_scaffold);
}
