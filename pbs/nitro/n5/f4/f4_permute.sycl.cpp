// Step 24 (N5.5). Fused gather + concat + Wigner rotation.
// y = wigner @ cat(x[src], x[dst]) along the channel axis.
// fc_m0 and the block GEMMs stay on the vendor library; this is the prologue.
// Not wired into the model.

#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>
#include <sycl/sycl.hpp>

namespace {

struct Dim {
  int64_t N;
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
  d.N = x.size(0);
  d.L = x.size(1);
  d.C = x.size(2);
  d.E = edge.size(1);
  d.M = wigner.size(1);
  TORCH_CHECK(edge.size(0) == 2 && wigner.size(0) == d.E && wigner.size(2) == d.L, "f4: edge or wigner shape");
  TORCH_CHECK(d.L <= 16 && d.M <= 16 && d.C > 0, "f4: L and M must be <= 16");
  return d;
}

template <int WG>
void fwd_edge(sycl::nd_item<1> it, const double* x, const int64_t* edge, const double* we,
              double* y, const Dim d) {
  const int64_t e = static_cast<int64_t>(it.get_group(0));
  const int tid = static_cast<int>(it.get_local_id(0));
  const int64_t src = edge[e];
  const int64_t dst = edge[d.E + e];
  const double* xs = x + (src * d.L) * d.C;
  const double* xt = x + (dst * d.L) * d.C;
  double* ye = y + (e * d.M) * (2 * d.C);
  for (int64_t m = 0; m < d.M; ++m) {
    const double* wm = we + m * d.L;
    for (int64_t c = tid; c < d.C; c += WG) {
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
}

template <int WG>
struct Fwd {
  const double* x;
  const int64_t* edge;
  const double* w;
  double* y;
  Dim d;
  void operator()(sycl::nd_item<1> it) const {
    const int64_t e = static_cast<int64_t>(it.get_group(0));
    fwd_edge<WG>(it, x, edge, w + (e * d.M) * d.L, y, d);
  }
};

template <int WG, int SG>
struct FwdSG {
  const double* x;
  const int64_t* edge;
  const double* w;
  double* y;
  Dim d;
  [[sycl::reqd_sub_group_size(SG)]] void operator()(sycl::nd_item<1> it) const {
    const int64_t e = static_cast<int64_t>(it.get_group(0));
    fwd_edge<WG>(it, x, edge, w + (e * d.M) * d.L, y, d);
  }
};

template <int WG>
void launch_fwd(sycl::queue& q, const Dim& d, const double* x, const int64_t* edge, const double* w,
                double* y) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                 Fwd<WG>{x, edge, w, y, d})
      .wait();
}

template <int WG, int SG>
void launch_fwd_sg(sycl::queue& q, const Dim& d, const double* x, const int64_t* edge, const double* w,
                   double* y) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                 FwdSG<WG, SG>{x, edge, w, y, d})
      .wait();
}

template <int WG>
void launch_fwd_local(sycl::queue& q, const Dim& d, const double* x, const int64_t* edge, const double* w,
                      double* y) {
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(static_cast<size_t>(d.M * d.L)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         const int tid = static_cast<int>(it.get_local_id(0));
                         double* lw = &slm[0];
                         const double* we = w + (e * d.M) * d.L;
                         for (int64_t i = tid; i < d.M * d.L; i += WG) lw[i] = we[i];
                         it.barrier(sycl::access::fence_space::local_space);
                         fwd_edge<WG>(it, x, edge, lw, y, d);
                       });
      })
      .wait();
}

using Atomic = sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                sycl::access::address_space::global_space>;

template <int WG>
void bwd_edge(sycl::nd_item<1> it, const double* x, const int64_t* edge, const double* w, const double* gy,
              double* gx, double* gw, const Dim d) {
  const int64_t e = static_cast<int64_t>(it.get_group(0));
  const int tid = static_cast<int>(it.get_local_id(0));
  const int64_t src = edge[e];
  const int64_t dst = edge[d.E + e];
  const double* xs = x + (src * d.L) * d.C;
  const double* xt = x + (dst * d.L) * d.C;
  const double* gye = gy + (e * d.M) * (2 * d.C);
  double* gwe = gw + (e * d.M) * d.L;
  for (int64_t l = 0; l < d.L; ++l) {
    for (int64_t c = tid; c < d.C; c += WG) {
      double gs = 0.0;
      double gt = 0.0;
      for (int64_t m = 0; m < d.M; ++m) {
        const double wv = w[((e * d.M + m) * d.L) + l];
        gs += wv * gye[m * (2 * d.C) + c];
        gt += wv * gye[m * (2 * d.C) + d.C + c];
      }
      Atomic(gx[((src * d.L + l) * d.C) + c]).fetch_add(gs);
      Atomic(gx[((dst * d.L + l) * d.C) + c]).fetch_add(gt);
    }
  }
  for (int64_t m = 0; m < d.M; ++m) {
    for (int64_t l = 0; l < d.L; ++l) {
      double partial = 0.0;
      for (int64_t c = tid; c < d.C; c += WG) {
        partial += gye[m * (2 * d.C) + c] * xs[l * d.C + c];
        partial += gye[m * (2 * d.C) + d.C + c] * xt[l * d.C + c];
      }
      // One channel owner writes when WG covers C in a single stride and tid==0
      // collects via a local reduction done by the caller. Here each thread
      // keeps its partial; the launch function reduces.
      (void)partial;
      (void)gwe;
    }
  }
}

template <int WG>
void launch_bwd(sycl::queue& q, const Dim& d, const double* x, const int64_t* edge, const double* w,
                const double* gy, double* gx, double* gw) {
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(static_cast<size_t>(WG)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         const int tid = static_cast<int>(it.get_local_id(0));
                         const int64_t src = edge[e];
                         const int64_t dst = edge[d.E + e];
                         const double* xs = x + (src * d.L) * d.C;
                         const double* xt = x + (dst * d.L) * d.C;
                         const double* gye = gy + (e * d.M) * (2 * d.C);
                         const double* we = w + (e * d.M) * d.L;
                         double* gwe = gw + (e * d.M) * d.L;
                         double* red = &slm[0];
                         for (int64_t l = 0; l < d.L; ++l) {
                           for (int64_t c = tid; c < d.C; c += WG) {
                             double gs = 0.0;
                             double gt = 0.0;
                             for (int64_t m = 0; m < d.M; ++m) {
                               const double wv = we[m * d.L + l];
                               gs += wv * gye[m * (2 * d.C) + c];
                               gt += wv * gye[m * (2 * d.C) + d.C + c];
                             }
                             Atomic(gx[((src * d.L + l) * d.C) + c]).fetch_add(gs);
                             Atomic(gx[((dst * d.L + l) * d.C) + c]).fetch_add(gt);
                           }
                         }
                         for (int64_t m = 0; m < d.M; ++m) {
                           for (int64_t l = 0; l < d.L; ++l) {
                             double partial = 0.0;
                             for (int64_t c = tid; c < d.C; c += WG) {
                               partial += gye[m * (2 * d.C) + c] * xs[l * d.C + c];
                               partial += gye[m * (2 * d.C) + d.C + c] * xt[l * d.C + c];
                             }
                             red[tid] = partial;
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int offset = WG / 2; offset > 0; offset >>= 1) {
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

void permute_forward(const at::Tensor& x, const at::Tensor& edge, const at::Tensor& wigner, at::Tensor& y,
                     int64_t variant) {
  const Dim d = check_dims(x, edge, wigner);
  auto xc = x.contiguous();
  auto ec = edge.contiguous();
  auto wc = wigner.contiguous();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  const double* xp = xc.data_ptr<double>();
  const int64_t* ep = ec.data_ptr<int64_t>();
  const double* wp = wc.data_ptr<double>();
  double* yp = y.data_ptr<double>();
  switch (variant) {
    case 0: launch_fwd<128>(q, d, xp, ep, wp, yp); break;
    case 1: launch_fwd<64>(q, d, xp, ep, wp, yp); break;
    case 2: launch_fwd<32>(q, d, xp, ep, wp, yp); break;
    case 3: launch_fwd_sg<128, 16>(q, d, xp, ep, wp, yp); break;
    case 4: launch_fwd_sg<128, 32>(q, d, xp, ep, wp, yp); break;
    case 5: launch_fwd_local<128>(q, d, xp, ep, wp, yp); break;
    case 6: launch_fwd_local<64>(q, d, xp, ep, wp, yp); break;
    case 7: launch_fwd_local<32>(q, d, xp, ep, wp, yp); break;
    default: TORCH_CHECK(false, "f4: unknown variant ", variant);
  }
}

void permute_backward(const at::Tensor& x, const at::Tensor& edge, const at::Tensor& wigner,
                      const at::Tensor& gy, at::Tensor& gx, at::Tensor& gw, int64_t variant) {
  const Dim d = check_dims(x, edge, wigner);
  auto xc = x.contiguous();
  auto ec = edge.contiguous();
  auto wc = wigner.contiguous();
  auto gyc = gy.contiguous();
  gx.zero_();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  const double* xp = xc.data_ptr<double>();
  const int64_t* ep = ec.data_ptr<int64_t>();
  const double* wp = wc.data_ptr<double>();
  const double* gyp = gyc.data_ptr<double>();
  double* gxp = gx.data_ptr<double>();
  double* gwp = gw.data_ptr<double>();
  const int which = variant >= 5 ? variant - 5 : variant;
  switch (which) {
    case 0: launch_bwd<128>(q, d, xp, ep, wp, gyp, gxp, gwp); break;
    case 1: launch_bwd<64>(q, d, xp, ep, wp, gyp, gxp, gwp); break;
    case 2: launch_bwd<32>(q, d, xp, ep, wp, gyp, gxp, gwp); break;
    case 3: launch_bwd<128>(q, d, xp, ep, wp, gyp, gxp, gwp); break;
    case 4: launch_bwd<128>(q, d, xp, ep, wp, gyp, gxp, gwp); break;
    default: TORCH_CHECK(false, "f4: unknown variant ", variant);
  }
}

at::Tensor aten_permute(const at::Tensor& x, const at::Tensor& edge, const at::Tensor& wigner) {
  auto cat = at::cat({x.index_select(0, edge[0]), x.index_select(0, edge[1])}, 2);
  return at::bmm(wigner, cat);
}

class PermuteAutograd : public torch::autograd::Function<PermuteAutograd> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x, torch::Tensor edge,
                               torch::Tensor wigner, int64_t variant) {
    ctx->save_for_backward({x, edge, wigner});
    ctx->saved_data["variant"] = variant;
    auto y = torch::empty({edge.size(1), wigner.size(1), x.size(2) * 2}, x.options());
    permute_forward(x, edge, wigner, y, variant);
    return y;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    const int64_t variant = ctx->saved_data["variant"].toInt();
    auto gx = torch::empty_like(saved[0]);
    auto gw = torch::empty_like(saved[2]);
    permute_backward(saved[0], saved[1], saved[2], grad_outputs[0], gx, gw, variant);
    return {gx, torch::Tensor(), gw, torch::Tensor()};
  }
};

torch::Tensor permute_autograd(const torch::Tensor& x, const torch::Tensor& edge, const torch::Tensor& wigner,
                               int64_t variant) {
  return PermuteAutograd::apply(x, edge, wigner, variant);
}

std::tuple<at::Tensor, at::Tensor> permute_scaffold(const at::Tensor& x, const at::Tensor& edge,
                                                    const at::Tensor& wigner) {
  auto xv = x.detach().requires_grad_(true);
  auto wv = wigner.detach().requires_grad_(true);
  auto y = aten_permute(xv, edge, wv);
  torch::autograd::variable_list outs;
  outs.push_back(y.sum());
  torch::autograd::variable_list ins;
  ins.push_back(xv);
  ins.push_back(wv);
  auto grads = torch::autograd::grad(outs, ins);
  return {grads[0], grads[1]};
}

}  // namespace

extern "C" void uma_n5f4_anchor() {}

TORCH_LIBRARY(uma_n5f4, m) {
  m.def("permute(Tensor x, Tensor edge_index, Tensor wigner, int variant) -> Tensor");
  m.def("permute_scaffold(Tensor x, Tensor edge_index, Tensor wigner) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(uma_n5f4, XPU, m) {
  m.impl("permute", [](const at::Tensor& x, const at::Tensor& edge, const at::Tensor& wigner, int64_t variant) {
    auto y = torch::empty({edge.size(1), wigner.size(1), x.size(2) * 2}, x.options());
    permute_forward(x, edge, wigner, y, variant);
    return y;
  });
}

TORCH_LIBRARY_IMPL(uma_n5f4, Autograd, m) { m.impl("permute", &permute_autograd); }

TORCH_LIBRARY_IMPL(uma_n5f4, CompositeExplicitAutograd, m) { m.impl("permute_scaffold", &permute_scaffold); }
