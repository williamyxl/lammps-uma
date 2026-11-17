// Step 25 (N5.6). Fused inverse Wigner rotation and scatter.
// y[center] += wigner_inv @ x, with center = edge_index[1] - node_offset.
// N2-b was not adopted, so the scatter stays an atomic add, the same math as
// index_add_. conv2 GEMMs stay on the vendor library. Not wired into the model.

#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>
#include <sycl/sycl.hpp>

namespace {

struct Dim {
  int64_t E;
  int64_t L;
  int64_t M;
  int64_t C;
  int64_t N;
  int64_t offset;
};

Dim check_dims(const at::Tensor& x, const at::Tensor& w, const at::Tensor& edge, int64_t n_nodes,
               int64_t offset) {
  TORCH_CHECK(x.dtype() == at::kDouble && w.dtype() == at::kDouble, "f5: FP64 required");
  TORCH_CHECK(x.device().is_xpu() && w.device().is_xpu() && edge.device().is_xpu(), "f5: XPU required");
  TORCH_CHECK(edge.dtype() == at::kLong && x.dim() == 3 && w.dim() == 3 && edge.dim() == 2, "f5: shapes");
  Dim d;
  d.E = x.size(0);
  d.M = x.size(1);
  d.C = x.size(2);
  d.L = w.size(1);
  d.N = n_nodes;
  d.offset = offset;
  TORCH_CHECK(w.size(0) == d.E && w.size(2) == d.M && edge.size(0) == 2 && edge.size(1) == d.E,
              "f5: wigner or edge shape");
  TORCH_CHECK(d.L <= 16 && d.M <= 16 && d.C > 0 && d.N > 0, "f5: L and M must be <= 16");
  return d;
}

using Atomic = sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                sycl::access::address_space::global_space>;

template <int WG>
void fwd_edge(sycl::nd_item<1> it, const double* x, const double* we, const int64_t* edge, double* y,
              Dim d) {
  const int64_t e = static_cast<int64_t>(it.get_group(0));
  const int tid = static_cast<int>(it.get_local_id(0));
  const int64_t center = edge[d.E + e] - d.offset;
  const double* xe = x + (e * d.M) * d.C;
  double* yn = y + (center * d.L) * d.C;
  for (int64_t l = 0; l < d.L; ++l) {
    const double* wl = we + l * d.M;
    for (int64_t c = tid; c < d.C; c += WG) {
      double acc = 0.0;
      for (int64_t m = 0; m < d.M; ++m) acc += wl[m] * xe[m * d.C + c];
      Atomic(yn[l * d.C + c]).fetch_add(acc);
    }
  }
}

template <int WG>
struct Fwd {
  const double* x;
  const double* w;
  const int64_t* edge;
  double* y;
  Dim d;
  void operator()(sycl::nd_item<1> it) const {
    const int64_t e = static_cast<int64_t>(it.get_group(0));
    fwd_edge<WG>(it, x, w + (e * d.L) * d.M, edge, y, d);
  }
};

template <int WG, int SG>
struct FwdSG {
  const double* x;
  const double* w;
  const int64_t* edge;
  double* y;
  Dim d;
  [[sycl::reqd_sub_group_size(SG)]] void operator()(sycl::nd_item<1> it) const {
    const int64_t e = static_cast<int64_t>(it.get_group(0));
    fwd_edge<WG>(it, x, w + (e * d.L) * d.M, edge, y, d);
  }
};

template <int WG>
void launch_fwd(sycl::queue& q, Dim d, const double* x, const double* w, const int64_t* edge, double* y) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                 Fwd<WG>{x, w, edge, y, d})
      .wait();
}

template <int WG, int SG>
void launch_fwd_sg(sycl::queue& q, Dim d, const double* x, const double* w, const int64_t* edge, double* y) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                 FwdSG<WG, SG>{x, w, edge, y, d})
      .wait();
}

template <int WG>
void launch_fwd_local(sycl::queue& q, Dim d, const double* x, const double* w, const int64_t* edge, double* y) {
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(static_cast<size_t>(d.L * d.M)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         const int tid = static_cast<int>(it.get_local_id(0));
                         double* lw = &slm[0];
                         const double* we = w + (e * d.L) * d.M;
                         for (int64_t i = tid; i < d.L * d.M; i += WG) lw[i] = we[i];
                         it.barrier(sycl::access::fence_space::local_space);
                         fwd_edge<WG>(it, x, lw, edge, y, d);
                       });
      })
      .wait();
}

template <int WG>
void launch_bwd(sycl::queue& q, Dim d, const double* x, const double* w, const int64_t* edge, const double* gy,
                double* gx, double* gw) {
  q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> slm(sycl::range<1>(static_cast<size_t>(WG)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d.E * WG)), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) {
                         const int64_t e = static_cast<int64_t>(it.get_group(0));
                         const int tid = static_cast<int>(it.get_local_id(0));
                         const int64_t center = edge[d.E + e] - d.offset;
                         const double* xe = x + (e * d.M) * d.C;
                         const double* we = w + (e * d.L) * d.M;
                         const double* yn = gy + (center * d.L) * d.C;
                         double* gxe = gx + (e * d.M) * d.C;
                         double* gwe = gw + (e * d.L) * d.M;
                         double* red = &slm[0];
                         for (int64_t m = 0; m < d.M; ++m) {
                           for (int64_t c = tid; c < d.C; c += WG) {
                             double acc = 0.0;
                             for (int64_t l = 0; l < d.L; ++l) acc += we[l * d.M + m] * yn[l * d.C + c];
                             gxe[m * d.C + c] = acc;
                           }
                         }
                         for (int64_t l = 0; l < d.L; ++l) {
                           for (int64_t m = 0; m < d.M; ++m) {
                             double partial = 0.0;
                             for (int64_t c = tid; c < d.C; c += WG) {
                               partial += yn[l * d.C + c] * xe[m * d.C + c];
                             }
                             red[tid] = partial;
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int offset = WG / 2; offset > 0; offset >>= 1) {
                               if (tid < offset) red[tid] += red[tid + offset];
                               it.barrier(sycl::access::fence_space::local_space);
                             }
                             if (tid == 0) gwe[l * d.M + m] = red[0];
                             it.barrier(sycl::access::fence_space::local_space);
                           }
                         }
                       });
      })
      .wait();
}

void scatter_forward(const at::Tensor& x, const at::Tensor& w, const at::Tensor& edge, at::Tensor& y,
                     int64_t n_nodes, int64_t offset, int64_t variant) {
  const Dim d = check_dims(x, w, edge, n_nodes, offset);
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto ec = edge.contiguous();
  y.zero_();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  const double* xp = xc.data_ptr<double>();
  const double* wp = wc.data_ptr<double>();
  const int64_t* ep = ec.data_ptr<int64_t>();
  double* yp = y.data_ptr<double>();
  switch (variant) {
    case 0: launch_fwd<128>(q, d, xp, wp, ep, yp); break;
    case 1: launch_fwd<64>(q, d, xp, wp, ep, yp); break;
    case 2: launch_fwd<32>(q, d, xp, wp, ep, yp); break;
    case 3: launch_fwd_sg<128, 16>(q, d, xp, wp, ep, yp); break;
    case 4: launch_fwd_sg<128, 32>(q, d, xp, wp, ep, yp); break;
    case 5: launch_fwd_local<128>(q, d, xp, wp, ep, yp); break;
    case 6: launch_fwd_local<64>(q, d, xp, wp, ep, yp); break;
    case 7: launch_fwd_local<32>(q, d, xp, wp, ep, yp); break;
    default: TORCH_CHECK(false, "f5: unknown variant ", variant);
  }
}

void scatter_backward(const at::Tensor& x, const at::Tensor& w, const at::Tensor& edge, const at::Tensor& gy,
                      at::Tensor& gx, at::Tensor& gw, int64_t n_nodes, int64_t offset, int64_t variant) {
  const Dim d = check_dims(x, w, edge, n_nodes, offset);
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto ec = edge.contiguous();
  auto gyc = gy.contiguous();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  const int which = (variant == 3 || variant == 4) ? 0 : (variant >= 5 ? variant - 5 : variant);
  switch (which) {
    case 0: launch_bwd<128>(q, d, xc.data_ptr<double>(), wc.data_ptr<double>(), ec.data_ptr<int64_t>(), gyc.data_ptr<double>(), gx.data_ptr<double>(), gw.data_ptr<double>()); break;
    case 1: launch_bwd<64>(q, d, xc.data_ptr<double>(), wc.data_ptr<double>(), ec.data_ptr<int64_t>(), gyc.data_ptr<double>(), gx.data_ptr<double>(), gw.data_ptr<double>()); break;
    case 2: launch_bwd<32>(q, d, xc.data_ptr<double>(), wc.data_ptr<double>(), ec.data_ptr<int64_t>(), gyc.data_ptr<double>(), gx.data_ptr<double>(), gw.data_ptr<double>()); break;
    default: TORCH_CHECK(false, "f5: unknown variant ", variant);
  }
}

at::Tensor aten_scatter(const at::Tensor& x, const at::Tensor& w, const at::Tensor& edge, int64_t n_nodes,
                        int64_t offset) {
  auto rotated = at::bmm(w, x);
  auto y = torch::zeros({n_nodes, w.size(1), x.size(2)}, x.options());
  y.index_add_(0, edge[1] - offset, rotated);
  return y;
}

class ScatterAutograd : public torch::autograd::Function<ScatterAutograd> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x, torch::Tensor w,
                               torch::Tensor edge, int64_t n_nodes, int64_t offset, int64_t variant) {
    ctx->save_for_backward({x, w, edge});
    ctx->saved_data["n"] = n_nodes;
    ctx->saved_data["off"] = offset;
    ctx->saved_data["variant"] = variant;
    auto y = torch::empty({n_nodes, w.size(1), x.size(2)}, x.options());
    scatter_forward(x, w, edge, y, n_nodes, offset, variant);
    return y;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    const int64_t n_nodes = ctx->saved_data["n"].toInt();
    const int64_t offset = ctx->saved_data["off"].toInt();
    const int64_t variant = ctx->saved_data["variant"].toInt();
    auto gx = torch::empty_like(saved[0]);
    auto gw = torch::empty_like(saved[1]);
    scatter_backward(saved[0], saved[1], saved[2], grad_outputs[0], gx, gw, n_nodes, offset, variant);
    return {gx, gw, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor scatter_autograd(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& edge,
                               int64_t n_nodes, int64_t offset, int64_t variant) {
  return ScatterAutograd::apply(x, w, edge, n_nodes, offset, variant);
}

std::tuple<at::Tensor, at::Tensor> scatter_scaffold(const torch::Tensor& x, const torch::Tensor& w,
                                                    const torch::Tensor& edge, int64_t n_nodes, int64_t offset) {
  auto xv = x.detach().requires_grad_(true);
  auto wv = w.detach().requires_grad_(true);
  auto y = aten_scatter(xv, wv, edge, n_nodes, offset);
  torch::autograd::variable_list outs;
  outs.push_back(y.sum());
  torch::autograd::variable_list ins;
  ins.push_back(xv);
  ins.push_back(wv);
  auto grads = torch::autograd::grad(outs, ins);
  return {grads[0], grads[1]};
}

}  // namespace

extern "C" void uma_n5f5_anchor() {}

TORCH_LIBRARY(uma_n5f5, m) {
  m.def("scatter(Tensor x, Tensor wigner_inv, Tensor edge_index, int n_nodes, int node_offset, int variant) -> Tensor");
  m.def("scatter_scaffold(Tensor x, Tensor wigner_inv, Tensor edge_index, int n_nodes, int node_offset) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(uma_n5f5, XPU, m) {
  m.impl("scatter", [](const at::Tensor& x, const at::Tensor& w, const at::Tensor& edge, int64_t n_nodes,
                       int64_t offset, int64_t variant) {
    auto y = torch::empty({n_nodes, w.size(1), x.size(2)}, x.options());
    scatter_forward(x, w, edge, y, n_nodes, offset, variant);
    return y;
  });
}

TORCH_LIBRARY_IMPL(uma_n5f5, Autograd, m) { m.impl("scatter", &scatter_autograd); }

TORCH_LIBRARY_IMPL(uma_n5f5, CompositeExplicitAutograd, m) {
  m.impl("scatter_scaffold", &scatter_scaffold);
}
