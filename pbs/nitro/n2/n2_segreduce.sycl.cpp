// Step 18 (N2-b). Segmented sum of center-sorted edge messages.
// One work-group per node. Backward is a gather: grad_src[e] = grad_out[center[e]].

#include <torch/torch.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

namespace {

constexpr int kWg = 128;

void segreduce_forward_xpu(const at::Tensor& src, const at::Tensor& offsets, at::Tensor& out) {
  const auto e = src.size(0);
  const auto n = out.size(0);
  const auto f = src.size(1);
  auto* sp = src.data_ptr<double>();
  auto* op = offsets.data_ptr<int64_t>();
  auto* yp = out.data_ptr<double>();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  q.parallel_for<class N2SegFwd>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(n * kWg)), sycl::range<1>(kWg)),
        [=](sycl::nd_item<1> it) {
          const int64_t node = static_cast<int64_t>(it.get_group(0));
          const int tid = static_cast<int>(it.get_local_id(0));
          const int64_t begin = op[node];
          const int64_t end = op[node + 1];
          for (int64_t feat = tid; feat < f; feat += kWg) {
            double acc = 0.0;
            for (int64_t edge = begin; edge < end; ++edge) {
              acc += sp[edge * f + feat];
            }
            yp[node * f + feat] = acc;
          }
        })
      .wait();
  (void)e;
}

void segreduce_backward_xpu(const at::Tensor& gy, const at::Tensor& centers, at::Tensor& gx) {
  const auto e = gx.size(0);
  const auto f = gx.size(1);
  auto* gp = gy.data_ptr<double>();
  auto* cp = centers.data_ptr<int64_t>();
  auto* xp = gx.data_ptr<double>();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  const int64_t nwork = e * kWg;
  q.parallel_for<class N2SegBwd>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(nwork)), sycl::range<1>(kWg)),
        [=](sycl::nd_item<1> it) {
          const int64_t edge = static_cast<int64_t>(it.get_group(0));
          const int tid = static_cast<int>(it.get_local_id(0));
          const int64_t node = cp[edge];
          for (int64_t feat = tid; feat < f; feat += kWg) {
            xp[edge * f + feat] = gp[node * f + feat];
          }
        })
      .wait();
}

class SegreduceAutograd : public torch::autograd::Function<SegreduceAutograd> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor src,
                               torch::Tensor centers, torch::Tensor offsets) {
    ctx->save_for_backward({centers});
    auto out = torch::empty({offsets.size(0) - 1, src.size(1)}, src.options());
    segreduce_forward_xpu(src, offsets, out);
    return out;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto gx = torch::empty({saved[0].size(0), grad_outputs[0].size(1)}, grad_outputs[0].options());
    segreduce_backward_xpu(grad_outputs[0], saved[0], gx);
    return {gx, torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor segreduce_autograd(const torch::Tensor& src, const torch::Tensor& centers,
                                 const torch::Tensor& offsets) {
  return SegreduceAutograd::apply(src, centers, offsets);
}

}  // namespace

TORCH_LIBRARY(uma_n2, m) {
  m.def("segreduce(Tensor src, Tensor centers, Tensor offsets) -> Tensor");
}

TORCH_LIBRARY_IMPL(uma_n2, XPU, m) {
  m.impl("segreduce", [](const at::Tensor& src, const at::Tensor& centers, const at::Tensor& offsets) {
    auto out = torch::empty({offsets.size(0) - 1, src.size(1)}, src.options());
    segreduce_forward_xpu(src, offsets, out);
    return out;
  });
}

TORCH_LIBRARY_IMPL(uma_n2, Autograd, m) {
  m.impl("segreduce", &segreduce_autograd);
}
