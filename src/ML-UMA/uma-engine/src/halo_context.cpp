#include "uma/halo_context.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <torch/torch.h>

namespace uma {

HaloContext& HaloContext::instance() {
  static HaloContext ctx;
  return ctx;
}

void HaloContext::set_callbacks(ExchangeFn forward_fn, ExchangeFn reverse_fn,
                                int64_t nlocal, int64_t nall) {
  std::lock_guard<std::mutex> lk(mu_);
  forward_fn_ = std::move(forward_fn);
  reverse_fn_ = std::move(reverse_fn);
  nlocal_ = nlocal;
  nall_ = nall;
  active_ = (forward_fn_ && reverse_fn_ && nall_ > 0);
}

void HaloContext::clear() {
  std::lock_guard<std::mutex> lk(mu_);
  forward_fn_ = nullptr;
  reverse_fn_ = nullptr;
  nlocal_ = 0;
  nall_ = 0;
  active_ = false;
}

bool HaloContext::active() const {
  std::lock_guard<std::mutex> lk(mu_);
  return active_;
}
int64_t HaloContext::nlocal() const {
  std::lock_guard<std::mutex> lk(mu_);
  return nlocal_;
}
int64_t HaloContext::nall() const {
  std::lock_guard<std::mutex> lk(mu_);
  return nall_;
}

namespace {

// Flatten [nall, ...] -> contiguous host double buffer [nall, per_node], run the
// callback, reshape back. FP64 throughout (DD path is precision double).
torch::Tensor run_exchange(const torch::Tensor& x,
                           const HaloContext::ExchangeFn& fn, int64_t nall) {
  if (!fn) throw std::runtime_error("HaloContext: exchange callback not set");
  if (x.size(0) != nall)
    throw std::runtime_error(
        "HaloContext: tensor row count != nall (node ordering mismatch)");
  const auto orig_sizes = x.sizes().vec();
  const auto orig_dtype = x.scalar_type();
  const auto orig_device = x.device();

  // Contiguous [nall, per_node] on CPU in FP64.
  auto x2d = x.reshape({nall, -1}).to(torch::kCPU, torch::kFloat64).contiguous();
  const int64_t per_node = x2d.size(1);

  // Diagnostic (UMA_DD_DEBUG): relative L2 norm of the change the exchange makes
  // to ALL ghost rows [nlocal_guess, nall). Only rank 0's first few calls print.
  // A well-working k=4 exchange should make a NONtrivial correction each layer;
  // a near-zero change means the exchange is a no-op; a huge change may indicate
  // it is clobbering good data. HaloContext doesn't know nlocal here, so estimate
  // ghost region from the callback's own knowledge via a static hook is overkill;
  // instead norm over the LAST 40% of rows (ghost-heavy) as a proxy.
  static int dbg_calls = 0;
  const bool dbg = (std::getenv("UMA_DD_DEBUG") != nullptr) && (dbg_calls < 8);
  std::vector<double> snap;
  const int64_t r0 = static_cast<int64_t>(nall * 0.6);   // proxy ghost region start
  if (dbg) {
    double* p = x2d.data_ptr<double>();
    snap.assign(p + r0 * per_node, p + nall * per_node);
  }

  fn(x2d.data_ptr<double>(), nall, per_node);

  if (dbg) {
    double* p = x2d.data_ptr<double>();
    double dn = 0.0, bn = 0.0;
    for (int64_t idx = 0; idx < static_cast<int64_t>(snap.size()); ++idx) {
      const double b = snap[idx];
      const double a = p[r0 * per_node + idx];
      dn += (a - b) * (a - b);
      bn += b * b;
    }
    std::fprintf(stderr,
                 "[halo dbg call %d] ghost-region ||delta||/||x|| = %.4e "
                 "(||x||=%.3e)\n",
                 dbg_calls, (bn > 0 ? std::sqrt(dn / bn) : std::sqrt(dn)),
                 std::sqrt(bn));
    dbg_calls++;
  }

  return x2d.reshape(orig_sizes).to(orig_device, orig_dtype);
}

}  // namespace

torch::Tensor HaloContext::forward_exchange(const torch::Tensor& x) {
  ExchangeFn fn;
  int64_t nall;
  {
    std::lock_guard<std::mutex> lk(mu_);
    fn = forward_fn_;
    nall = nall_;
  }
  return run_exchange(x, fn, nall);
}

torch::Tensor HaloContext::reverse_exchange(const torch::Tensor& grad) {
  ExchangeFn fn;
  int64_t nall;
  {
    std::lock_guard<std::mutex> lk(mu_);
    fn = reverse_fn_;
    nall = nall_;
  }
  return run_exchange(grad, fn, nall);
}

// Forward kernel (no autograd node).
torch::Tensor uma_halo_op_exchange(const torch::Tensor& x) {
  auto& ctx = HaloContext::instance();
  if (!ctx.active()) {
    // Single-rank / non-DD: identity (no ghosts to refresh).
    return x;
  }
  // Diagnostic A/B: UMA_DD_NO_HALO=1 makes the op identity at runtime (ghosts
  // stay frozen at their block outputs). If parity is UNCHANGED vs the real
  // exchange, the exchange is a no-op (bug); if WORSE, the exchange is working.
  static const bool no_halo = [] {
    const char* e = std::getenv("UMA_DD_NO_HALO");
    return e && e[0] == '1' && e[1] == '\0';
  }();
  if (no_halo) return x;
  return ctx.forward_exchange(x);
}

// Autograd: forward scatters owned->ghost; backward accumulates ghost->owner.
// The reverse callback ADDS ghost-row grads onto owner rows and zeros ghosts, so
// after reverse_exchange the owned rows carry (local grad + remote ghost grads),
// which is exactly d(anything downstream)/d(owned feature). Ghost rows are zeroed
// because their gradient has been delivered to the owner (avoids double count in
// the next upstream op, which will re-scatter from owned).
class HaloExchangeFn : public torch::autograd::Function<HaloExchangeFn> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* /*ctx*/,
                               const torch::Tensor& x) {
    at::AutoDispatchBelowADInplaceOrView guard;
    return uma_halo_op_exchange(x);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* /*ctx*/,
      torch::autograd::variable_list grad_outputs) {
    auto& hctx = HaloContext::instance();
    if (!hctx.active()) return {grad_outputs[0]};
    at::AutoDispatchBelowADInplaceOrView guard;
    auto g = grad_outputs[0].contiguous();
    return {hctx.reverse_exchange(g)};
  }
};

torch::Tensor halo_exchange_autograd(const torch::Tensor& x) {
  return HaloExchangeFn::apply(x);
}

}  // namespace uma

TORCH_LIBRARY(uma_halo, m) {
  m.def("exchange(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(uma_halo, CompositeExplicitAutograd, m) {
  m.impl("exchange", TORCH_FN(uma::uma_halo_op_exchange));
}

TORCH_LIBRARY_IMPL(uma_halo, Autograd, m) {
  m.impl("exchange", TORCH_FN(uma::halo_exchange_autograd));
}
