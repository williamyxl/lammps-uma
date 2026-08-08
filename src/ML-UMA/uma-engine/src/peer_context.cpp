#include "uma/peer_context.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <torch/torch.h>

namespace uma {

PeerContext& PeerContext::instance() {
  static PeerContext ctx;
  return ctx;
}

void PeerContext::reset_shared(kokkos_peer::SharedPeerGatherSlot* slot) {
  std::lock_guard<std::mutex> lk(mu_);
  slot_ = slot;
  world_ = slot ? slot->world() : 1;
  if (slot) kokkos_peer::enable_peer_access(world_);
}

void PeerContext::clear() {
  std::lock_guard<std::mutex> lk(mu_);
  slot_ = nullptr;
  world_ = 1;
}

int PeerContext::world() const { return world_; }

kokkos_peer::SharedPeerGatherSlot& PeerContext::slot() {
  if (!slot_) {
    throw std::runtime_error("PeerContext: shared slot not set");
  }
  return *slot_;
}

void PeerContext::set_thread_rank(int rank) {
  instance().process_rank_ = rank;
}
int PeerContext::thread_rank() { return instance().process_rank_; }

void register_uma_peer_ops() {}

int64_t uma_peer_op_rank() {
  return static_cast<int64_t>(PeerContext::thread_rank());
}

int64_t uma_peer_op_world() {
  return static_cast<int64_t>(PeerContext::instance().world());
}

// Forward kernels (no Autograd node). Called under AutoDispatchBelowADInplaceOrView.
torch::Tensor uma_peer_op_all_gather_nodes(const torch::Tensor& local,
                                           int64_t n_atoms) {
  auto& ctx = PeerContext::instance();
  const int world = ctx.world();
  const int rank = PeerContext::thread_rank();
  const int64_t pad = kokkos_peer::padded_local_size(n_atoms, world);
  auto padded = kokkos_peer::pad_nodes(local.contiguous(), pad);
  return ctx.slot().all_gather_concat(rank, padded, n_atoms);
}

torch::Tensor uma_peer_op_all_reduce_sum(const torch::Tensor& local) {
  return PeerContext::instance().slot().all_reduce(
      PeerContext::thread_rank(), local.contiguous());
}

// Process-per-rank default: all_reduce grads in backward (nn.functional.all_reduce).
// Pair with force all_reduce + grad energy scale 1/world (see mp worker).
// UMA_ALLREDUCE_WITH_GRAD_BWD=0: identity backward (FairChem ReduceFromMP / diag).
class AllReduceSumFn : public torch::autograd::Function<AllReduceSumFn> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* /*ctx*/,
                               const torch::Tensor& local) {
    at::AutoDispatchBelowADInplaceOrView guard;
    return uma_peer_op_all_reduce_sum(local);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* /*ctx*/,
      torch::autograd::variable_list grad_outputs) {
    const char* mode = std::getenv("UMA_ALLREDUCE_WITH_GRAD_BWD");
    // Default ON for process-per-rank (sweep 20925383: needed for F green).
    const bool ar_bwd = !(mode && std::string(mode) == "0");
    if (ar_bwd) {
      at::AutoDispatchBelowADInplaceOrView guard;
      return {uma_peer_op_all_reduce_sum(grad_outputs[0].contiguous())};
    }
    return {grad_outputs[0]};
  }
};

// FairChem gather sum_grad: all_reduce(grad_full) then local slice (padded).
class AllGatherNodesFn : public torch::autograd::Function<AllGatherNodesFn> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const torch::Tensor& local, int64_t n_atoms) {
    ctx->saved_data["n_atoms"] = n_atoms;
    ctx->saved_data["in0"] = static_cast<int64_t>(local.size(0));
    at::AutoDispatchBelowADInplaceOrView guard;
    return uma_peer_op_all_gather_nodes(local, n_atoms);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const int64_t n_atoms = ctx->saved_data["n_atoms"].toInt();
    const int64_t in0 = ctx->saved_data["in0"].toInt();
    auto g = grad_outputs[0].contiguous();
    at::AutoDispatchBelowADInplaceOrView guard;
    // Each rank holds a full [n_atoms, ...] grad; sum across ranks.
    auto summed = uma_peer_op_all_reduce_sum(g);
    const int world = PeerContext::instance().world();
    const int rank = PeerContext::thread_rank();
    auto sizes = kokkos_peer::size_list(n_atoms, world);
    int64_t start = 0;
    for (int r = 0; r < rank; ++r) start += sizes[static_cast<size_t>(r)];
    const int64_t nloc = sizes[static_cast<size_t>(rank)];

    std::vector<int64_t> shape = summed.sizes().vec();
    if (shape.empty()) {
      shape = {in0};
    } else {
      shape[0] = in0;
    }
    auto local_grad = torch::zeros(shape, summed.options());
    if (nloc > 0) {
      local_grad.narrow(0, 0, nloc).copy_(summed.narrow(0, start, nloc));
    }
    // Second slot is for non-differentiable int n_atoms.
    return {local_grad, torch::Tensor()};
  }
};

torch::Tensor all_gather_nodes_autograd(const torch::Tensor& local,
                                        int64_t n_atoms) {
  return AllGatherNodesFn::apply(local, n_atoms);
}

torch::Tensor all_reduce_sum_autograd(const torch::Tensor& local) {
  return AllReduceSumFn::apply(local);
}

}  // namespace uma

TORCH_LIBRARY(uma_peer, m) {
  m.def("rank() -> int");
  m.def("world() -> int");
  m.def("all_gather_nodes(Tensor local, int n_atoms) -> Tensor");
  m.def("all_reduce_sum(Tensor local) -> Tensor");
}

TORCH_LIBRARY_IMPL(uma_peer, CompositeExplicitAutograd, m) {
  m.impl("rank", TORCH_FN(uma::uma_peer_op_rank));
  m.impl("world", TORCH_FN(uma::uma_peer_op_world));
  m.impl("all_gather_nodes", TORCH_FN(uma::uma_peer_op_all_gather_nodes));
  m.impl("all_reduce_sum", TORCH_FN(uma::uma_peer_op_all_reduce_sum));
}

// Autograd key: wrap collectives so TorchScript mid-forward ops keep grad_fn.
TORCH_LIBRARY_IMPL(uma_peer, Autograd, m) {
  m.impl("all_gather_nodes", TORCH_FN(uma::all_gather_nodes_autograd));
  m.impl("all_reduce_sum", TORCH_FN(uma::all_reduce_sum_autograd));
}
