#pragma once

// Multi-node spatial domain-decomposition (DD) per-layer halo exchange (k=4).
//
// The traced UMA top module, when exported for DD, calls a custom op ONCE before
// each of the num_layers message-passing blocks:
//   uma_halo::exchange(Tensor x) -> Tensor x     # x: [nall, F, C]
// where the node axis is the LAMMPS owned+ghost order for THIS rank:
//   [0, nlocal)      owned atoms
//   [nlocal, nall)   ghost atoms (copies of atoms owned by other ranks)
//
// forward:  overwrite each ghost row with the current feature of the atom it is
//           an image of, fetched from that atom's OWNER rank (owned rows kept).
// backward: each ghost row's grad belongs to its OWNER; send ghost-row grads
//           back and ACCUMULATE onto the owner's owned row (adjoint of the
//           forward owned->ghost scatter). Owned-row grads pass through.
//
// This is the spatial analogue of uma_peer::all_gather_nodes (peer_context.*).
// The actual data movement is delegated to LAMMPS' own ghost communication via a
// callback the pair style installs (Option T1: comm->forward_comm /
// reverse_comm), so the DD op reuses LAMMPS' proven, load-balanced comm plan
// instead of reimplementing a ghost map in the engine.

#include <cstdint>
#include <functional>
#include <mutex>

#include <torch/torch.h>

namespace uma {

// Callbacks the pair style installs. Each operates on a contiguous host buffer
// of per-node feature rows in LAMMPS owned+ghost order.
//
//   forward_fn(buf, nall, per_node):  buf holds [nall, per_node] doubles with
//     owned rows [0,nlocal) valid on entry; fills ghost rows [nlocal,nall) from
//     their owners (LAMMPS forward_comm).
//   reverse_fn(buf, nall, per_node):  buf holds [nall, per_node] grad rows;
//     ADDS each ghost row's grad onto its owner's owned row and zeros the ghost
//     row (LAMMPS reverse_comm). Owned rows then hold owned + accumulated remote.
class HaloContext {
 public:
  static HaloContext& instance();

  using ExchangeFn =
      std::function<void(double* buf, int64_t nall, int64_t per_node)>;

  void set_callbacks(ExchangeFn forward_fn, ExchangeFn reverse_fn,
                     int64_t nlocal, int64_t nall);
  void clear();

  bool active() const;
  int64_t nlocal() const;
  int64_t nall() const;

  // Run the installed forward/reverse exchange on a [nall, F, C] (or [nall, K])
  // tensor. Stages to a contiguous host double buffer, calls the LAMMPS
  // callback, and returns a new tensor with ghost rows filled (forward) or
  // ghost grads accumulated onto owners (reverse).
  torch::Tensor forward_exchange(const torch::Tensor& x);
  torch::Tensor reverse_exchange(const torch::Tensor& grad);

 private:
  HaloContext() = default;
  mutable std::mutex mu_;
  ExchangeFn forward_fn_;
  ExchangeFn reverse_fn_;
  int64_t nlocal_ = 0;
  int64_t nall_ = 0;
  bool active_ = false;
};

// Autograd-aware wrapper (schema uma_halo::exchange(Tensor) -> Tensor).
torch::Tensor halo_exchange_autograd(const torch::Tensor& x);

}  // namespace uma
