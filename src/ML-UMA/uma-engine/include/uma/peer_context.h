#pragma once

#include <cstdint>
#include <memory>
#include <mutex>

#include <torch/torch.h>

#include "uma/shared_peer.h"

namespace uma {

class PeerContext {
 public:
  static PeerContext& instance();

  void reset_shared(kokkos_peer::SharedPeerGatherSlot* slot);
  void clear();

  int world() const;
  kokkos_peer::SharedPeerGatherSlot& slot();

  // Process-per-rank: rank is process-global (NOT thread_local). Autograd
  // engine worker threads must see the same rank for mid-backward collectives.
  static void set_thread_rank(int rank);
  static int thread_rank();

 private:
  PeerContext() = default;
  std::mutex mu_;
  int world_ = 1;
  int process_rank_ = 0;
  kokkos_peer::SharedPeerGatherSlot* slot_ = nullptr;  // not owned
};

void register_uma_peer_ops();

// Autograd-aware wrappers (same schemas as torch.ops.uma_peer.*).
torch::Tensor all_gather_nodes_autograd(const torch::Tensor& local,
                                        int64_t n_atoms);
torch::Tensor all_reduce_sum_autograd(const torch::Tensor& local);

}  // namespace uma
