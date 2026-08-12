#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <torch/script.h>
#include <torch/torch.h>

#include "uma/metadata.h"
#include "uma/predictor.h"

namespace uma {

/// In-process edge-parallel UMA peer for MULTI-NODE (one per MPI rank).
///
/// Unlike LibtorchMpRuntime (which forks W local workers and moves geometry
/// through /dev/shm -- single node only), this class IS one peer: the calling
/// MPI rank. It loads this rank's shard (model_mp_w{W}_n{N}_r{R}.pt), builds the
/// full-system neighbor graph, evaluates its 1/W edge shard, and force-all-reduces
/// across all W GPUs via NCCL bootstrapped over MPI (init_nccl_external).
///
/// Memory is ~O(N/W): the point of the multi-node path. Every rank passes the
/// SAME tag-ordered global system in; forces come back fully reduced on rank 0
/// (and, since all_reduce is symmetric, on every rank).
class MpiPeerPredictor {
 public:
  /// Collective across all W ranks (each calls ncclCommInitRank). The NCCL
  /// unique id must have been generated on rank 0 and MPI_Bcast to every rank
  /// BEFORE calling this (nccl_unique_id points at unique_id_bytes() bytes).
  static std::unique_ptr<MpiPeerPredictor> create(
      const std::string& artifact_dir, const ArtifactMetadata& metadata,
      int world, int rank, int device_index, const void* nccl_unique_id,
      torch::ScalarType compute_dtype);

  /// Size (bytes) of the opaque NCCL id the caller must broadcast.
  static size_t nccl_unique_id_bytes();
  /// Rank 0 fills a unique id for MPI_Bcast (buffer >= nccl_unique_id_bytes()).
  static void make_nccl_unique_id(void* out);

  ~MpiPeerPredictor();
  MpiPeerPredictor(const MpiPeerPredictor&) = delete;
  MpiPeerPredictor& operator=(const MpiPeerPredictor&) = delete;

  /// Full global system in (n atoms, tag-ordered identically on every rank).
  /// forces_out (optional) receives the fully reduced [n,3] FP64 forces.
  Prediction predict_host(int n, const double* pos_xyz, const int* atomic_numbers,
                          const double* cell_3x3, const int* pbc_3,
                          double* forces_out_optional = nullptr);

  int world() const { return world_; }
  int rank() const { return rank_; }

 private:
  struct Impl;
  MpiPeerPredictor();

  int world_ = 1;
  int rank_ = 0;
  int device_index_ = 0;
  ArtifactMetadata metadata_;
  torch::ScalarType compute_dtype_ = torch::kFloat64;
  std::unique_ptr<Impl> impl_;
};

}  // namespace uma
