#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "uma/metadata.h"

namespace uma {

class GraphParallelRuntime;

struct Prediction {
  double energy = 0.0;   // physical energy (FP32 or FP64 model → host double)
  /// Forces [N,3] float64; typically on the engine device (traced) or CPU (GP).
  torch::Tensor forces;
  /// P0'.1 step 2: global virial tensor W = -dE/dstrain in LAMMPS Voigt order
  /// {xx, yy, zz, xy, xz, yz} (eV). Computed only when requested (see
  /// Predictor::predict_body's want_virial). has_virial=false leaves it unset and
  /// the energy/force path byte-identical to the no-virial case.
  bool has_virial = false;
  double virial[6] = {0, 0, 0, 0, 0, 0};
};

/// GPU-persistent UMA inference: energy + forces via autograd on energy.
///
/// Compute dtype (positions, cell, energy) is set from artifact metadata and
/// may be overridden via set_compute_dtype() for LAMMPS `precision mixed|double`.
/// Forces are always FP64.
///
/// ``num_devices == 1``: TorchScript ``model_traced.pt`` on a single device.
/// ``num_devices > 1``: ``GraphParallelRuntime`` → default C++ LibTorch MP
/// (``model_mp_wN_r*.pt`` + ``uma_peer`` + vesin). Opt-in Python with
/// ``UMA_PYTHON_GP_WORKER=1``; legacy Ray with ``UMA_ALLOW_RAY_GP=1``.
class Predictor {
 public:
  static Predictor from_artifact(const std::string& artifact_dir,
                                 torch::Device device = torch::kCUDA,
                                 int num_devices = 1);

  ~Predictor();
  Predictor(Predictor&&) noexcept;
  Predictor& operator=(Predictor&&) noexcept;
  Predictor(const Predictor&) = delete;
  Predictor& operator=(const Predictor&) = delete;

  /// P0'.1 step 2: request the global virial (stress) via strain autograd on the
  /// next predict(). Off by default so the energy/force path is byte-identical.
  void set_want_virial(bool on) { want_virial_ = on; }
  bool want_virial() const { return want_virial_; }

  Prediction predict(const torch::Tensor& pos,             // [N,3]
                     const torch::Tensor& atomic_numbers,  // [N] int64
                     const torch::Tensor& cell,            // [3,3]
                     const torch::Tensor& pbc,             // [3] bool
                     int64_t charge = 0, int64_t spin = 0);

  /// Like predict() but with a caller-supplied neighbor graph (no rebuild).
  /// edge_index [2,E] int64 (row0=neighbor, row1=center); cell_offsets [E,3].
  Prediction predict_extgraph(const torch::Tensor& pos,
                              const torch::Tensor& atomic_numbers,
                              const torch::Tensor& cell, const torch::Tensor& pbc,
                              const torch::Tensor& edge_index,
                              const torch::Tensor& cell_offsets,
                              int64_t charge = 0, int64_t spin = 0);

  /// Host FP32 positions (downcasts if artifact is FP64).
  Prediction predict_host(int n, const float* pos_xyz, const int* atomic_numbers,
                          const double* cell_3x3, const int* pbc_3,
                          double* forces_out_optional = nullptr);

  /// Host FP64 positions (preferred for float64 / double precision).
  Prediction predict_host(int n, const double* pos_xyz, const int* atomic_numbers,
                          const double* cell_3x3, const int* pbc_3,
                          double* forces_out_optional = nullptr);

  /// Host FP64 positions with a CALLER-SUPPLIED neighbor graph (external graph).
  ///
  /// Identical to predict_host(double) EXCEPT the engine does NOT rebuild the
  /// neighbor list: it uses the supplied edge_index + cell_offsets directly.
  /// Use this from LAMMPS pair_uma to avoid the O(N^2) internal build_neighbor_graph
  /// (which hangs at large N).
  ///
  /// Edge convention (MUST match rebuild_neighbors() output, predictor.cpp:197):
  ///   edge_index_2E is int64 row-major [2, n_edges]:
  ///     row 0 (edge_index_2E[0*n_edges + e]) = neighbor atom j
  ///     row 1 (edge_index_2E[1*n_edges + e]) = center   atom i
  /// cell_offsets_E3 is float64 row-major [n_edges, 3]: INTEGER periodic shifts
  ///   stored as double; edge_distance_vec = pos[j] + offset @ cell - pos[i].
  ///
  /// NOTE: positions are NOT wrapped into the cell in this path (unlike
  /// predict_host). The supplied offsets already encode periodicity and the
  /// edge_distance_vec is translation-invariant, so wrapping is unnecessary and
  /// would be inconsistent with offsets computed against unwrapped LAMMPS coords.
  Prediction predict_host_extgraph(int n, const double* pos_xyz,
                                   const int* atomic_numbers, const double* cell9,
                                   const int* pbc3, int64_t n_edges,
                                   const int64_t* edge_index_2E,
                                   const double* cell_offsets_E3,
                                   double* forces_out);

  /// DD k=4 external-graph path. The DD artifact's top module returns
  /// (node_energy[n], total). Backprop from E_owned = sum(node_energy[0:nlocal])
  /// (NOT the whole subsystem: ghost-energy terms would inject spurious force
  /// gradients), so owned-atom forces are correct once the halo backward routes
  /// ghost grads to owners. Writes per-node energy into energy_out[n] (owned rows
  /// summed by the caller into eng_vdwl) and forces into forces_out[n*3].
  /// Returns Prediction.energy = sum(node_energy[0:nlocal]) (this rank's owned
  /// energy contribution).
  Prediction predict_host_extgraph_dd(int n, int nlocal, const double* pos_xyz,
                                      const int* atomic_numbers,
                                      const double* cell9, const int* pbc3,
                                      int64_t n_edges,
                                      const int64_t* edge_index_2E,
                                      const double* cell_offsets_E3,
                                      double* energy_out, double* forces_out);

  /// Override position/energy compute dtype (must match TorchScript artifact dtype).
  void set_compute_dtype(torch::ScalarType dtype);

  const ArtifactMetadata& metadata() const { return metadata_; }
  torch::Device device() const { return device_; }
  torch::ScalarType compute_dtype() const { return compute_dtype_; }
  int num_devices() const { return num_devices_; }
  bool uses_graph_parallel() const { return static_cast<bool>(gp_); }

 private:
  Predictor(torch::jit::script::Module module, torch::Device device,
            ArtifactMetadata metadata, int num_devices);
  Predictor(std::unique_ptr<GraphParallelRuntime> gp, torch::Device device,
            ArtifactMetadata metadata, int num_devices);

  void ensure_buffers(int64_t n);
  void rebuild_neighbors();

  // Shared post-neighbor forward/backward body used by both predict() (which
  // rebuilds the neighbor list first) and predict_extgraph() (which sets
  // edge_index_/cell_offsets_ from the caller first). Assumes pos_,
  // atomic_numbers_, cell_, pbc_, charge_, spin_, edge_index_, cell_offsets_
  // are already populated. Does NOT rebuild neighbors and does NOT wrap positions.
  Prediction predict_body();

  // DD k=4 body: reads the tuple (node_energy, total) from the DD artifact,
  // applies per-atom denorm/refs, backprops from sum(node_energy[0:nlocal]), and
  // writes per-node physical energy to energy_out (if non-null).
  Prediction predict_body_dd(int nlocal, double* energy_out);

  // Common input staging shared by predict() and predict_extgraph(): fills the
  // persistent pos_/atomic_numbers_/cell_/pbc_/charge_/spin_ buffers. Returns N.
  int64_t stage_inputs(const torch::Tensor& pos,
                       const torch::Tensor& atomic_numbers,
                       const torch::Tensor& cell, const torch::Tensor& pbc,
                       int64_t charge, int64_t spin);

  torch::jit::script::Module module_;
  bool has_traced_module_ = false;
  bool want_virial_ = false;  // P0'.1 step 2
  // P0'.4: true only for the live object that should clear the process-wide
  // BlockContext singleton on destruction. Reset to false when moved-from so the
  // moved-out temporary (e.g. from_artifact's return value) does not wipe the
  // blocks the live object still needs. See predictor.cpp move ops + ~Predictor.
  bool owns_block_context_ = true;
  std::unique_ptr<GraphParallelRuntime> gp_;
  torch::Device device_;
  ArtifactMetadata metadata_;
  torch::ScalarType compute_dtype_ = torch::kFloat32;
  int num_devices_ = 1;

  // Persistent device tensors (reused when N unchanged). Traced path only.
  int64_t n_cached_ = -1;
  torch::Tensor pos_;              // compute_dtype [N,3]
  torch::Tensor atomic_numbers_;   // int64 [N]
  torch::Tensor cell_;             // compute_dtype [3,3]
  torch::Tensor pbc_;              // bool [3]
  torch::Tensor edge_index_;       // int64 [2,E]
  torch::Tensor cell_offsets_;     // compute_dtype [E,3]
  torch::Tensor charge_;
  torch::Tensor spin_;
  torch::Tensor batch_;
  torch::Tensor element_refs_;     // compute_dtype on device
};

}  // namespace uma
