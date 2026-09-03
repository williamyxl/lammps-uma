/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(uma,PairUMA);
// clang-format on
#else

#ifndef LMP_PAIR_UMA_H
#define LMP_PAIR_UMA_H

#include "pair.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace uma {
class Predictor;
class MpiPeerPredictor;
}

namespace LAMMPS_NS {

class PairUMA : public Pair {
 public:
  // Mirrors GPU/INTEL package naming: mixed = FP32 pos/energy + FP64 forces;
  // double = FP64 pos/energy/forces accumulation path for positions+energy.
  enum Precision { PRECISION_MIXED = 0, PRECISION_DOUBLE = 1 };

  PairUMA(class LAMMPS *);
  ~PairUMA() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  void init_list(int, class NeighList *) override;
  double init_one(int, int) override;

 protected:
  virtual void allocate();
  void load_predictor();
  // A5 (audit rev 26 §G.18.6): load_predictor() decomposition. init_mpi_peer()
  // builds the GP-over-MPI peer and returns true when it handled construction
  // (multi-node, non-DD). The single-tile device pick is a file-local helper in
  // pair_uma.cpp (returns torch::Device, kept out of this torch-free header).
  // Pure code-motion out of load_predictor(); no numeric change.
  bool init_mpi_peer();
  // Convert the LAMMPS full neighbor list into FairChem edge format
  // (edge_index [2,E] int64 row0=neighbor row1=center, cell_offsets [E,3]).
  // Single-tile (non-mn) orthorhombic path only. Fills ext_edge_index_ /
  // ext_cell_offsets_ and returns E.
  int64_t build_ext_graph(int nlocal);

  // Per-model compute handlers (E.8.3 #3 decomposition). compute() stages the
  // shared pos/z/cell/pbc member buffers then dispatches to exactly one of these;
  // run_compute_dd() (below) is the third model. Keeping each execution model in
  // its own method keeps compute() a thin dispatcher.
  void run_compute_single_tile(int eflag, int vflag, int nlocal, bool use_f64);
  void run_compute_gp(int eflag, int vflag, int nlocal, bool use_f64);
  void teardown_peer();  // P0'.5: collective-safe GP peer delete (dtor + reload)

  // --- Multi-node spatial domain decomposition (Phase A, deep halo k=1) ------
  // Distinct from the mn_active GP-over-MPI path above. Under DD each rank owns
  // a LAMMPS subdomain; LAMMPS supplies ghosts out to the model receptive field
  // (num_layers * cutoff, set via `comm_modify cutoff`). The engine graph spans
  // owned+ghost atoms as first-class nodes; forces are kept for owned atoms
  // only. Ghosts carry absolute unwrapped coords, so edge_vec = x[j]-x[i]
  // directly -> no cell offsets, no orthorhombic-only restriction.
  // Enabled by env UMA_DD=1. Returns E; fills dd_edge_index_ / dd positions.
  void run_compute_dd(int eflag, int vflag);
  int64_t build_dd_graph(int nall);        // edges among owned+ghost within cutoff
  // A5 (audit rev 26 §G.18.6): the P2.1 edge-padding block factored out of
  // run_compute_dd(). Pads dd_edge_index_/dd_cell_offsets_ from the real edge
  // count E up to UMA_DD_EDGE_CAP with inert atom0->dummy edges; returns the new
  // (padded) edge count. `dummy` is the appended far-node index (nall). Pure
  // code-motion; identical arithmetic to the pre-split inline block.
  int64_t pad_dd_edges(int64_t E, int64_t edge_cap, int dummy);
  // A2/S2 (audit rev 26 §G.18.6): reverse-comm the ghost force rows onto their
  // owners and deposit owned rows into f. Ghost rows hold this rank's
  // -dE_owned/dx_ghost, a real cross-rank force term that used to be discarded
  // (the cos = 0.644 DD bug). See the derivation above the definition.
  void reduce_dd_ghost_forces(int nlocal, int nall, double **f);
  // A11 (audit rev 29 §G.25): pre-flight per-rank memory estimate at init_style.
  // WARN-only (never aborts, never changes numerics); names the flag that fixes
  // an expected OOM. Calibrated to the measured ceilings in the report.
  void preflight_memory_check();
  void mole_composition_allreduce();        // owned-only per-Z counts, cross-rank
  bool mole_composition_done_ = false;      // R4: run the (discarded) diagnostic + warn ONCE, not per DD step
  void install_halo_callbacks();            // bind HaloContext to LAMMPS comm
  bool dd_active_;                           // UMA_DD=1
  int64_t dd_edge_count_;
  std::vector<int64_t> dd_edge_index_;      // [2,E] row0=neighbor row1=center
  std::vector<double> dd_cell_offsets_;     // [E,3] zeros (ghosts are absolute)
  std::vector<double> dd_pos_;              // [nall,3] owned+ghost, boxlo-shifted
  std::vector<int> dd_z_;                   // [nall] atomic numbers owned+ghost
  std::vector<double> dd_force_;            // [nnodes,3] forces for all graph nodes
  std::vector<double> dd_energy_;           // [nnodes] per-node energy (DD k=4)
  // A2/S2 (audit rev 26 §G.18.6): [nall,3] staging buffer for the ghost-force
  // reverse_comm. Ghost rows of dd_force_ hold -dE_owned(this rank)/dx_ghost,
  // which is a REAL contribution to the force on the atom's owner and used to be
  // discarded (the cos = 0.644 bug). reverse_comm accumulates them onto owners.
  std::vector<double> dd_fbuf_;

  // Per-layer halo exchange (k=4) scratch. The engine's uma_halo::exchange op
  // calls back into this pair style; halo_buf_ holds [nall, halo_per_node] node
  // features in owned+ghost order while comm->forward_comm/reverse_comm run the
  // owned<->ghost movement through pack/unpack_{forward,reverse}_comm.
  double *halo_buf_;                         // borrowed view during one exchange
  int64_t halo_per_node_;                    // F*C (comm width) for current op

 public:
  // LAMMPS comm hooks for the halo feature exchange (DD k=4). comm_forward /
  // comm_reverse are set to halo_per_node_ before each exchange.
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:

  std::string artifact_dir;
  int *map;    // map type -> atomic number (0 unused)
  double cutoff;
  int precision;    // PRECISION_MIXED or PRECISION_DOUBLE
  int num_devices;  // graph-parallel GPU count (1 = traced Predictor)
  // Multi-node: one MPI rank per GPU. Each rank evaluates its own LAMMPS
  // subdomain and contributes E/F for the atoms it owns; the global energy is
  // summed over ranks by LAMMPS' own eng_vdwl reduction.
  int mn_world;   // MPI world size when > 1
  int mn_rank;    // this rank
  bool mn_active; // true when running one-rank-per-GPU multi-node
  // Scratch for the multi-node global gather. Tag ordering makes the assembled
  // system identical on every rank, so all ranks build the same graph.
  std::vector<int> mn_tag, mn_tag_all, mn_counts, mn_displs, mn_order;
  std::vector<int> mn_z_all, mn_z_sorted;
  std::vector<double> mn_pos_all, mn_pos_sorted, mn_force_sorted;
  bool devices_explicit;  // true if pair_style set devices N

  std::unique_ptr<uma::Predictor> predictor;  // A3/G.5: was raw owning ptr
  // Multi-node edge-parallel peer (one per MPI rank; memory-sharded model).
  // Non-null only when mn_world > 1. Replaces the O(N)/GPU Scheme-A path.
  std::unique_ptr<uma::MpiPeerPredictor> mpi_peer;  // A3/G.5: was raw owning ptr
  int gpus_per_node;  // for local-rank -> device_index binding
  std::vector<float> pos_buf;
  std::vector<double> pos_buf_d;
  std::vector<int> z_buf;
  std::vector<double> force_buf;
  double cell_buf[9];
  int pbc_buf[3];

  // Neighbor list handed to us by LAMMPS (full list, cutoff = UMA cutoff + skin).
  // Stored from init_list(); consumed in compute() to feed the engine an
  // externally-built edge graph so the engine skips its own O(N^2) rebuild.
  class NeighList *list;
  // Externally-built edge graph scratch (single-tile path).
  std::vector<int64_t> ext_edge_index_;   // [2,E] row0=neighbor(jr) row1=center(i)
  std::vector<double> ext_cell_offsets_;   // [E,3] integer image triples (as double)
  bool engine_build_graph_;                // UMA_ENGINE_BUILD_GRAPH=1 -> old path
  bool want_virial_flag_;                   // UMA_COMPUTE_VIRIAL=1 -> single-tile stress (P0'.1 step 2)
};

}    // namespace LAMMPS_NS

#endif
#endif
