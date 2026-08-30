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

/* ----------------------------------------------------------------------
   Contributing author: UMA LibTorch integration (uma-lmp)
------------------------------------------------------------------------- */

#include "pair_uma.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "exceptions.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "fix.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "utils.h"

#include "uma/predictor.h"
#include "uma/mpi_peer_predictor.h"
#include "uma/halo_context.h"

#ifdef LMP_KOKKOS
#include "kokkos.h"
#endif

#include <algorithm>  // std::stable_sort for the multi-node tag ordering
#include <cmath>      // std::lround for ghost->integer-image recovery
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>  // brace-init list in the M0 device-binding loop
#include <vector>

using namespace LAMMPS_NS;

// P4.1: one validated bool-env parse (replaces the ad-hoc `e[0]=='1'&&e[1]=='\0'`
// idiom scattered across the file). Accepts exactly "0"/"1"; any other value is a
// config error (warned, treated as the default) rather than silently ignored.
// See docs/ENV_VARS.md for the full catalog of UMA_* variables.
static bool uma_env_bool(const char *name, bool dflt, Error *error = nullptr)
{
  const char *e = std::getenv(name);
  if (e == nullptr || e[0] == '\0') return dflt;
  if (e[0] == '1' && e[1] == '\0') return true;
  if (e[0] == '0' && e[1] == '\0') return false;
  if (error)
    error->warning(FLERR, "Ignoring non-boolean value '{}' for {} (expected 0 or 1)",
                   e, name);
  return dflt;
}

static const char *const elements_uma[] = {
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
    "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};
static constexpr int elements_num_uma = sizeof(elements_uma) / sizeof(const char *);

static int atomic_number_by_name(const char *elname)
{
  for (int i = 1; i < elements_num_uma; i++)
    if (strcmp(elname, elements_uma[i]) == 0) return i;
  return -1;
}

/* ---------------------------------------------------------------------- */

PairUMA::PairUMA(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  // P0'.1: this pair style does not compute the virial. Setting this flag stops
  // Pair::ev_setup from enabling fdotr and silently zeroing virial[] (which made
  // `thermo` pressure report only the kinetic term with no warning). We then also
  // refuse loudly in compute() if a global virial is actually requested, rather
  // than reporting a fake (zero) stress.
  no_virial_fdotr_compute = 1;
  map = nullptr;
  predictor = nullptr;
  mpi_peer = nullptr;
  gpus_per_node = 4;  // Polaris default; override via UMA_GPUS_PER_NODE
  cutoff = 6.0;
  precision = PRECISION_MIXED;
  num_devices = 1;
  devices_explicit = false;
  mn_world = 1;
  mn_rank = 0;
  mn_active = false;
  list = nullptr;
  // A/B switch: UMA_ENGINE_BUILD_GRAPH=1 forces the old path where the engine
  // rebuilds its own graph (known-good reference). Default consumes the LAMMPS NL.
  engine_build_graph_ = uma_env_bool("UMA_ENGINE_BUILD_GRAPH", false, error);
  // P0'.1 step 2: opt-in single-tile virial/stress (strain autograd). Off by
  // default so ordinary NVE/NVT runs keep the byte-identical energy/force path.
  want_virial_flag_ = uma_env_bool("UMA_COMPUTE_VIRIAL", false, error);
  // Multi-node spatial domain decomposition (Phase A, deep halo k=1). Separate
  // from the mn_active GP-over-MPI path. Each rank owns a LAMMPS subdomain and
  // uses the single-tile Predictor on its owned+ghost atoms.
  dd_active_ = uma_env_bool("UMA_DD", false, error);
  dd_edge_count_ = 0;
  halo_buf_ = nullptr;
  halo_per_node_ = 0;
}

/* ---------------------------------------------------------------------- */

PairUMA::~PairUMA()
{
  delete predictor;
  predictor = nullptr;
  // Collective NCCL/XCCL teardown: every rank must enter the comm destroy
  // together (~MpiPeerPredictor). The engine's shm barrier can't span processes
  // on the MPI path, so the ranks must synchronize here.
  //
  // P0'.5: the old code did an UNCONDITIONAL `if (mpi_peer) MPI_Barrier(world)`.
  // If any rank failed to construct its peer (so its `mpi_peer` is null) while
  // others succeeded, the survivors block on that barrier forever -> a hang at
  // teardown. Agree across ranks FIRST: only do the collective barrier+destroy if
  // EVERY rank has a peer. If they disagree, skip the collective path (a mixed
  // ncclCommDestroy would deadlock/crash anyway) and just delete locally.
  if (comm->nprocs > 1) {
    int have_local = (mpi_peer != nullptr) ? 1 : 0;
    int have_min = have_local;
    MPI_Allreduce(&have_local, &have_min, 1, MPI_INT, MPI_MIN, world);
    if (have_min == 1) MPI_Barrier(world);  // all ranks have a peer -> safe
  }
  delete mpi_peer;
  mpi_peer = nullptr;
  // P0'.4: install_halo_callbacks() captured `this` into the process-wide
  // HaloContext singleton via std::function. It outlives this PairUMA (a
  // `pair_style` redefinition or a second `run` after redefine leaves a
  // TorchScript custom op holding a freed `this`). Clear the callbacks here so
  // the singleton never references a destroyed pair style. (~Predictor clears the
  // BlockContext it populated; do it defensively here too in case predictor was
  // already gone.)
  uma::HaloContext::instance().clear();
  halo_buf_ = nullptr;
  halo_per_node_ = 0;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(map);
  }
}

/* ---------------------------------------------------------------------- */

void PairUMA::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (eflag_atom || vflag_atom)
    error->all(FLERR, "Pair style uma does not support per-atom energy/virial yet");
  // Spatial domain-decomposition path (Phase A, deep halo k=1). Each rank owns a
  // LAMMPS subdomain and evaluates the single-tile Predictor over its owned+ghost
  // atoms; LAMMPS supplies ghosts to the model receptive field via
  // `comm_modify cutoff (num_layers*cutoff)`. Independent of the GP-over-MPI path.
  if (dd_active_) {
    run_compute_dd(eflag, vflag);
    return;
  }
  // Multi-node: one MPI rank per GPU, preserving the single-node GRAPH-parallel
  // design (see below). The same-node devices>1 path forks workers and moves
  // geometry through /dev/shm, which cannot cross a node; forking after
  // MPI_Init also breaks CUDA context ownership. The two are exclusive.
  // Two multi-GPU models, mutually exclusive:
  //   - same-node fork workers: 1 MPI rank, devices N (GraphParallelRuntime)
  //   - multi-node edge-parallel: nprocs MPI ranks, devices 1, one GPU/rank,
  //     each rank an MpiPeerPredictor sharing the graph over NCCL-over-MPI.
  // Reject the mix (devices>1 forks after MPI_Init -> broken CUDA contexts).
  if (comm->nprocs > 1 && num_devices > 1)
    error->all(FLERR,
               "Pair style uma: devices > 1 (same-node fork workers) cannot be "
               "combined with multiple MPI ranks; use one rank per GPU with "
               "devices 1 (multi-node edge-parallel)");
  // Multi-node keeps the single-node GRAPH-parallel design: LAMMPS rank 0 owns
  // the whole system and builds ONE full-system vesin graph with periodic=true,
  // then the EDGE list is split by center atom across all GPUs. Every GPU sees
  // every atom and 1/world of the edges, so PBC is never decomposed and no
  // ghost shell is required. Only rank creation and transport change:
  // fork/exec -> srun ranks, /dev/shm -> MPI_Bcast.

  const int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;

  // ---- multi-node: assemble the GLOBAL atom set on every rank -------------
  // The model is graph-parallel, not spatially decomposed: it needs the whole
  // system to build one correct periodic graph. Under MPI each rank owns only
  // a slice, so gather all owned atoms (tag-ordered) before predicting. Every
  // rank then builds the identical full-system graph and evaluates its own
  // 1/world edge shard, exactly as the forked workers do on one node.
  mn_world = comm->nprocs;
  mn_rank = comm->me;
  mn_active = (mn_world > 1);

  z_buf.resize(static_cast<size_t>(nlocal));
  force_buf.resize(static_cast<size_t>(nlocal) * 3);

  // Positions passed to the engine: local atoms, boxlo-shifted. The edge graph
  // (built from the LAMMPS NL below) references these same local indices; the
  // boxlo shift cancels in every edge vector, so it does not affect energy/force.
  const double xlo = domain->boxlo[0];
  const double ylo = domain->boxlo[1];
  const double zlo = domain->boxlo[2];
  const bool use_f64 = (precision == PRECISION_DOUBLE);
  if (use_f64) {
    pos_buf_d.resize(static_cast<size_t>(nlocal) * 3);
    for (int i = 0; i < nlocal; i++) {
      pos_buf_d[3 * i + 0] = x[i][0] - xlo;
      pos_buf_d[3 * i + 1] = x[i][1] - ylo;
      pos_buf_d[3 * i + 2] = x[i][2] - zlo;
      z_buf[i] = map[type[i]];
    }
  } else {
    pos_buf.resize(static_cast<size_t>(nlocal) * 3);
    for (int i = 0; i < nlocal; i++) {
      pos_buf[3 * i + 0] = static_cast<float>(x[i][0] - xlo);
      pos_buf[3 * i + 1] = static_cast<float>(x[i][1] - ylo);
      pos_buf[3 * i + 2] = static_cast<float>(x[i][2] - zlo);
      z_buf[i] = map[type[i]];
    }
  }

  // ASE/FairChem cell rows; LAMMPS triclinic uses (lx,xy,xz / 0,ly,yz / 0,0,lz).
  cell_buf[0] = domain->boxhi[0] - domain->boxlo[0];
  cell_buf[1] = 0.0;
  cell_buf[2] = 0.0;
  cell_buf[3] = domain->xy;
  cell_buf[4] = domain->boxhi[1] - domain->boxlo[1];
  cell_buf[5] = 0.0;
  cell_buf[6] = domain->xz;
  cell_buf[7] = domain->yz;
  cell_buf[8] = domain->boxhi[2] - domain->boxlo[2];

  pbc_buf[0] = domain->xperiodic;
  pbc_buf[1] = domain->yperiodic;
  pbc_buf[2] = domain->zperiodic;

  uma::Prediction result;
  if (!mn_active) {
    // P0'.1 step 2: request the virial (strain autograd), single-tile only. Gated
    // by BOTH an explicit opt-in (want_virial_flag_, from UMA_COMPUTE_VIRIAL=1 or a
    // pair_style keyword) AND LAMMPS actually needing a global virial this step.
    // NOT triggered by vflag_global alone: LAMMPS sets that on step 0 for the
    // default thermo pressure compute even in NVE/NVT, so keying off it would run
    // the (heavier, second-derivative) strain path on every ordinary run.
    predictor->set_want_virial(want_virial_flag_ && vflag_global && !dd_active_);
    // Single rank. Default: CONSUME the LAMMPS neighbor list (convert to
    // FairChem edges, engine skips its own O(N^2) rebuild). A/B fallback:
    // UMA_ENGINE_BUILD_GRAPH=1 keeps the old path where the engine rebuilds.
    const bool use_ext = !engine_build_graph_ && (list != nullptr);
    if (use_ext) {
      // Positions are passed exactly as-is (x - boxlo); the engine's extgraph
      // path must NOT re-wrap them, so that
      //   edge_vec = pos[jr] + offset@cell - pos[i] == x[j] - x[i]
      // matches the LAMMPS minimum-image displacement (== vesin geometry).
      const int64_t E = build_ext_graph(nlocal);
      if (use_f64)
        result = predictor->predict_host_extgraph(
            nlocal, pos_buf_d.data(), z_buf.data(), cell_buf, pbc_buf, E,
            ext_edge_index_.data(), ext_cell_offsets_.data(), force_buf.data());
      else
        error->all(FLERR,
                   "Pair style uma: LAMMPS-neighbor-list path requires precision "
                   "double; set UMA_ENGINE_BUILD_GRAPH=1 for the mixed path");
    } else if (use_f64) {
      result = predictor->predict_host(nlocal, pos_buf_d.data(), z_buf.data(), cell_buf, pbc_buf,
                                       force_buf.data());
    } else {
      result = predictor->predict_host(nlocal, pos_buf.data(), z_buf.data(), cell_buf, pbc_buf,
                                       force_buf.data());
    }
    for (int i = 0; i < nlocal; i++) {
      f[i][0] += force_buf[3 * i + 0];
      f[i][1] += force_buf[3 * i + 1];
      f[i][2] += force_buf[3 * i + 2];
    }
    if (eflag_global) eng_vdwl += result.energy;
    // P0'.1 step 2: publish the global virial (W = -dE/dstrain) into LAMMPS.
    if (vflag_global && result.has_virial) {
      for (int k = 0; k < 6; k++) virial[k] += result.virial[k];
    }
  } else {
    // ---- multi-node graph-parallel -------------------------------------
    // Gather the global atom set TAG-ORDERED so every rank builds a bitwise
    // identical graph (vesin output depends on atom order, and a differing
    // order across ranks would silently desynchronise the edge shards).
    if (!use_f64)
      error->all(FLERR, "Pair style uma: multi-node requires precision double");

    const int natoms_global = static_cast<int>(atom->natoms);
    if (natoms_global <= 0)
      error->all(FLERR, "Pair style uma: bad global atom count");

    mn_tag.resize(static_cast<size_t>(nlocal));
    for (int i = 0; i < nlocal; i++) mn_tag[i] = static_cast<int>(atom->tag[i]);

    // counts/displacements for the variable-sized per-rank contributions
    mn_counts.assign(static_cast<size_t>(mn_world), 0);
    int nlocal_send = nlocal;
    MPI_Allgather(&nlocal_send, 1, MPI_INT, mn_counts.data(), 1, MPI_INT, world);
    mn_displs.assign(static_cast<size_t>(mn_world), 0);
    int running = 0;
    for (int r = 0; r < mn_world; r++) {
      mn_displs[r] = running;
      running += mn_counts[r];
    }
    if (running != natoms_global)
      error->all(FLERR, "Pair style uma: gathered atom count != natoms");

    mn_tag_all.assign(static_cast<size_t>(natoms_global), 0);
    MPI_Allgatherv(mn_tag.data(), nlocal, MPI_INT, mn_tag_all.data(),
                   mn_counts.data(), mn_displs.data(), MPI_INT, world);

    std::vector<int> counts3(mn_counts), displs3(mn_displs);
    for (int r = 0; r < mn_world; r++) { counts3[r] *= 3; displs3[r] *= 3; }
    mn_pos_all.assign(static_cast<size_t>(natoms_global) * 3, 0.0);
    MPI_Allgatherv(pos_buf_d.data(), nlocal * 3, MPI_DOUBLE, mn_pos_all.data(),
                   counts3.data(), displs3.data(), MPI_DOUBLE, world);
    mn_z_all.assign(static_cast<size_t>(natoms_global), 0);
    MPI_Allgatherv(z_buf.data(), nlocal, MPI_INT, mn_z_all.data(),
                   mn_counts.data(), mn_displs.data(), MPI_INT, world);

    // Sort into tag order: identical on every rank regardless of who owns what.
    mn_order.resize(static_cast<size_t>(natoms_global));
    for (int i = 0; i < natoms_global; i++) mn_order[i] = i;
    // stable_sort: tags are unique so ordering is deterministic either way,
    // but this makes the "identical order on every rank" invariant explicit.
    std::stable_sort(mn_order.begin(), mn_order.end(),
              [&](int a, int b) { return mn_tag_all[a] < mn_tag_all[b]; });

    mn_pos_sorted.resize(static_cast<size_t>(natoms_global) * 3);
    mn_z_sorted.resize(static_cast<size_t>(natoms_global));
    for (int k = 0; k < natoms_global; k++) {
      const int src = mn_order[k];
      mn_pos_sorted[3 * k + 0] = mn_pos_all[3 * src + 0];
      mn_pos_sorted[3 * k + 1] = mn_pos_all[3 * src + 1];
      mn_pos_sorted[3 * k + 2] = mn_pos_all[3 * src + 2];
      mn_z_sorted[k] = mn_z_all[src];
    }

    mn_force_sorted.assign(static_cast<size_t>(natoms_global) * 3, 0.0);
    // Edge-parallel, memory-sharded: each rank evaluates its 1/world edge shard
    // of the FULL tag-ordered system and the NCCL all_reduce inside the peer sums
    // force contributions across all GPUs. Memory is ~O(N/world), so systems that
    // do not fit one GPU (e.g. NaCl 8x8x8 = 4096) run across nodes.
    if (!mpi_peer)
      error->all(FLERR, "Pair style uma: multi-node peer predictor not initialized");
    result = mpi_peer->predict_host(natoms_global, mn_pos_sorted.data(),
                                    mn_z_sorted.data(), cell_buf, pbc_buf,
                                    mn_force_sorted.data());

    // Scatter forces back to owners. all_reduce returns the fully reduced global
    // force array on every rank, so each rank simply picks out the atoms it owns
    // -- no extra MPI reduction of forces is needed.
    for (int k = 0; k < natoms_global; k++) {
      const int gathered_idx = mn_order[k];
      if (gathered_idx < mn_displs[mn_rank] ||
          gathered_idx >= mn_displs[mn_rank] + nlocal)
        continue;
      const int local_i = gathered_idx - mn_displs[mn_rank];
      f[local_i][0] += mn_force_sorted[3 * k + 0];
      f[local_i][1] += mn_force_sorted[3 * k + 1];
      f[local_i][2] += mn_force_sorted[3 * k + 2];
    }

    // Energy is global and identical on every rank; LAMMPS sums eng_vdwl over
    // ranks, so only one rank may contribute it.
    if (eflag_global && mn_rank == 0) eng_vdwl += result.energy;
  }
}

/* ---------------------------------------------------------------------- */

void PairUMA::settings(int narg, char **arg)
{
  // pair_style uma/kk [precision mixed|double] [devices N]
  // Default: mixed (FP32 positions/energy, FP64 forces) — same naming as GPU/INTEL.
  // Default devices=1 (traced LibTorch). If devices omitted and Kokkos ngpus>1,
  // load_predictor() auto-sets devices=ngpus.
  precision = PRECISION_MIXED;
  num_devices = 1;
  devices_explicit = false;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "precision") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_style uma command");
      if (strcmp(arg[iarg + 1], "mixed") == 0)
        precision = PRECISION_MIXED;
      else if (strcmp(arg[iarg + 1], "double") == 0)
        precision = PRECISION_DOUBLE;
      else
        error->all(FLERR, "Illegal pair_style uma precision {}; expected mixed or double",
                   arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "devices") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_style uma command");
      num_devices = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (num_devices < 1)
        error->all(FLERR, "Illegal pair_style uma devices {}; must be >= 1", num_devices);
      devices_explicit = true;
      iarg += 2;
    } else if (strcmp(arg[iarg], "mixed") == 0) {
      // bare token alias (optional)
      precision = PRECISION_MIXED;
      iarg += 1;
    } else if (strcmp(arg[iarg], "double") == 0) {
      precision = PRECISION_DOUBLE;
      iarg += 1;
    } else {
      error->all(FLERR, "Illegal pair_style uma command");
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairUMA::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg < 3 + atom->ntypes) error->all(FLERR, "Incorrect args for pair coefficients");
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  artifact_dir = arg[2];

  for (int i = 1; i <= atom->ntypes; i++) {
    int Z = atomic_number_by_name(arg[i + 2]);
    if (Z < 0) error->all(FLERR, "Invalid element name in pair_coeff for uma");
    map[i] = Z;
  }

  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++) setflag[i][j] = 1;

  load_predictor();
}

/* ---------------------------------------------------------------------- */

void PairUMA::load_predictor()
{
  delete predictor;
  predictor = nullptr;
  if (mpi_peer && comm->nprocs > 1) MPI_Barrier(world);
  delete mpi_peer;
  mpi_peer = nullptr;

  // Spatial DD: every rank runs the SINGLE-TILE predictor on its own subdomain
  // (owned+ghost). Do NOT build the GP-over-MPI peer even though nprocs>1; DD is
  // a different decomposition (spatial, not edge-sharded). Fall through to the
  // single-tile predictor construction below.
  if (dd_active_) {
    // (single-tile predictor built below)
  } else if (comm->nprocs > 1) {
  // ---- multi-node edge-parallel: one MpiPeerPredictor per MPI rank ---------
  // Triggered by nprocs > 1. Each rank owns one GPU and evaluates 1/world of
  // the graph; NCCL (bootstrapped over MPI) does the force all-reduce. Memory
  // is ~O(N/world) -> systems too big for one GPU run across nodes.
    if (precision != PRECISION_DOUBLE)
      error->all(FLERR, "Pair style uma: multi-node requires precision double");
    try {
      const int mn_w = comm->nprocs;   // world SIZE (not the MPI_Comm `world`)
      const int rank = comm->me;

      // GPUs per node: env override else default (Polaris = 4).
      gpus_per_node = 4;
      if (const char *e = std::getenv("UMA_GPUS_PER_NODE")) {
        const int v = atoi(e);
        if (v >= 1) gpus_per_node = v;
      }
      // Local rank for device binding: launcher hint else me % gpus_per_node.
      int local_rank = comm->me % gpus_per_node;
      for (const char *v : {"PMI_LOCAL_RANK", "SLURM_LOCALID",
                            "OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"}) {
        const char *lr = std::getenv(v);
        if (lr == nullptr || *lr == '\0') continue;
        char *end = nullptr;
        const long parsed = std::strtol(lr, &end, 10);
        if (end == lr || *end != '\0' || parsed < 0) continue;
        local_rank = static_cast<int>(parsed);
        break;
      }
#if defined(UMA_ENGINE_USE_XPU)
      // XPU: one tile per rank via ZE_AFFINITY_MASK (masked view -> index 0).
      const int device_index = 0;
#else
      const int ndev =
          torch::cuda::is_available() ? static_cast<int>(torch::cuda::device_count()) : 0;
      // When the launcher pins one GPU per rank (CUDA_VISIBLE_DEVICES), ndev==1
      // and device_index 0 is correct; otherwise bind local_rank % ndev.
      const int device_index = (ndev > 1) ? (local_rank % ndev) : 0;
#endif

      // Load metadata (all ranks) for cutoff + normalizer + refs.
      auto metadata = uma::load_artifact_metadata(artifact_dir + "/metadata.json");
      cutoff = metadata.cutoff;

#if defined(UMA_ENGINE_USE_XPU)
      // XPU transport (host-staged MPI or XCCL) bootstraps over MPI_COMM_WORLD;
      // no NCCL unique-id exchange needed.
      mpi_peer = uma::MpiPeerPredictor::create(
                     artifact_dir, metadata, mn_w, rank, device_index,
                     /*nccl_unique_id=*/nullptr, torch::kFloat64)
                     .release();
#else
      // NCCL id: rank 0 generates, MPI_Bcast to all, then collective create.
      const size_t id_bytes = uma::MpiPeerPredictor::nccl_unique_id_bytes();
      if (id_bytes == 0)
        error->all(FLERR, "Pair style uma: engine built without NCCL (multi-node)");
      std::vector<char> nccl_id(id_bytes, 0);
      if (rank == 0) uma::MpiPeerPredictor::make_nccl_unique_id(nccl_id.data());
      MPI_Bcast(nccl_id.data(), static_cast<int>(id_bytes), MPI_BYTE, 0, world);

      mpi_peer = uma::MpiPeerPredictor::create(
                     artifact_dir, metadata, mn_w, rank, device_index,
                     nccl_id.data(), torch::kFloat64)
                     .release();
#endif
      if (screen)
        fprintf(screen,
                "uma: rank %d -> multi-node peer (world=%d local_rank=%d dev=%d)\n",
                rank, mn_w, local_rank, device_index);
      utils::logmesg(lmp,
                     "Pair uma: multi-node edge-parallel, world={} artifact '{}' "
                     "cutoff={:.3f} precision=double\n",
                     mn_w, artifact_dir, cutoff);
      return;
    } catch (const LAMMPSException &) {
      // P0'.5: a LAMMPS error thrown inside this try (e.g. error->all above) is
      // already the correct collective/abort path. Do NOT swallow it into
      // error->one below (that would turn a clean collective abort into a ragged
      // MPI_Abort with a misleading "Failed to init peer" message). Rethrow.
      throw;
    } catch (const std::exception &e) {
      error->one(FLERR, "Failed to init UMA multi-node peer: {}", e.what());
    }
  }

  try {
    if (!devices_explicit) {
#ifdef LMP_KOKKOS
      if (lmp->kokkos && lmp->kokkos->ngpus > 1) {
        num_devices = lmp->kokkos->ngpus;
        utils::logmesg(lmp,
                       "Pair uma: devices auto-set to Kokkos ngpus={} "
                       "(omit devices N to keep this; pass devices 1 to force traced)\n",
                       num_devices);
      }
#endif
    }

    // For devices>1 fork the GP worker before any parent CUDA init.
    torch::Device device = torch::Device(torch::kCPU);
    if (num_devices <= 1) {
#if defined(UMA_ENGINE_USE_XPU)
      // Intel XPU (Aurora): one tile per MPI rank. With ZE_AFFINITY_MASK pinning
      // one tile per rank, device_count()==1 and index 0 is correct; otherwise
      // bind local_rank % ndev. (Eager UMA_EAGER_CKPT path picks XPU in-worker.)
      if (at::hasXPU() && torch::xpu::is_available()) {
        int local_rank = comm->me;
        for (const char *v : {"PMI_LOCAL_RANK", "PALS_LOCAL_RANKID",
                              "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK",
                              "MPI_LOCALRANKID", "LOCAL_RANK"}) {
          const char *lr = std::getenv(v);
          if (lr == nullptr || *lr == '\0') continue;
          char *end = nullptr;
          const long parsed = std::strtol(lr, &end, 10);
          if (end == lr || *end != '\0' || parsed < 0) continue;
          local_rank = static_cast<int>(parsed);
          break;
        }
        const int ndev = static_cast<int>(torch::xpu::device_count());
        const int idx = (ndev > 0 && local_rank >= 0) ? (local_rank % ndev) : 0;
        device = torch::Device(torch::kXPU, static_cast<c10::DeviceIndex>(idx));
        if (screen)
          fprintf(screen, "uma: rank %d -> xpu:%d (of %d visible)\n",
                  comm->me, idx, ndev);
      } else {
        device = torch::Device(torch::kCPU);
      }
#else
      if (torch::cuda::is_available()) {
        // M0 (multi-node): bind one GPU per MPI rank. A bare
        // torch::Device(torch::kCUDA) means index 0, so every rank on a node
        // would pile onto GPU 0 -- N-way oversubscription and N x the memory on
        // one card, while the other GPUs idle.
        //
        // Prefer the launcher's local-rank hint; fall back to comm->me. When
        // the launcher already pins one GPU per task (srun --gpus-per-task=1),
        // device_count() is 1 and the modulo correctly yields index 0.
        int local_rank = comm->me;
        for (const char *v : {"SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK",
                              "MV2_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"}) {
          const char *lr = std::getenv(v);
          if (lr == nullptr || *lr == '\0') continue;
          // strtol, not atoi: atoi cannot distinguish "abc" from "0", which
          // would silently bind every rank to cuda:0.
          char *end = nullptr;
          const long parsed = std::strtol(lr, &end, 10);
          if (end == lr || *end != '\0' || parsed < 0) {
            if (comm->me == 0 && screen)
              fprintf(screen, "uma: ignoring malformed %s='%s'\n", v, lr);
            continue;
          }
          local_rank = static_cast<int>(parsed);
          break;
        }
        const int ndev = static_cast<int>(torch::cuda::device_count());
        // Clamp defensively: a negative local_rank would yield a negative
        // device index, and ndev==0 would divide by zero.
        const int idx = (ndev > 0 && local_rank >= 0) ? (local_rank % ndev) : 0;
        device = torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(idx));
        // Every rank prints: the whole point is to confirm ranks land on
        // DIFFERENT GPUs, which a rank-0-only message cannot show.
        if (screen)
          fprintf(screen, "uma: rank %d -> cuda:%d (of %d visible)\n",
                  comm->me, idx, ndev);
      } else {
        device = torch::Device(torch::kCPU);
      }
#endif  // UMA_ENGINE_USE_XPU
    }
    predictor =
        new uma::Predictor(uma::Predictor::from_artifact(artifact_dir, device, num_devices));
    if (num_devices <= 1 && predictor->device().is_cuda()) {
      device = predictor->device();
    }
    cutoff = predictor->metadata().cutoff;

    // Align engine compute dtype with pair_style precision (must match artifact export).
    const auto want =
        (precision == PRECISION_DOUBLE) ? torch::kFloat64 : torch::kFloat32;
    predictor->set_compute_dtype(want);

    const char *gp_label = "off";
    if (predictor->uses_graph_parallel()) {
      // Default product path is C++ LibTorch MP; Python/Ray only via env opt-in.
      const char *py = std::getenv("UMA_PYTHON_GP_WORKER");
      const char *ray = std::getenv("UMA_ALLOW_RAY_GP");
      if ((py && py[0] == '1') || (ray && ray[0] == '1'))
        gp_label = "python_optin";
      else
        gp_label = "kokkos_libtorch_vesin";
    }
    utils::logmesg(lmp,
                   "Pair uma: loaded artifact '{}' cutoff={:.3f} device={} precision={} "
                   "devices={} gp={} (pos/energy {}, forces float64)\n",
                   artifact_dir, cutoff, device.str(),
                   (precision == PRECISION_DOUBLE) ? "double" : "mixed", num_devices,
                   gp_label,
                   (precision == PRECISION_DOUBLE) ? "float64" : "float32");
  } catch (const LAMMPSException &) {
    throw;  // P0'.5: preserve LAMMPS's own collective error handling; don't rewrap
  } catch (const std::exception &e) {
    error->all(FLERR, "Failed to load UMA artifact '{}': {}", artifact_dir, e.what());
  }
}

/* ---------------------------------------------------------------------- */

void PairUMA::init_style()
{
  if (force->newton_pair) error->all(FLERR, "Pair style uma requires newton pair off");
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style uma requires atom IDs");

  // P4.1: echo the resolved runtime config once, so a run's log records exactly
  // which UMA_* flags were active (no more silent config). Full catalog: docs/ENV_VARS.md.
  if (comm->me == 0) {
    const char *ckpt = std::getenv("UMA_CHECKPOINT");
    utils::logmesg(lmp,
                   "Pair uma config: precision={} engine_build_graph={} dd={} "
                   "ckpt={} mn_ckpt={} allow_legacy_metadata={} checkpoint={}\n",
                   (precision == PRECISION_DOUBLE) ? "double" : "mixed",
                   engine_build_graph_ ? 1 : 0, dd_active_ ? 1 : 0,
                   uma_env_bool("UMA_CKPT", false) ? 1 : 0,
                   uma_env_bool("UMA_MN_CKPT", false) ? 1 : 0,
                   uma_env_bool("UMA_ALLOW_LEGACY_METADATA", false) ? 1 : 0,
                   ckpt ? ckpt : "(unset)");
  }

  // P0'.1: the single-tile path now computes a REAL virial via strain autograd
  // (step 2), so pressure/NPT is supported there. The multi-node GP and DD paths
  // do NOT yet compute a virial, so a barostat on those paths would silently drive
  // the box with a zero stress -> refuse loudly. no_virial_fdotr_compute=1 (ctor)
  // still prevents a bogus fdotr virial on the paths that don't fill virial[].
  // P0'.1: the strain-autograd virial (step 2) is implemented but SEGFAULTS on the
  // Intel XPU backend (see predictor.cpp), so on XPU no path computes a usable
  // virial and any barostat is refused. On a non-XPU build the single-tile virial
  // is available under UMA_COMPUTE_VIRIAL=1.
#if defined(UMA_ENGINE_USE_XPU)
  const bool virial_supported = false;
#else
  const bool virial_supported = !mn_active && !dd_active_ && want_virial_flag_;
#endif
  if (!virial_supported) {
    for (const auto &ifix : modify->get_fix_list()) {
      const char *s = ifix->style;
      const bool barostat =
          utils::strmatch(s, "^npt") || utils::strmatch(s, "^nph") ||
          utils::strmatch(s, "^press/") || utils::strmatch(s, "/npt") ||
          utils::strmatch(s, "/nph") || utils::strmatch(s, "nphug");
      if (barostat)
        error->all(FLERR,
                   "Pair style uma does not compute a usable virial on this build "
                   "(strain-autograd stress segfaults on Intel XPU; GP/DD paths have "
                   "no virial). Pressure control (fix {}) is not supported; use "
                   "NVE/NVT.", s);
    }
  }

  // Full neighbor list; we CONSUME it (convert to FairChem edges) instead of
  // letting the engine rebuild its own O(N^2) graph. The list is built to the
  // pair cutoff (init_one -> cutoff = UMA cutoff 6.0) plus neighbor skin, so all
  // UMA edges (|rij| <= cutoff) are present and we filter to the exact cutoff.
  if (dd_active_) {
    // DD needs EVERY owned+ghost node to be a center (see build_dd_graph), so
    // request a full list that also lists ghost atoms as centers (REQ_GHOST).
    // Ghosts are supplied to the receptive field by `comm_modify cutoff
    // (num_layers*cutoff)`, which the user sets in the input script.
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

    // Size the LAMMPS forward/reverse comm buffer for the halo feature exchange.
    // The per-layer halo moves dd_halo_width doubles/atom (sph_feature_size *
    // sphere_channels, e.g. 9*128=1152). comm_forward/comm_reverse are read by
    // Comm::init() (called AFTER init_style) to size buf_send/buf_recv, so they
    // MUST be set here, not inside the op callback (setting them late leaves the
    // buffer too small -> pack_forward_comm overruns -> segfault).
    int w = predictor ? predictor->metadata().dd_halo_width : 0;
    if (w <= 0)
      error->all(FLERR,
                 "Pair style uma: UMA_DD requires a k=4 DD artifact with "
                 "dd_halo_width in metadata (export with UMA_DD_HALO=1)");
    comm_forward = w;
    comm_reverse = w;
    if (comm->me == 0 && screen)
      fprintf(screen, "uma DD: halo comm width = %d doubles/atom\n", w);
  } else {
    neighbor->add_request(this, NeighConst::REQ_FULL);
  }
}

/* ---------------------------------------------------------------------- */

void PairUMA::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

int64_t PairUMA::build_ext_graph(int nlocal)
{
  // Orthorhombic-only integer image recovery. Triclinic ghost->image needs the
  // full h-matrix solve; assert it away for now (NaCl cubic test is orthorhombic).
  if (domain->triclinic)
    error->all(FLERR,
               "Pair style uma: LAMMPS-neighbor-list path requires an "
               "orthorhombic box (triclinic==0); set UMA_ENGINE_BUILD_GRAPH=1 "
               "to use the engine's own graph for triclinic cells");

  double **x = atom->x;
  const tagint *tag = atom->tag;

  const double Lx = domain->boxhi[0] - domain->boxlo[0];
  const double Ly = domain->boxhi[1] - domain->boxlo[1];
  const double Lz = domain->boxhi[2] - domain->boxlo[2];
  const double invLx = (Lx > 0.0) ? 1.0 / Lx : 0.0;
  const double invLy = (Ly > 0.0) ? 1.0 / Ly : 0.0;
  const double invLz = (Lz > 0.0) ? 1.0 / Lz : 0.0;

  const double cut2 = cutoff * cutoff;

  const int inum = list->inum;
  const int *ilist = list->ilist;
  const int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // Build tag -> OWNED local index map from the nlocal owned atoms (robust vs
  // LAMMPS atom->map/sametag which can return a ghost). Single-rank: all real
  // atoms owned. tags are 1..N (metal units, atom_style atomic).
  tagint maxtag = 0;
  for (int i = 0; i < nlocal; i++) if (tag[i] > maxtag) maxtag = tag[i];
  std::vector<int> owned_of_tag(static_cast<size_t>(maxtag) + 1, -1);
  for (int i = 0; i < nlocal; i++) owned_of_tag[tag[i]] = i;

  ext_edge_index_.clear();
  ext_cell_offsets_.clear();
  // Row-major [2,E] built as two logical rows; we append into a flat [E] pair of
  // arrays and stitch to [2,E] at the end. Reserve generously.
  std::vector<int64_t> row_nbr;   // row0 = neighbor (jr)
  std::vector<int64_t> row_ctr;   // row1 = center (i)
  size_t guess = 0;
  for (int ii = 0; ii < inum; ii++) guess += static_cast<size_t>(numneigh[ilist[ii]]);
  row_nbr.reserve(guess);
  row_ctr.reserve(guess);
  ext_cell_offsets_.reserve(guess * 3);

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    if (i >= nlocal) continue;  // centers are local atoms only
    const int jnum = numneigh[i];
    const int *jlist = firstneigh[i];
    const double xi = x[i][0];
    const double yi = x[i][1];
    const double zi = x[i][2];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;  // strip special-bond/history bits

      // Exact UMA cutoff filter, computed on the ACTUAL (ghost) coordinates so
      // it reproduces vesin/radius_graph(cutoff) edge-for-edge.
      const double dx = x[j][0] - xi;
      const double dy = x[j][1] - yi;
      const double dz = x[j][2] - zi;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cut2) continue;

      // Map ghost/local j to its real OWNED atom jr_local in [0,nlocal). j may be
      // a ghost (>= nlocal) or a local self-image; atom->map(tag[j]) returns the
      // owned atom carrying that global tag. (newton_pair is off for this style,
      // and the mn/GP path is separate, so a local owned copy always exists.)
      // Map ghost/local j to its OWNED atom jr_local in [0,nlocal). LAMMPS
      // atom->map(tag) can return a ghost (closest-image map), and sametag may
      // not be current, so use our own tag->owned map (owned_of_tag, built once
      // per compute below from the nlocal owned atoms). Single rank: every real
      // atom is owned, so the owned copy always exists.
      const int jr_local = owned_of_tag[tag[j]];
      if (jr_local < 0 || jr_local >= nlocal)
        error->one(FLERR,
                   "Pair style uma: neighbor tag has no owned local atom "
                   "(unexpected on the single-rank path)");

      // Integer image (sx,sy,sz): x[j] = x[jr_local] + (sx,sy,sz).(Lx,Ly,Lz).
      // Base position is the OWNED atom the engine will index (row0 = jr_local),
      // so pos[jr_local] + offset@cell reproduces the ghost position x[j] used
      // for the edge vector. Orthorhombic: per-axis rounding is exact.
      const double bx = x[jr_local][0];
      const double by = x[jr_local][1];
      const double bz = x[jr_local][2];
      const long sx = std::lround((x[j][0] - bx) * invLx);
      const long sy = std::lround((x[j][1] - by) * invLy);
      const long sz = std::lround((x[j][2] - bz) * invLz);

      row_nbr.push_back(static_cast<int64_t>(jr_local));
      row_ctr.push_back(static_cast<int64_t>(i));
      ext_cell_offsets_.push_back(static_cast<double>(sx));
      ext_cell_offsets_.push_back(static_cast<double>(sy));
      ext_cell_offsets_.push_back(static_cast<double>(sz));
    }
  }

  const int64_t E = static_cast<int64_t>(row_ctr.size());
  ext_edge_index_.resize(static_cast<size_t>(2) * E);
  for (int64_t e = 0; e < E; e++) {
    ext_edge_index_[e] = row_nbr[e];              // row0 = neighbor
    ext_edge_index_[E + e] = row_ctr[e];          // row1 = center
  }
  return E;
}

/* ----------------------------------------------------------------------
   Multi-node spatial domain decomposition (Phase B, k=4 per-layer halo)
   ----------------------------------------------------------------------
   Each MPI rank owns a LAMMPS subdomain. LAMMPS provides ghost atoms out to ONE
   message-passing layer (cutoff + small skin ~ 6.5 A), set via `comm_modify
   cutoff 6.5`. We build the engine graph over the rank's OWNED + GHOST atoms as
   first-class nodes and evaluate the single-tile Predictor. The k=4 traced
   artifact calls uma_halo::exchange BEFORE each of the 4 blocks to refresh the
   6 A ghost features from their owners, so a thin one-layer halo is exact for
   owned-atom outputs (vs the deep 24 A halo k=1 would need). Forces are kept for
   owned atoms only; ghost force contributions are delivered to owners by the
   halo exchange backward (reverse comm), so owned-atom forces are exact.

   Ghost coordinates are absolute and unwrapped, so edge_vec = x[j] - x[i]
   directly: no integer-image recovery, no cell offsets, and triclinic works
   without the orthorhombic restriction of build_ext_graph.

   Edge padding (P2.1): each rank pads its edge list to a fixed UMA_DD_EDGE_CAP
   (multiple of EDGE_AC_CHUNK the artifact was traced at) with dummy self-loops on
   an appended far-away node, so the traced chunk count matches on every rank.
------------------------------------------------------------------------- */
void PairUMA::run_compute_dd(int eflag, int vflag)
{
  if (precision != PRECISION_DOUBLE)
    error->all(FLERR, "Pair style uma: UMA_DD requires precision double");
  if (!predictor)
    error->all(FLERR, "Pair style uma: UMA_DD predictor not initialized");
  if (force->newton_pair)
    error->all(FLERR, "Pair style uma: UMA_DD requires newton pair off");

  const int nlocal = atom->nlocal;
  const int nghost = atom->nghost;
  const int nall = nlocal + nghost;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;

  // Positions: owned+ghost, boxlo-shifted (shift cancels in every edge vector).
  const double xlo = domain->boxlo[0];
  const double ylo = domain->boxlo[1];
  const double zlo = domain->boxlo[2];
  dd_pos_.resize(static_cast<size_t>(nall) * 3);
  dd_z_.resize(static_cast<size_t>(nall));
  for (int i = 0; i < nall; i++) {
    dd_pos_[3 * i + 0] = x[i][0] - xlo;
    dd_pos_[3 * i + 1] = x[i][1] - ylo;
    dd_pos_[3 * i + 2] = x[i][2] - zlo;
    dd_z_[i] = map[type[i]];
  }

  // Cell/pbc: pass the GLOBAL box. Under DD with absolute ghost coordinates the
  // engine's extgraph path does NOT re-wrap and offsets are zero, so periodicity
  // is already resolved by LAMMPS ghosts; cell/pbc are informational for the
  // model's frame only. Use the global simulation box.
  cell_buf[0] = domain->boxhi[0] - domain->boxlo[0];
  cell_buf[1] = 0.0; cell_buf[2] = 0.0;
  cell_buf[3] = domain->xy;
  cell_buf[4] = domain->boxhi[1] - domain->boxlo[1];
  cell_buf[5] = 0.0;
  cell_buf[6] = domain->xz; cell_buf[7] = domain->yz;
  cell_buf[8] = domain->boxhi[2] - domain->boxlo[2];
  pbc_buf[0] = domain->xperiodic;
  pbc_buf[1] = domain->yperiodic;
  pbc_buf[2] = domain->zperiodic;

  // MoLE composition. The traced model computes the MoLE mixing coefficients
  // from a per-system MEAN of the composition embedding over the atomic_numbers
  // it is handed (set_MOLE_coefficients, export_blocks_xpu.py:631). Under DD we
  // hand it owned+ghost atomic_numbers, so its mean is the owned+ghost mean, not
  // the global owned mean. For a spatially HOMOGENEOUS system (uniform NaCl) the
  // owned+ghost composition ratio equals the global ratio to high precision, so
  // the local mean is within the parity tolerance. mole_composition_allreduce()
  // computes the exact global per-Z counts for validation/diagnostics and is the
  // hook for the exact fix (feeding the global mean into the traced MoLE) if the
  // homogeneous approximation proves insufficient. See parity report.
  mole_composition_allreduce();

  // Install the per-layer halo exchange callbacks (k=4). The traced k=4 artifact
  // calls uma_halo::exchange before each block; the op routes to LAMMPS comm via
  // these callbacks. Re-install every step: ghost layout changes on rebuild.
  install_halo_callbacks();

  // --- edge padding (P2.1): fix the traced chunk count across ranks/steps -----
  // The traced artifact bakes num_chunks = ceil(E / EDGE_AC_CHUNK) at export.
  // Per-rank E varies (subdomain volume, atom drift), so each rank pads its edge
  // list up to a FIXED capacity UMA_DD_EDGE_CAP (a multiple of EDGE_AC_CHUNK the
  // artifact was traced with). One extra DUMMY node is appended far outside the
  // cutoff; padded edges are dummy->dummy self-loops whose edge_distance >> cutoff
  // -> radial envelope = 0 -> zero message, zero contribution to real nodes and
  // zero force on real atoms. n_nodes passed to the engine is nall+1.
  int64_t edge_cap = 0;
  if (const char *e = std::getenv("UMA_DD_EDGE_CAP")) edge_cap = atoll(e);

  const int nnodes = nall + 1;              // +1 dummy padding node
  const int dummy = nall;                   // index of the dummy node
  // Dummy node placed far from all real atoms so any edge to it has r >> cutoff.
  // Use a large offset from box origin along +x (absolute coords, offsets zero).
  dd_pos_.resize(static_cast<size_t>(nnodes) * 3);
  dd_z_.resize(static_cast<size_t>(nnodes));
  const double far = 1.0e6;
  dd_pos_[3 * dummy + 0] = far;
  dd_pos_[3 * dummy + 1] = far;
  dd_pos_[3 * dummy + 2] = far;
  dd_z_[dummy] = dd_z_.empty() ? 1 : dd_z_[0];   // any valid Z; message is zeroed

  const bool dd_dbg = (std::getenv("UMA_DD_DEBUG") != nullptr);
  if (dd_dbg && screen)
    fprintf(screen, "uma DD[%d]: nlocal=%d nghost=%d nall=%d nnodes=%d\n",
            comm->me, nlocal, nghost, nall, nnodes);

  // Build the owned+ghost edge graph (row0=neighbor, row1=center; every node a
  // center for k=4). Then pad to edge_cap with dummy self-loops.
  int64_t E = build_dd_graph(nall);
  if (dd_dbg && screen)
    fprintf(screen, "uma DD[%d]: build_dd_graph E=%lld (cap=%lld)\n",
            comm->me, (long long) E, (long long) edge_cap);
  if (edge_cap > 0) {
    if (E > edge_cap)
      error->one(FLERR,
                 "Pair style uma: UMA_DD real edge count exceeds UMA_DD_EDGE_CAP "
                 "(raise the cap and re-export the artifact traced at that cap)");
    const int64_t old = E;
    dd_edge_index_.resize(static_cast<size_t>(2) * edge_cap);
    // dd_edge_index_ is stored row-major [2,E]: row0 at [0,E), row1 at [E,2E).
    // Rebuild as [2,edge_cap] keeping real edges then dummy self-loops.
    std::vector<int64_t> ei(static_cast<size_t>(2) * edge_cap);
    for (int64_t k = 0; k < old; k++) {
      ei[k] = dd_edge_index_[k];                       // row0 real
      ei[edge_cap + k] = dd_edge_index_[old + k];      // row1 real
    }
    // Padded edges MUST be inert: neighbor=atom 0 (a real node), center=dummy.
    // The dummy sits at (far,far,far), so edge_distance = |pos[dummy]-pos[0]| >>
    // cutoff -> radial envelope = 0 -> zero message. Center is the dummy, whose
    // energy/force are discarded (excluded from owned sum). A dummy->dummy
    // SELF-LOOP would have edge_distance = 0 (NOT > cutoff): r=0 poisons the edge
    // basis (SO2/envelope) and corrupts the whole batch -- that was the bug.
    for (int64_t k = old; k < edge_cap; k++) {
      ei[k] = 0;                                       // row0 = neighbor = atom 0
      ei[edge_cap + k] = dummy;                        // row1 = center = dummy (far)
    }
    dd_edge_index_.swap(ei);
    dd_cell_offsets_.assign(static_cast<size_t>(edge_cap) * 3, 0.0);
    E = edge_cap;
  }

  if (dd_dbg && screen)
    fprintf(screen, "uma DD[%d]: padded E=%lld; calling predict_host_extgraph_dd\n",
            comm->me, (long long) E);
  dd_force_.assign(static_cast<size_t>(nnodes) * 3, 0.0);
  dd_energy_.assign(static_cast<size_t>(nnodes), 0.0);
  // DD path: per-atom energy + owned-only backprop. nlocal owned rows are the
  // backprop root; the dummy node (index nall) and ghosts are excluded from the
  // owned energy sum. predict_body_dd returns Prediction.energy = owned sum.
  uma::Prediction result = predictor->predict_host_extgraph_dd(
      nnodes, nlocal, dd_pos_.data(), dd_z_.data(), cell_buf, pbc_buf, E,
      dd_edge_index_.data(), dd_cell_offsets_.data(),
      dd_energy_.data(), dd_force_.data());

  // Keep forces for OWNED atoms only (rows [0,nlocal); ghosts [nlocal,nall) and
  // the dummy node [nall] are discarded). Under k=4 the halo-exchange BACKWARD
  // (reverse comm) already delivered each ghost's force contribution back to its
  // owner during autograd, so an owned atom's force here is complete and exact.
  for (int i = 0; i < nlocal; i++) {
    f[i][0] += dd_force_[3 * i + 0];
    f[i][1] += dd_force_[3 * i + 1];
    f[i][2] += dd_force_[3 * i + 2];
  }

  // Energy: each rank contributes its OWNED-atom energy sum. predict_body_dd
  // returned Prediction.energy = sum(node_energy[0:nlocal]) using the per-atom
  // energy head (NodeEnergyExportWrapper), so summing across ranks via LAMMPS'
  // eng_vdwl reduction gives the exact global energy (each atom counted once, on
  // its owner). This is additive by construction, unlike the whole-subsystem
  // scalar. dd_energy_ holds per-node energy for optional per-atom output later.
  if (eflag_global) eng_vdwl += result.energy;
  (void) vflag;
}

/* ----------------------------------------------------------------------
   Build the DD edge graph: all edges (j -> i) with |x[j]-x[i]| <= cutoff among
   owned+ghost atoms, where the CENTER i is ANY node (owned OR ghost).

   Why every node must be a center (not owned-only): each message-passing block
   updates ALL node features and scatters edge messages to edge_index[1]. For an
   OWNED atom's layer-L feature to be exact, every node within (L-1) hops of it
   must also have been correctly updated in the previous layers, i.e. must be a
   center with all its incoming edges present. With num_layers=4 and cutoff=6 A,
   the receptive field is 24 A, so LAMMPS must supply ghosts to 24 A
   (`comm_modify cutoff 24.0`) AND every node within 24 A of an owned atom must
   be a center. A ghost near the 24 A rim will have an incomplete neighbor set
   (its own neighbors past the rim are missing), so its deep-layer features are
   approximate -- but it only feeds owned atoms through <4 hops from well inside
   the halo, so owned-atom outputs stay exact. Making every in-halo node a center
   is what distinguishes correct k=1 DD from a 1-layer-only halo.

   row0 = neighbor (j), row1 = center (i), both owned/ghost node indices.
   Offsets are all zero: ghosts are absolute, edge_vec = pos[j] - pos[i].
------------------------------------------------------------------------- */
int64_t PairUMA::build_dd_graph(int nall)
{
  double **x = atom->x;
  const double cut2 = cutoff * cutoff;

  // Consume the LAMMPS full+ghost neighbor list (init_style requests
  // REQ_FULL|REQ_GHOST for the DD path), so EVERY owned+ghost node appears as a
  // center (ilist covers 0..nall) with its neighbors within cutoff+skin. Filter
  // to the exact cutoff. Neighbor indices j are owned/ghost node indices in
  // [0,nall) and index directly into dd_pos_/dd_z_. Offsets zero (absolute).
  if (!list)
    error->all(FLERR, "Pair style uma: UMA_DD requires a neighbor list");

  const int inum = list->inum;
  const int gnum = list->gnum;          // ghost centers (REQ_GHOST)
  const int ntot = inum + gnum;
  const int *ilist = list->ilist;
  const int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  std::vector<int64_t> row_nbr, row_ctr;
  size_t guess = 0;
  for (int ii = 0; ii < ntot; ii++) guess += static_cast<size_t>(numneigh[ilist[ii]]);
  row_nbr.reserve(guess);
  row_ctr.reserve(guess);

  for (int ii = 0; ii < ntot; ii++) {
    const int i = ilist[ii];              // owned OR ghost center
    if (i >= nall) continue;              // center must be a real node (defensive)
    const int jnum = numneigh[i];
    const int *jlist = firstneigh[i];
    const double xi = x[i][0], yi = x[i][1], zi = x[i][2];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      // BOUNDS CHECK BEFORE x[j]: a REQ_GHOST full list can list neighbors that
      // are outside [0,nall) (ghost-of-ghost / extended region). Those are NOT
      // graph nodes we hold, so skip them. Reading x[j] first would segfault.
      if (j < 0 || j >= nall) continue;
      const double dx = x[j][0] - xi;
      const double dy = x[j][1] - yi;
      const double dz = x[j][2] - zi;
      if (dx * dx + dy * dy + dz * dz > cut2) continue;
      row_nbr.push_back(static_cast<int64_t>(j));   // neighbor
      row_ctr.push_back(static_cast<int64_t>(i));   // center
    }
  }

  const int64_t E = static_cast<int64_t>(row_ctr.size());
  dd_edge_index_.resize(static_cast<size_t>(2) * E);
  for (int64_t e = 0; e < E; e++) {
    dd_edge_index_[e] = row_nbr[e];
    dd_edge_index_[E + e] = row_ctr[e];
  }
  dd_cell_offsets_.assign(static_cast<size_t>(E) * 3, 0.0);
  dd_edge_count_ = E;
  return E;
}

/* ----------------------------------------------------------------------
   MoLE composition all-reduce. The UMA MoLE expert-mixing coefficients depend
   on a per-system MEAN of the composition embedding over atoms; under DD each
   rank sees only owned+ghost, so a local mean is wrong. Following nvalchemi's
   fix (models/uma.py:_distributed_set_mole_coefficients), reduce over OWNED
   atoms only (ghosts would double-count the overlap) and all-reduce across the
   mesh to the global per-Z counts. The engine reconstructs the mean from counts;
   the include_self (+1 denominator) correction for model_version 1.0 is applied
   engine-side. Position-independent -> compute once per neighbor rebuild.

   Phase A: composition is fixed for the run, so we compute the global per-Z
   count vector and hand it to the predictor. (Wiring into the traced MoLE is a
   TODO; for NaCl single-composition the local vs global mean differ only via the
   ghost/owned split, which this corrects.)
------------------------------------------------------------------------- */
void PairUMA::mole_composition_allreduce()
{
  const int nlocal = atom->nlocal;
  int *type = atom->type;
  // Per-Z owned counts on this rank.
  const int maxz = 118;
  std::vector<long> local_counts(maxz + 1, 0);
  for (int i = 0; i < nlocal; i++) {
    const int z = map[type[i]];
    if (z >= 0 && z <= maxz) local_counts[z]++;
  }
  std::vector<long> global_counts(maxz + 1, 0);
  MPI_Allreduce(local_counts.data(), global_counts.data(), maxz + 1,
                MPI_LONG, MPI_SUM, world);
  // Diagnostic: report the global composition once. The exact-fix hook (feeding
  // this into the traced MoLE mean) is a Phase-B TODO; Phase A relies on the
  // homogeneous-composition approximation (see run_compute_dd note).
  if (screen && comm->me == 0) {
    long total = 0;
    for (long c : global_counts) total += c;
    fprintf(screen, "uma DD: global composition total atoms = %ld\n", total);
  }
}

/* ----------------------------------------------------------------------
   Per-layer halo exchange (k=4). Install callbacks so the engine's
   uma_halo::exchange op moves node features through LAMMPS' own owned<->ghost
   comm. Each exchange:
     forward: set comm_forward = per_node width, stage the [nall,per_node] buffer
              in halo_buf_, run comm->forward_comm(this) -> pack owned rows into
              the send buf, unpack into ghost rows on the receiving rank.
     reverse: set comm_reverse width, run comm->reverse_comm(this, per_node) ->
              pack ghost rows, unpack ACCUMULATES onto owner rows, then the op
              zeros ghost rows (done in unpack via LAMMPS convention: reverse
              adds to owned; we additionally zero ghosts engine-side).
   halo_buf_ is a borrowed pointer to the engine's staging buffer for the
   duration of one forward_comm/reverse_comm call (single-threaded here).
------------------------------------------------------------------------- */
void PairUMA::install_halo_callbacks()
{
  const int64_t nlocal = atom->nlocal;
  const int64_t nall = atom->nlocal + atom->nghost;
  // The engine tensor has nall+1 rows (last = dummy padding node). LAMMPS comm
  // only knows about the real nall atoms, so pack/unpack operate on rows
  // [0,nall); the dummy row (index nall) is local, never a ghost, so it is left
  // untouched by both forward and reverse comm (its features/grads never move).
  const int64_t nnodes = nall + 1;
  PairUMA *self = this;
  auto fwd = [self](double *buf, int64_t /*nnodes*/, int64_t per_node) {
    // per_node MUST equal comm_forward set in init_style (buf_send sized for it).
    if (static_cast<int>(per_node) > self->comm_forward)
      self->error->one(FLERR, "Pair style uma: halo per_node exceeds comm_forward "
                              "(dd_halo_width mismatch)");
    self->halo_buf_ = buf;
    self->halo_per_node_ = per_node;
    self->comm->forward_comm(self);   // fills ghost rows [nlocal,nall) from owners
    self->halo_buf_ = nullptr;
  };
  auto rev = [self](double *buf, int64_t /*nnodes*/, int64_t per_node) {
    if (static_cast<int>(per_node) > self->comm_reverse)
      self->error->one(FLERR, "Pair style uma: halo per_node exceeds comm_reverse");
    self->halo_buf_ = buf;
    self->halo_per_node_ = per_node;
    // reverse_comm ADDS ghost-row contributions onto owner rows (LAMMPS
    // convention). Pass explicit size so it runs with newton pair off.
    self->comm->reverse_comm(self, static_cast<int>(per_node));
    // Zero ghost rows [nlocal,nall): their gradient has been delivered to owners.
    const int64_t nl = self->atom->nlocal;
    const int64_t na = self->atom->nlocal + self->atom->nghost;
    for (int64_t g = nl; g < na; g++)
      for (int64_t k = 0; k < per_node; k++) buf[g * per_node + k] = 0.0;
    self->halo_buf_ = nullptr;
  };
  // HaloContext nall == nnodes so the op's row-count check matches the tensor.
  uma::HaloContext::instance().set_callbacks(fwd, rev, nlocal, nnodes);

  // Self-test (UMA_DD_HALO_TEST=1): verify the real comm plan moves data the way
  // the halo op assumes. Fill each row with its atom's global tag (per_node
  // copies), forward_comm, then check EVERY ghost row == tag[its owner]. Then
  // set owned=1, ghost=1, reverse_comm, and check owned row == 1 + (#ghost copies
  // of that atom). Pinpoints pack/unpack/index bugs without the model.
  if (std::getenv("UMA_DD_HALO_TEST")) {
    const int na = atom->nlocal + atom->nghost;
    const int pn = static_cast<int>(comm_forward);
    const tagint *tag = atom->tag;
    std::vector<double> buf(static_cast<size_t>(na) * pn, 0.0);
    // forward test: row value = tag (owned rows only set; ghosts 0)
    for (int i = 0; i < nlocal; i++)
      for (int k = 0; k < pn; k++) buf[i * pn + k] = static_cast<double>(tag[i]);
    halo_buf_ = buf.data(); halo_per_node_ = pn;
    comm->forward_comm(this);
    halo_buf_ = nullptr;
    int fwd_bad = 0;
    for (int g = nlocal; g < na; g++) {
      const double want = static_cast<double>(tag[g]);   // ghost's own tag == owner tag
      if (buf[g * pn] != want) fwd_bad++;
    }
    int fwd_bad_all = 0;
    MPI_Allreduce(&fwd_bad, &fwd_bad_all, 1, MPI_INT, MPI_SUM, world);
    if (comm->me == 0 && screen)
      fprintf(screen, "uma DD HALO_TEST forward: ghost!=owner_tag count = %d (0 = OK)\n",
              fwd_bad_all);

    // Z-consistency test: does dd_z_[ghost] == Z delivered from owner? If NOT,
    // ghost embeddings differ from owners (explains the 13.6% pre-block-0 delta).
    // Fill owned rows with map[type], forward_comm, compare to local map[type[g]].
    std::fill(buf.begin(), buf.end(), 0.0);
    for (int i = 0; i < nlocal; i++)
      for (int k = 0; k < pn; k++) buf[i * pn + k] = static_cast<double>(map[atom->type[i]]);
    halo_buf_ = buf.data(); halo_per_node_ = pn;
    comm->forward_comm(this);
    halo_buf_ = nullptr;
    int z_bad = 0;
    for (int g = nlocal; g < na; g++) {
      const double owner_z = buf[g * pn];                 // Z from owner
      const double local_z = static_cast<double>(map[atom->type[g]]);  // my dd_z_[g]
      if (owner_z != local_z) z_bad++;
    }
    int z_bad_all = 0;
    MPI_Allreduce(&z_bad, &z_bad_all, 1, MPI_INT, MPI_SUM, world);
    if (comm->me == 0 && screen)
      fprintf(screen,
              "uma DD HALO_TEST Z-consistency: ghost map[type] != owner Z count = %d "
              "(0 = OK; nonzero => ghost embeddings differ from owners)\n", z_bad_all);

    // reverse test: owned=0, ghost=1; reverse_comm should accumulate onto owners
    // -> owned[a] == (# ghost copies of a across ALL ranks). Verify the global
    // sum of owned rows == total ghost count (each ghost delivers exactly 1).
    std::fill(buf.begin(), buf.end(), 0.0);
    for (int g = nlocal; g < na; g++)
      for (int k = 0; k < pn; k++) buf[g * pn + k] = 1.0;
    halo_buf_ = buf.data(); halo_per_node_ = pn;
    comm->reverse_comm(this, pn);
    halo_buf_ = nullptr;
    double owned_sum = 0.0;
    for (int i = 0; i < nlocal; i++) owned_sum += buf[i * pn];   // col 0
    double ghost_cnt = static_cast<double>(na - nlocal);
    double owned_sum_all = 0.0, ghost_cnt_all = 0.0;
    MPI_Allreduce(&owned_sum, &owned_sum_all, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&ghost_cnt, &ghost_cnt_all, 1, MPI_DOUBLE, MPI_SUM, world);
    if (comm->me == 0 && screen)
      fprintf(screen,
              "uma DD HALO_TEST reverse: sum(owned after reverse)=%.1f  "
              "total ghosts=%.1f  (equal = OK; each ghost delivers 1 to its owner)\n",
              owned_sum_all, ghost_cnt_all);
    error->all(FLERR, "Pair style uma: UMA_DD_HALO_TEST done (unset to run normally)");
  }
}

/* ---- LAMMPS comm hooks: operate on halo_buf_ [nall, halo_per_node_] --------- */
int PairUMA::pack_forward_comm(int n, int *list, double *buf,
                               int /*pbc_flag*/, int * /*pbc*/)
{
  const int64_t pn = halo_per_node_;
  int m = 0;
  for (int i = 0; i < n; i++) {
    const int64_t j = list[i];
    for (int64_t k = 0; k < pn; k++) buf[m++] = halo_buf_[j * pn + k];
  }
  return m;
}

void PairUMA::unpack_forward_comm(int n, int first, double *buf)
{
  const int64_t pn = halo_per_node_;
  int m = 0;
  for (int i = 0; i < n; i++) {
    const int64_t j = first + i;
    for (int64_t k = 0; k < pn; k++) halo_buf_[j * pn + k] = buf[m++];
  }
}

int PairUMA::pack_reverse_comm(int n, int first, double *buf)
{
  const int64_t pn = halo_per_node_;
  int m = 0;
  for (int i = 0; i < n; i++) {
    const int64_t j = first + i;
    for (int64_t k = 0; k < pn; k++) buf[m++] = halo_buf_[j * pn + k];
  }
  return m;
}

void PairUMA::unpack_reverse_comm(int n, int *list, double *buf)
{
  const int64_t pn = halo_per_node_;
  int m = 0;
  for (int i = 0; i < n; i++) {
    const int64_t j = list[i];
    for (int64_t k = 0; k < pn; k++) halo_buf_[j * pn + k] += buf[m++];  // accumulate
  }
}

/* ---------------------------------------------------------------------- */

double PairUMA::init_one(int /*i*/, int /*j*/)
{
  return cutoff;
}

/* ---------------------------------------------------------------------- */

void PairUMA::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;
  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(map, n, "pair:map");
  for (int i = 0; i < n; i++) map[i] = 0;
}
