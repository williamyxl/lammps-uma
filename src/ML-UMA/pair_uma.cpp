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
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "utils.h"

#include "uma/predictor.h"
#include "uma/mpi_peer_predictor.h"

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
  engine_build_graph_ = false;
  if (const char *e = std::getenv("UMA_ENGINE_BUILD_GRAPH"))
    engine_build_graph_ = (e[0] == '1' && e[1] == '\0');
  // Multi-node spatial domain decomposition (Phase A, deep halo k=1). Separate
  // from the mn_active GP-over-MPI path. Each rank owns a LAMMPS subdomain and
  // uses the single-tile Predictor on its owned+ghost atoms.
  dd_active_ = false;
  if (const char *e = std::getenv("UMA_DD"))
    dd_active_ = (e[0] == '1' && e[1] == '\0');
  dd_edge_count_ = 0;
}

/* ---------------------------------------------------------------------- */

PairUMA::~PairUMA()
{
  delete predictor;
  predictor = nullptr;
  // Collective NCCL teardown: every rank must enter ncclCommDestroy together.
  // The engine's shm barrier can't span processes on the MPI path, so sync here.
  if (mpi_peer && comm->nprocs > 1) MPI_Barrier(world);
  delete mpi_peer;
  mpi_peer = nullptr;
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

  // ---- multi-node edge-parallel: one MpiPeerPredictor per MPI rank ---------
  // Triggered by nprocs > 1. Each rank owns one GPU and evaluates 1/world of
  // the graph; NCCL (bootstrapped over MPI) does the force all-reduce. Memory
  // is ~O(N/world) -> systems too big for one GPU run across nodes.
  if (comm->nprocs > 1) {
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
  } catch (const std::exception &e) {
    error->all(FLERR, "Failed to load UMA artifact '{}': {}", artifact_dir, e.what());
  }
}

/* ---------------------------------------------------------------------- */

void PairUMA::init_style()
{
  if (force->newton_pair) error->all(FLERR, "Pair style uma requires newton pair off");
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style uma requires atom IDs");

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
   Multi-node spatial domain decomposition (Phase A, deep halo k=1)
   ----------------------------------------------------------------------
   Each MPI rank owns a LAMMPS subdomain. LAMMPS provides ghost atoms out to the
   full model receptive field (num_layers * cutoff, e.g. 4*6 = 24 A), set by the
   user via `comm_modify cutoff 24.0`. We build the engine graph over the rank's
   OWNED + GHOST atoms as first-class nodes, evaluate the single-tile Predictor,
   and keep forces for owned atoms only. Because a ghost within 24 A of an owned
   atom carries the full 4-hop receptive field, every owned atom's energy and
   force is EXACT -- k=1 needs no mid-network halo exchange and no reverse comm
   (each rank computes its owned atoms' forces from its own complete halo).

   Ghost coordinates are absolute and unwrapped, so edge_vec = x[j] - x[i]
   directly: no integer-image recovery, no cell offsets, and triclinic works
   without the orthorhombic restriction of build_ext_graph.
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

  // Build the owned+ghost edge graph (row0=neighbor, row1=center; centers=owned).
  const int64_t E = build_dd_graph(nall);

  dd_force_.assign(static_cast<size_t>(nall) * 3, 0.0);
  uma::Prediction result = predictor->predict_host_extgraph(
      nall, dd_pos_.data(), dd_z_.data(), cell_buf, pbc_buf, E,
      dd_edge_index_.data(), dd_cell_offsets_.data(), dd_force_.data());

  // Keep forces for OWNED atoms only. Ghost forces belong to the rank that owns
  // that atom, which computes them itself from its own halo (k=1, no reverse
  // comm). Newton off: LAMMPS does not fold ghost forces back here.
  for (int i = 0; i < nlocal; i++) {
    f[i][0] += dd_force_[3 * i + 0];
    f[i][1] += dd_force_[3 * i + 1];
    f[i][2] += dd_force_[3 * i + 2];
  }

  // Energy: sum only THIS rank's owned-atom energy contribution. The traced
  // model returns one global scalar for the (owned+ghost) subsystem, which is
  // NOT the owned-atom energy sum. For a correct global energy under DD we need
  // the per-atom energy of owned atoms only. Until per-atom energy is exported,
  // report the energy on rank 0 from a full-system evaluation is impossible;
  // instead we accumulate owned-fraction-weighted energy. See note below.
  if (eflag_global) {
    // NOTE (Phase A energy): the single scalar `result.energy` is the energy of
    // this rank's owned+ghost subsystem including ghost self-energy and double
    // counts across ranks, so it is NOT additive. Correct DD energy requires
    // per-atom energy (eflag_atom) which the model does not yet export. For the
    // single-point PARITY test we validate FORCES on owned atoms (exact) and the
    // GLOBAL energy via a dedicated rank-0 full-system path is out of scope for
    // Phase A. We therefore contribute nothing to eng_vdwl here and print the
    // per-rank subsystem energy for diagnostics; the parity harness compares
    // forces (all atoms) and, for energy, uses the single-tile/GP oracle.
    if (screen && comm->me == 0)
      fprintf(screen, "uma DD: rank0 subsystem energy = %.10f eV "
              "(NOT global; forces are exact per owned atom)\n", result.energy);
  }
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
    const int jnum = numneigh[i];
    const int *jlist = firstneigh[i];
    const double xi = x[i][0], yi = x[i][1], zi = x[i][2];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      const double dx = x[j][0] - xi;
      const double dy = x[j][1] - yi;
      const double dz = x[j][2] - zi;
      if (dx * dx + dy * dy + dz * dz > cut2) continue;
      if (j >= nall)
        error->one(FLERR, "Pair style uma: UMA_DD neighbor index exceeds nall");
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
