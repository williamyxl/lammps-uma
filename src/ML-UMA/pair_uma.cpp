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
#include "utils.h"

#include "uma/predictor.h"

#ifdef LMP_KOKKOS
#include "kokkos.h"
#endif

#include <cstdlib>
#include <cstring>

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
  cutoff = 6.0;
  precision = PRECISION_MIXED;
  num_devices = 1;
  devices_explicit = false;
}

/* ---------------------------------------------------------------------- */

PairUMA::~PairUMA()
{
  delete predictor;
  predictor = nullptr;
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
  if (comm->nprocs > 1)
    error->all(FLERR, "Pair style uma currently supports a single MPI rank");

  const int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;

  z_buf.resize(static_cast<size_t>(nlocal));
  force_buf.resize(static_cast<size_t>(nlocal) * 3);

  // Engine owns its own neighbor graph; only local atoms are passed.
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
  if (use_f64)
    result = predictor->predict_host(nlocal, pos_buf_d.data(), z_buf.data(), cell_buf, pbc_buf,
                                     force_buf.data());
  else
    result = predictor->predict_host(nlocal, pos_buf.data(), z_buf.data(), cell_buf, pbc_buf,
                                     force_buf.data());

  for (int i = 0; i < nlocal; i++) {
    f[i][0] += force_buf[3 * i + 0];
    f[i][1] += force_buf[3 * i + 1];
    f[i][2] += force_buf[3 * i + 2];
  }

  if (eflag_global) eng_vdwl += result.energy;
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
      device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                           : torch::Device(torch::kCPU);
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

  // Register cutoff with neighbor; engine builds its own UMA graph.
  neighbor->add_request(this, NeighConst::REQ_FULL);
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
