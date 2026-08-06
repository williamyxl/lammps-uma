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

#include "pair_uma_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template <class DeviceType>
PairUMAKokkos<DeviceType>::PairUMAKokkos(LAMMPS *lmp) : PairUMA(lmp)
{
  respa_enable = 0;
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  // LibTorch gather runs on Host; sync x/type/f Host → compute → mark f modified.
  datamask_read = X_MASK | F_MASK | TYPE_MASK;
  datamask_modify = F_MASK;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
PairUMAKokkos<DeviceType>::~PairUMAKokkos()
{
  if (copymode) return;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void PairUMAKokkos<DeviceType>::compute(int eflag, int vflag)
{
  atomKK->sync(Host, X_MASK | TYPE_MASK | F_MASK);
  PairUMA::compute(eflag, vflag);
  atomKK->modified(Host, F_MASK);
  atomKK->sync(Device, F_MASK);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void PairUMAKokkos<DeviceType>::init_style()
{
  PairUMA::init_style();

  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                           !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
}

namespace LAMMPS_NS {
template class PairUMAKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairUMAKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
