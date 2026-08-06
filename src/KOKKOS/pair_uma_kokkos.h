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
PairStyle(uma/kk,PairUMAKokkos<LMPDeviceType>);
PairStyle(uma/kk/device,PairUMAKokkos<LMPDeviceType>);
PairStyle(uma/kk/host,PairUMAKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_UMA_KOKKOS_H
#define LMP_PAIR_UMA_KOKKOS_H

#include "kokkos_base.h"
#include "pair_uma.h"

namespace LAMMPS_NS {

template <class DeviceType>
class PairUMAKokkos : public PairUMA, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairUMAKokkos(class LAMMPS *);
  ~PairUMAKokkos() override;
  void compute(int, int) override;
  void init_style() override;

  // Engine owns the neighbor graph; LAMMPS neighbor list unused for forces.
  enum { EnabledNeighFlags = FULL };
  enum { COUL_FLAG = 0 };
};

}    // namespace LAMMPS_NS

#endif
#endif
