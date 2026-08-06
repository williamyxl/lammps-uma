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

#include <string>
#include <vector>

namespace uma {
class Predictor;
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
  double init_one(int, int) override;

 protected:
  virtual void allocate();
  void load_predictor();

  std::string artifact_dir;
  int *map;    // map type -> atomic number (0 unused)
  double cutoff;
  int precision;    // PRECISION_MIXED or PRECISION_DOUBLE

  uma::Predictor *predictor;
  std::vector<float> pos_buf;
  std::vector<double> pos_buf_d;
  std::vector<int> z_buf;
  std::vector<double> force_buf;
  double cell_buf[9];
  int pbc_buf[3];
};

}    // namespace LAMMPS_NS

#endif
#endif
