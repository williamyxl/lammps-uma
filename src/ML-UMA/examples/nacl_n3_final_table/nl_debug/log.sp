LAMMPS (22 Jul 2025 - Update 4)
KOKKOS mode with Kokkos version 4.6.2 is enabled
  will use up to 1 GPU(s) per node
package kokkos
units metal
atom_style atomic
boundary p p p
read_data data.nacl
Reading data file ...
  orthogonal box = (0 0 0) to (17.0892 17.0892 17.0892)
  1 by 1 by 1 MPI processor grid
  reading atoms ...
  216 atoms
  read_data CPU = 0.005 seconds
pair_style uma/kk precision double
pair_coeff * * /home/xyan11/workdir/uma-lmp/uma-engine/artifacts/uma-s-1p2-omat-f64 Na Cl
Pair uma: loaded artifact '/home/xyan11/workdir/uma-lmp/uma-engine/artifacts/uma-s-1p2-omat-f64' cutoff=6.000 device=cuda precision=double (pos/energy float64, forces float64)
newton off
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo 1
thermo_style custom step pe
run 0
WARNING: No fixes with time integration, atoms won't move
For more information see https://docs.lammps.org/err0028 (src/verlet.cpp:60)
Neighbor list info ...
  update: every = 1 steps, delay = 0 steps, check = yes
  max neighbors/atom: 2000, page size: 100000
  master list distance cutoff = 8
  ghost atom cutoff = 8
  binsize = 8, bins = 3 3 3
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair uma/kk, perpetual
      attributes: full, newton off, kokkos_device
      pair build: full/bin/kk/device
      stencil: full/bin/3d
      bin: kk/device
Per MPI rank memory allocation (min/avg/max) = 3.359 | 3.359 | 3.359 Mbytes
   Step         PotEng    
         0  -731.78156    
Loop time of 0.000211295 on 1 procs for 0 steps with 216 atoms

69.6% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 0.0002113  |            |       |100.00

Nlocal:            216 ave         216 max         216 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1115 ave        1115 max        1115 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:        17280 ave       17280 max       17280 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 17280
Ave neighs/atom = 80
Neighbor list builds = 0
Dangerous builds = 0
print "Final PE = $(pe)"
Final PE = -731.78155517578125
Total wall time: 0:00:04
