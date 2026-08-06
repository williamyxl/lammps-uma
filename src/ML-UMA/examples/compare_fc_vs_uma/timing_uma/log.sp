LAMMPS (22 Jul 2025 - Update 4)
KOKKOS mode with Kokkos version 4.6.2 is enabled
  will use up to 1 GPU(s) per node
package kokkos
units metal
atom_style atomic
boundary p p p
read_data data.nacl
Reading data file ...
  orthogonal box = (0 0 0) to (11.28 11.28 11.28)
  1 by 1 by 1 MPI processor grid
  reading atoms ...
  64 atoms
  read_data CPU = 0.004 seconds
pair_style uma/kk precision mixed
pair_coeff * * /home/xyan11/workdir/uma-lmp/uma-engine/artifacts/uma-s-1p2-omat Na Cl
Pair uma: loaded artifact '/home/xyan11/workdir/uma-lmp/uma-engine/artifacts/uma-s-1p2-omat' cutoff=6.000 device=cuda precision=mixed (pos/energy float32, forces float64)
newton off
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo_style custom step pe
thermo 1
dump 1 all custom 1 forces.dump id fx fy fz
dump_modify 1 sort id
run 0
WARNING: No fixes with time integration, atoms won't move
For more information see https://docs.lammps.org/err0028 (src/verlet.cpp:60)
Neighbor list info ...
  update: every = 1 steps, delay = 0 steps, check = yes
  max neighbors/atom: 2000, page size: 100000
  master list distance cutoff = 8
  ghost atom cutoff = 8
  binsize = 8, bins = 2 2 2
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair uma/kk, perpetual
      attributes: full, newton off, kokkos_device
      pair build: full/bin/kk/device
      stencil: full/bin/3d
      bin: kk/device
Per MPI rank memory allocation (min/avg/max) = 3.359 | 3.359 | 3.359 Mbytes
   Step         PotEng    
         0  -216.26805    
Loop time of 0.000114614 on 1 procs for 0 steps with 64 atoms

70.7% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 0.0001146  |            |       |100.00

Nlocal:             64 ave          64 max          64 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            665 ave         665 max         665 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         5626 ave        5626 max        5626 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5626
Ave neighs/atom = 87.90625
Neighbor list builds = 0
Dangerous builds = 0
undump 1
run 20
WARNING: No fixes with time integration, atoms won't move
For more information see https://docs.lammps.org/err0028 (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.359 | 3.359 | 3.359 Mbytes
   Step         PotEng    
         0  -216.26805    
         1  -216.26804    
         2  -216.26805    
         3  -216.26804    
         4  -216.26805    
         5  -216.26805    
         6  -216.26804    
         7  -216.26804    
         8  -216.26805    
         9  -216.26804    
        10  -216.26804    
        11  -216.26805    
        12  -216.26805    
        13  -216.26805    
        14  -216.26805    
        15  -216.26805    
        16  -216.26805    
        17  -216.26805    
        18  -216.26804    
        19  -216.26804    
        20  -216.26805    
Loop time of 0.868627 on 1 procs for 20 steps with 64 atoms

Performance: 1.989 ns/day, 12.064 hours/ns, 23.025 timesteps/s, 1.474 katom-step/s
96.2% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.86158    | 0.86158    | 0.86158    |   0.0 | 99.19
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0.0011738  | 0.0011738  | 0.0011738  |   0.0 |  0.14
Output  | 0.00026039 | 0.00026039 | 0.00026039 |   0.0 |  0.03
Modify  | 3.5742e-05 | 3.5742e-05 | 3.5742e-05 |   0.0 |  0.00
Other   |            | 0.005576   |            |       |  0.64

Nlocal:             64 ave          64 max          64 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            665 ave         665 max         665 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         5626 ave        5626 max        5626 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5626
Ave neighs/atom = 87.90625
Neighbor list builds = 0
Dangerous builds = 0
Total wall time: 0:00:04
