units metal
atom_style atomic
boundary p p p
read_data data.nacl
pair_style uma/kk precision double
pair_coeff * * /home/xyan11/workdir/uma-lmp/lammps/src/ML-UMA/uma-engine/artifacts/uma-s-1p2-omat-f64 Na Cl
newton off
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo 1
thermo_style custom step pe
run 0
print "Final PE = $(pe)"
