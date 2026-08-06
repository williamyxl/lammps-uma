# NaCl 2×2×2 UMA energy minimization (Kokkos / CUDA)

Perturbed rocksalt (seed 0, σ=0.05), matching ASE refs under uma-lmp `refs/`.

## Files
- `data.nacl` — LAMMPS data
- `in.nacl_minim` — minimize + dump
- `run_minim.sh` — launch helper
- `nacl_init.npz` — same geometry as data

## Run

```bash
# from uma-lmp root, after building lammps/build-uma
./lammps/src/ML-UMA/examples/nacl_minim/run_minim.sh
```

Or manually:

```bash
source /home/xyan11/miniforge3/etc/profile.d/conda.sh && conda activate uma312
ROOT=$PWD
export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"
$ROOT/lammps/build-uma/lmp -k on g 1 -sf kk -in $ROOT/lammps/src/ML-UMA/examples/nacl_minim/in.nacl_minim
```

ASE ground truth: `refs/ase_nacl_minim.npz` (E ≈ −216.76 eV).

## Precision
Positions FP32 → engine; energy FP32; forces FP64 written back to Kokkos.
