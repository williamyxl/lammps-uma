# ML-UMA examples and parity reports

Test inputs, frozen geometries, and JSON reports for `pair_style uma` / `uma/kk`.

```
src/ML-UMA/examples/
  getting_started/      # Build + short NaCl 5x5x5 NVT (FP64; start here)
  multi_gpu_nacl6/      # Delta multi-GPU NaCl6 (ASE / FC / uma double; mixed disabled)
  multi_node_nacl6/     # Multi-node MPI campaign (ase / fc / uma_double)
  nacl_n3_final_table/  # prior Titan-V NaCl / Si / Al / Si4 table
  si_sp/ si4_sp/ al_sp/ # single-point smoke inputs
  nacl_f64/             # legacy NaCl SP inputs
  ...
```

On Delta the clone **is** the LAMMPS tree (`lammps-uma/`).

## First run (recommended)

See [`getting_started/README.md`](getting_started/README.md):

```bash
sbatch src/ML-UMA/examples/getting_started/build_uma.slurm
# then run in.nacl5_nvt as described in that README
```

## Multi-GPU NaCl6 suite

```bash
cd src/ML-UMA/examples/multi_gpu_nacl6 && ./submit_path_jobs.sh
```

See [`multi_gpu_nacl6/README.md`](multi_gpu_nacl6/README.md).

## Local smoke (after `build-uma/lmp`)

```bash
ROOT=$ROOT
ENG=$ROOT/src/ML-UMA/uma-engine
LMP=$ROOT/build-uma/lmp
VESIN=$ENG/third_party/vesin/lib
TORCH_LIB=$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
export LD_LIBRARY_PATH="${VESIN}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

cd $ROOT/src/ML-UMA/examples/getting_started
$LMP -k on g 1 -sf kk -in in.nacl5_nvt
```

Prior Titan-V report: [`nacl_n3_final_table/final_table.json`](nacl_n3_final_table/final_table.json).
