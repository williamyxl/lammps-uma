# ML-UMA examples and parity reports

Test inputs, frozen geometries, and JSON reports for `pair_style uma` / `uma/kk`.

```
src/ML-UMA/examples/
  delta_parity/         # Delta A100 4-path suite (ASE / FC-lmp / uma double+mixed)
  multi_gpu_nacl6/      # Delta multi-GPU NaCl6 parity (ASE workers / FC / uma Kokkos g N)
  nacl_minim/           # CG minim deliverable
  nacl_n3_final_table/  # prior Titan-V NaCl / Si / Al / Si4 table
  si_sp/ si4_sp/ al_sp/ # single-point smoke inputs
  nacl_f64/             # FP64 parity harness
  ...
```

Roots are auto-detected by [`_repo.py`](_repo.py). On Delta the clone **is** the LAMMPS tree (`lammps-uma/`).

## Delta A100 (preferred)

```bash
sbatch src/ML-UMA/examples/delta_parity/run_parity.slurm
```

See [`delta_parity/README.md`](delta_parity/README.md).

## Local smoke (after `build-uma/lmp`)

```bash
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=$ROOT/src/ML-UMA/uma-engine
LMP=$ROOT/build-uma/lmp
VESIN=$ENG/third_party/vesin/lib
TORCH_LIB=$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
export LD_LIBRARY_PATH="${VESIN}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

cd $ROOT/src/ML-UMA/examples/nacl_minim
$LMP -k on g 1 -sf kk -in in.nacl_minim
```

Prior Titan-V report: [`nacl_n3_final_table/final_table.json`](nacl_n3_final_table/final_table.json).
