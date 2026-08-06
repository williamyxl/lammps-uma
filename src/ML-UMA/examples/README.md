# ML-UMA examples and parity reports

Test inputs, frozen geometries, and JSON reports for `pair_style uma` / `uma/kk`.

```
lammps/src/ML-UMA/examples/
  nacl_minim/           # CG minim deliverable
  nacl_n3_final_table/  # NaCl / Si / Al / Si4 rattled parity table
  si_sp/ si4_sp/ al_sp/ # single-point smoke inputs
  nacl_f64/             # FP64 parity harness
  ...
```

Roots are auto-detected by [`_repo.py`](_repo.py) (`ML-UMA/`, vendored `uma-engine/`, workspace).

```bash
# from uma-lmp root
ROOT=$PWD
ENG=$ROOT/lammps/src/ML-UMA/uma-engine
LMP=$ROOT/lammps/build-uma/lmp
VESIN=$ENG/third_party/vesin/lib
TORCH_LIB=$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda/lib64:${VESIN}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

cd $ROOT/lammps/src/ML-UMA/examples/nacl_minim
$LMP -k on g 1 -sf kk -in in.nacl_minim

cd $ROOT/lammps/src/ML-UMA/examples/nacl_n3_final_table
python run_final_table_compare.py
```

Canonical report: [`nacl_n3_final_table/final_table.json`](nacl_n3_final_table/final_table.json).
