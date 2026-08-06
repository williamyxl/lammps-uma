# Si / Al / Si4 uma/kk single-point examples

Companion systems for covalent (Si diamond) and metallic (Al fcc) bonding.
Geometries match the final compare table (lattice × 1.01 + rattle).

| Dir | System | Atoms | Input |
|-----|--------|------:|-------|
| `si_sp/` | Si diamond 3×3×3 | 216 | `in.si_sp` |
| `si4_sp/` | Si diamond 4×4×4 | 512 | `in.si4_sp` |
| `al_sp/` | Al fcc 3×3×3 | 108 | `in.al_sp` |

```bash
source /home/xyan11/miniforge3/etc/profile.d/conda.sh && conda activate uma312
ROOT=/home/xyan11/workdir/uma-lmp
VESIN=$ROOT/lammps/src/ML-UMA/uma-engine/third_party/vesin/lib
TORCH_LIB=$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda/lib64:${VESIN}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
LMP=$ROOT/lammps/build-uma/lmp
EX=$ROOT/lammps/src/ML-UMA/examples

cd $EX/si_sp && $LMP -k on g 1 -sf kk -in in.si_sp
cd $EX/si4_sp && $LMP -k on g 1 -sf kk -in in.si4_sp
cd $EX/al_sp && $LMP -k on g 1 -sf kk -in in.al_sp
```

Parity: `../nacl_n3_final_table/final_table.json`.
For FP64 energy matching ASE, use `precision double` and artifact `uma-s-1p2-omat-f64`.
