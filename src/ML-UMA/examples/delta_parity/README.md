# Delta A100 — 4-path UMA parity

Single-point **total energy** + **per-atom forces** on rattled crystals, comparing:

| Path | Binary / API | Precision |
|------|----------------|-----------|
| ASE FairChem | `FAIRChemCalculator` | FP64 |
| FairChem LAMMPS fix external | `/u/xyan11/miniforge3-x86_64/envs/uma312/bin/lmp` | FP64 predict; cell built FP32 in `lammps_fc` |
| `pair_style uma/kk` | local `build-uma/lmp` | `precision double` |
| `pair_style uma/kk` | local `build-uma/lmp` | `precision mixed` |

## Systems

Frozen under `structures/` (Unif[-0.10,+0.10] Å atomic rattle only, seed=0; no lattice scale):

- NaCl rocksalt **3×3×3** (216 atoms)
- Al FCC **3×3×3** (108 atoms)
- Si diamond **3×3×3** (216 atoms)
- Si diamond **4×4×4** (512 atoms)

Energy is **total only** (scalar). Forces are **per-atom**. Saved in `results/<sys>/per_atom_forces.npz`.

## Submit (recommended)

```bash
sbatch /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/delta_parity/run_parity.slurm
```

The SLURM script vendors vesin, exports TorchScript artifacts if missing, builds `build-uma/lmp`, freezes structures, then runs `run_parity.py`.

Subset / skip flags:

```bash
sbatch --export=ALL,ONLY_SYSTEM=nacl,SKIP_EXPORT=1,SKIP_BUILD=1 \
  .../delta_parity/run_parity.slurm
```

## Manual

```bash
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh && conda activate uma312
module load cuda/12.8 cmake/3.31.8

# once
bash $ROOT/src/ML-UMA/uma-engine/scripts/vendor_vesin_for_arch.sh
PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_omat.py \
  --checkpoint /work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt \
  --dtype float32 --output $ENG/artifacts/uma-s-1p2-omat
PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_omat.py \
  --checkpoint /work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt \
  --dtype float64 --output $ENG/artifacts/uma-s-1p2-omat-f64
bash $ROOT/scripts/build_lammps_uma.sh

python freeze_structures.py
ONLY_SYSTEM=all python run_parity.py
```

## Outputs

```
results/
  parity_table.json              # machine-readable
  PARITY_REPORT.md               # markdown tables (energy, forces, timing)
  uma-delta-parity.canvas.tsx    # Cursor canvas (also mirrored to IDE canvases/)
  nacl|al|si|si4/
    per_atom_forces.npz          # total E_* + forces_* for all paths
    *_block.json
    fc/  uma_double/  uma_mixed/ # LAMMPS inputs + force dumps
```

Reports are written automatically at the end of `run_parity.py` (and again by the SLURM script via `write_reports.py`).