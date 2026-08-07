# Multi-GPU NaCl 6×6×6 parity (Delta A100)

Four-path energy/force parity on the **frozen** NaCl 6×6×6 rattled crystal
(1728 atoms), varying `NGPUS` ∈ {1, 2, 4}.

Paths: ASE FairChem FP64 · FairChem LAMMPS fix-external · `uma/kk` double · `uma/kk` mixed.

## Latest results

Canonical write-up: **[`results/RESULTS.md`](results/RESULTS.md)**  
Also: `results/SUMMARY.md`, `results/MULTIGPU_REPORT.md`, `results/COORD_ANALYSIS.md`.

## Geometry contract (DO NOT regenerate)

Always load:

```
../delta_parity/structures/nacl6_rattle_fixed.extxyz
```

Manifest: `nacl6_rattle_fixed.manifest.json`. Same as prior
`structure_nacl6_rattle.npz` (rocksalt a=5.64 Å, Unif[-0.1,0.1] Å, seed=0).
Coordinates written with 12 significant digits (`.12g`). **Never re-rattle.**

`load_geometry.py` asserts `natoms == 1728`.

## FP64 policy

- **ASE** and **uma/kk double**: FP64 only (`inference_settings_with_dtype("float64")`
  / `uma-s-1p2-omat-f64`). Do **not** use turbo settings for this campaign.
- **uma/kk mixed**: explicit separate path (`uma-s-1p2-omat`).
- FairChem FC may build the cell in FP32 inside `lammps_fc` (documented in results).

## Multi-GPU recipes

### 1. ASE FairChem (`workers=N`)

```python
from fairchem.core.units.mlip_unit import load_predict_unit
predictor = load_predict_unit(
    ckpt, device="cuda", inference_settings=fp64_settings, workers=NGPUS
)
```

- `workers=1` → `MLIPPredictUnit` (single GPU).
- `workers>1` → `ParallelMLIPPredictUnit` (Ray + NCCL graph-parallel).
- Requires `fairchem-core[extras]` / Ray (`uma312` has this).

### 2. FairChem LAMMPS fix-external

Same predictor with `workers=NGPUS`, then:

```python
from fairchem.lammps.lammps_fc import run_lammps_with_fairchem
lmp = run_lammps_with_fairchem(predictor, inp, "omat")
```

LAMMPS is a **single** Python process; multi-GPU is Ray workers inside the predictor.

### 3. uma/kk — single MPI rank + Kokkos `g N` + `devices N`

```bash
# NGPUS=1 — devices=1 baseline
pair_style uma/kk precision double devices 1
lmp -k on g 1 -sf kk -in in.sp

# NGPUS=2 or 4 (same node; NO mpirun) — graph-parallel UMA inference
pair_style uma/kk precision double devices ${NGPUS}
lmp -k on g ${NGPUS} -sf kk -in in.sp
```

Helper: [`launch_uma_kk.sh`](launch_uma_kk.sh) (`NGPUS`, `UMA_DEVICES` env).

`UMA_DEVICES` defaults to `NGPUS` and is written into `pair_style` by
`run_multigpu.py`. Set `UMA_DEVICES=1` explicitly to force single-device UMA
while Kokkos still sees `g N` (debug only).

**Do not** use `mpirun -np NGPUS` / domain decomposition. `pair_uma.cpp` keeps:

```cpp
if (comm->nprocs > 1)
  error->all(FLERR, "Pair style uma currently supports a single MPI rank");
```

#### Graph-parallel scope (`devices N`)

| Layer | Role |
|-------|------|
| Kokkos `-k on g N` | Same-node package init (`--ntasks=1`). |
| `pair_style uma/kk ... devices N` | Shards UMA graph inference across N GPUs (NCCL). |
| Parity gate | `devices=N` vs `devices=1` @ same precision (not vs ASE). |

Thresholds (vs devices=1):

| Mode | \|ΔE\| | max \|ΔF\| | cosine |
|------|--------|------------|--------|
| double | ≲ 1e-8 eV | ≲ 1e-6 eV/Å | ≥ 1−1e-12 |
| mixed | ≲ 1e-4 eV | ≲ 1e-5 eV/Å | ≥ 1−1e-10 |

ASE/FC remain optional oracle paths (`workers=N`).

Local `build-uma/lmp` is configured with `BUILD_MPI=OFF`
([`scripts/build_lammps_uma.sh`](../../../../scripts/build_lammps_uma.sh)),
which matches this single-rank recipe.

#### Delta `gpuA100x4` binding

```bash
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4   # match NGPUS
```

SLURM sets `CUDA_VISIBLE_DEVICES` to the allocated GPUs. Do not remap unless
debugging. Kokkos `g N` counts devices in the visible set (so request
`--gpus-per-node=NGPUS`).

## How to run

### gp_round (graph-parallel `devices N` campaign)

```bash
cd /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6

# Dry-run checklist (no sbatch)
./gp_round/rebuild_and_submit.sh

# After WRITE lands devices N + rebuild:
./gp_round/rebuild_and_submit.sh --submit
```

Defaults: `ONLY_PATHS=uma_double,uma_mixed`, `RECOMPILE=1`, results under
`results/gp_round/ngpu{N}/`. See [`gp_round/DRY_RUN_CHECKLIST.md`](gp_round/DRY_RUN_CHECKLIST.md).

### Manual / dev (login or interactive GPU)

```bash
cd /work/nvme/bfzx/xyan11/workdir/lammps-uma
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312
module unload cudatoolkit 2>/dev/null || true
module load cuda/12.8

export NGPUS=2
export UMA_DEVICES=2
export N_TIMING=5
export ONLY_PATHS=uma_double,uma_mixed
export RESULTS_DIR=src/ML-UMA/examples/multi_gpu_nacl6/results/gp_round/ngpu2

python src/ML-UMA/examples/multi_gpu_nacl6/run_multigpu.py
```

Outputs under `results/ngpu${NGPUS}/`:

| File | Content |
|------|---------|
| `parity.json` | Energies, ms/eval, force errors vs ASE, multi-GPU notes |
| `forces.npz` | Per-atom forces + energies for each path |
| `run.log` | Human-readable log |

## Rebuild

When WRITE lands `pair_style uma/kk devices N` (C++ / engine changes):

```bash
bash scripts/build_lammps_uma.sh
```

gp_round SLURM sets `RECOMPILE=1` by default; `rebuild_and_submit.sh` rebuilds
before `sbatch`.

## Paths / defaults

| Item | Default |
|------|---------|
| Checkpoint | `/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt` |
| Artifact FP64 | `uma-engine/artifacts/uma-s-1p2-omat-f64` |
| Artifact mixed | `uma-engine/artifacts/uma-s-1p2-omat` |
| LMP_UMA | `build-uma/lmp` |
| LMP_FC | `.../envs/uma312/bin/lmp` |

See also [`MULTI_GPU_API_NOTES.md`](MULTI_GPU_API_NOTES.md).
