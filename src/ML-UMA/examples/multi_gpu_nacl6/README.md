# Multi-GPU NaCl 6×6×6 parity (Delta A100)

Three-path energy/force parity on the **frozen** NaCl 6×6×6 rattled crystal
(1728 atoms), varying `NGPUS` ∈ {1, 2, 4}.

Paths: ASE FairChem FP64 · FairChem LAMMPS fix-external · `uma/kk` **double**
(FP64). **`uma/kk` mixed is disabled** (do not submit / do not report as active).

## Latest results

Canonical write-up: **[`results/RESULTS.md`](results/RESULTS.md)**  
Also: `results/SUMMARY.md`, `results/MULTIGPU_REPORT.md`, `results/SUMMARY.json`.

| Path | 1 / 2 / 4 GPU (ms) | \|ΔE\| vs ASE | max\|ΔF\| vs ASE | Notes |
|------|-------------------|-------------:|-----------------:|-------|
| ASE FairChem FP64 | 396.5 / 193.9 / 115.2 | ~0 | ~0 | Ray — reference |
| FairChem FC LAMMPS | 345.5 / 193.2 / 118.0 | ≈4.9×10⁻⁶ | ≈7.1×10⁻⁶ | Ray — reference |
| **uma/kk double (product)** | **321.5 / 190.8 / 140.9** | **≈1.2×10⁻¹⁰** | **≈5.0×10⁻⁷** | P2 job `20933393` · beats ASE/FC @2 |

Full tables: [`results/RESULTS.md`](results/RESULTS.md) § Three-path · [`results/MULTIGPU_REPORT.md`](results/MULTIGPU_REPORT.md). Canvas: `nacl6-multigpu-results` (refreshed each tick).

## Geometry contract (DO NOT regenerate)

Always load:

```
structures/nacl6_rattle_fixed.extxyz
```

Manifest: `structures/nacl6_rattle_fixed.manifest.json`. Same as prior
`structure_nacl6_rattle.npz` (rocksalt a=5.64 Å, Unif[-0.1,0.1] Å, seed=0).
Coordinates written with 12 significant digits (`.12g`). **Never re-rattle.**

`load_geometry.py` asserts `natoms == 1728`.

## FP64 policy

- **ASE** and **uma/kk double**: FP64 only (`inference_settings_with_dtype("float64")`
  / `uma-s-1p2-omat-f64`). Do **not** use FairChem **`InferenceSettings` turbo**
  for the product parity recipe below.
- **Speed-path export** (Tier1+): `execution_mode=umas_fast_pytorch` and
  `merge_mole=True` → artifact `*-f64-fast`. That is **not** the FairChem
  `InferenceSettings` turbo preset. Knob table:
  [`agent_stamps/cpp_libtorch/perf_campaign/GLOSSARY.md`](agent_stamps/cpp_libtorch/perf_campaign/GLOSSARY.md).
- **`uma/kk` mixed: DISABLED** — FP32 energy graph with upcast forces; not an
  active path. Historical mixed rows in `results/` are HTML-commented.
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

## Timing policy (SLURM wall only)

Reported `ms_per_eval` is **only** from the SLURM script wall clock around
`python run_multigpu.py`:

```text
ms_per_eval = 1000 * slurm_wall_s / N_TIMING
```

- Timed region: `run_multigpu.py` alone (load + evals + path teardown).
- Not timed: module load, rebuild, `collect_results`, report writers.
- Python CUDA-loop / LAMMPS Pair timers are kept as `ms_per_eval_python` for
  debug only and are **not** used in reports/canvas.
- Artifact: `results/.../timing_slurm.json` + `parity.json` → `timing_policy`.

## VRAM isolation (required)

**One path per SLURM job.** Running ASE + FC + uma in the same allocation
leaves Ray/NCCL/Kokkos/LibTorch state on the GPUs and skews later timings.

| Do | Don't |
|----|-------|
| `sbatch run_ngpu2_ase.slurm` | `ONLY_PATHS=ase,fc,uma_double` in one job |
| `./submit_path_jobs.sh` | deprecated `run_ngpu{1,2,4}.slurm` multi-path |
| `./submit_path_jobs.sh --gp` | `ONLY_PATHS=uma_mixed` or any mixed path |

Scripts: `run_ngpu{N}_{ase,fc,uma_double}.slurm` and
`gp_round/run_ngpu{N}_uma_double.slurm` (regenerate with
`./generate_path_jobs.sh`). **`uma_mixed` scripts may still exist on disk but
must not be submitted.** Escape hatch: `ALLOW_MULTI_PATH=1` (login debug only).

## How to run

### Preferred — path-isolated submit

```bash
cd /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6

# ASE / FC / uma_double × 1/2/4 GPUs (chained afterok)
./submit_path_jobs.sh

# gp_round uma double only
RECOMPILE=1 ./submit_path_jobs.sh --gp

# ASE+FC @4 only
./submit_path_jobs.sh --paths ase,fc --ngpus 4
```

### gp_round (graph-parallel `devices N` campaign)

```bash
./gp_round/rebuild_and_submit.sh            # dry-run
./gp_round/rebuild_and_submit.sh --submit   # rebuild + uma_double jobs
./gp_round/rebuild_and_submit.sh --submit --ngpu4
```

Defaults: **`uma_double` only** (mixed disabled), `RECOMPILE=1` on rebuild
step, results under `results/gp_round/ngpu{N}/`. See
[`gp_round/DRY_RUN_CHECKLIST.md`](gp_round/DRY_RUN_CHECKLIST.md).

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
export ONLY_PATHS=uma_double          # ONE path
export MERGE_RESULTS=1
export RESULTS_DIR=src/ML-UMA/examples/multi_gpu_nacl6/results/gp_round/ngpu2

python src/ML-UMA/examples/multi_gpu_nacl6/run_multigpu.py
```

Outputs under `results/ngpu${NGPUS}/` (suite) or `results/gp_round/ngpu${NGPUS}/`:

| File | Content |
|------|---------|
| `parity.json` | Energies, ms/eval, force errors vs ASE (merged across path jobs) |
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
| Artifact FP64 (omat) | `uma-engine/artifacts/uma-s-1p2-omat-f64` |
| Other task FP64 | `uma-engine/artifacts/uma-s-1p2-<task>-f64` (via `export_artifact.py --all-tasks`) |
| Artifact mixed | ~~`uma-s-1p2-omat`~~ **disabled** |
| Export script | `uma-engine/python/export_artifact.py` |
| LMP_UMA | `build-uma/lmp` |
| LMP_FC | `.../envs/uma312/bin/lmp` |

See also [`MULTI_GPU_API_NOTES.md`](MULTI_GPU_API_NOTES.md).
