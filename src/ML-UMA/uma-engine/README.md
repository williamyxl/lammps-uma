# uma-engine — GPU-persistent LibTorch UMA inference

Native C++/CUDA engine for FairChem UMA (`uma-s-1p2`). Vendored under
`lammps/src/ML-UMA/uma-engine/` and linked by package **ML-UMA**.

Produces energy and forces from positions. Neighbor list is built on **CPU then
uploaded** in v1; the TorchScript module and work tensors remain on the GPU
across MD/minim steps and reuse storage when `N` is fixed.

Forces: `forces = -dE/dpos` via `torch::autograd::grad` on a differentiable
energy TorchScript module (same idea as FairChem `compute_forces`).

**Precision:** production / LAMMPS campaigns use **FP64** artifacts
(`*-f64`). Mixed / FP32 export is disabled for campaigns.

## Checkpoints vs artifacts

| Item | Path | Notes |
|------|------|-------|
| Multihead checkpoint | `workdir/uma-cache/uma-s-1p2.pt` | FairChem Hydra `.pt` (Python only) |
| LibTorch artifact | `artifacts/uma-s-1p2-<task>-f64/` | TorchScript; one energy task per dir |

LibTorch **cannot** load the multihead checkpoint directly. Export traces one
task head at a time (`omat`, `odac`, `omol`, `oc20`, …).

## Export artifact(s)

Script: [`python/export_artifact.py`](python/export_artifact.py)

```bash
source $CONDA_SETUP && conda activate uma312
ROOT=$ROOT
ENG=$ROOT/src/ML-UMA/uma-engine
CKPT=$UMA_CHECKPOINT

# Single task (omat) FP64 — default for LAMMPS getting_started
PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_artifact.py \
  --checkpoint $CKPT --dtype float64 --task omat \
  --output $ENG/artifacts/uma-s-1p2-omat-f64

# All energy task heads FP64 (oc20 oc22 oc25 omat odac omc omol)
PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_artifact.py \
  --checkpoint $CKPT --dtype float64 --all-tasks --skip-existing \
  --artifacts-root $ENG/artifacts

# Or via SLURM:
# sbatch $ROOT/src/ML-UMA/examples/getting_started/export_all_tasks_fp64.slurm
```

Each artifact directory contains `model_traced.pt` + `metadata.json`.
`pair_coeff` points at the **directory** for the task you want.

## Build

Usually built as part of LAMMPS (`scripts/build_lammps_uma.sh`). Standalone:

```bash
source $CONDA_SETUP && conda activate uma312
ENG=$ROOT/src/ML-UMA/uma-engine
TORCH_CMAKE=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -S $ENG -B $ENG/build \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUMA_ENGINE_USE_CUDA=ON
cmake --build $ENG/build -j$(nproc)
```

## Parity (omat FP64)

```bash
ENG=$ROOT/src/ML-UMA/uma-engine
PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/parity_nacl.py \
  --dtype float64 --artifact $ENG/artifacts/uma-s-1p2-omat-f64
./$ENG/build/uma_parity_cli $ENG/artifacts/uma-s-1p2-omat-f64
```

## Precision modes

| Mode | Artifact | Status |
|------|----------|--------|
| `precision double` | `uma-s-1p2-<task>-f64` | **use this** |
| `precision mixed` | `uma-s-1p2-<task>` (FP32 graph) | **disabled** for campaigns |
