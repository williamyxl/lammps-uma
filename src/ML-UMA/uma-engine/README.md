# uma-engine — GPU-persistent LibTorch UMA inference

Native C++/CUDA engine for FairChem UMA (`uma-s-1p2`, task **omat**).
Vendored under `lammps/src/ML-UMA/uma-engine/` and linked by package **ML-UMA**.

Produces energy and forces from positions. Neighbor list is built on **CPU then
uploaded** in v1; the TorchScript module and work tensors remain on the GPU
across MD/minim steps and reuse storage when `N` is fixed.

Forces: `forces = -dE/dpos` via `torch::autograd::grad` on a differentiable
energy TorchScript module (same idea as FairChem `compute_forces`).

## Export artifact

```bash
source /home/xyan11/miniforge3/etc/profile.d/conda.sh && conda activate uma312
cd /home/xyan11/workdir/uma-lmp
ENG=lammps/src/ML-UMA/uma-engine
PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_omat.py \
  --checkpoint /mnt/d/workdir/uma-cache/uma-s-1p2.pt \
  --output $ENG/artifacts/uma-s-1p2-omat
```

Produces `model_traced.pt` + `metadata.json`.

## Build

Usually built as part of LAMMPS (`scripts/build_lammps_uma.sh`). Standalone:

```bash
source /home/xyan11/miniforge3/etc/profile.d/conda.sh && conda activate uma312
ENG=lammps/src/ML-UMA/uma-engine
TORCH_CMAKE=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -S $ENG -B $ENG/build \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUMA_ENGINE_USE_CUDA=ON
cmake --build $ENG/build -j$(nproc)
```

## Parity

```bash
ENG=lammps/src/ML-UMA/uma-engine
PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/parity_nacl.py \
  --artifact $ENG/artifacts/uma-s-1p2-omat
./$ENG/build/uma_parity_cli $ENG/artifacts/uma-s-1p2-omat
```

## Mixed precision

| Quantity | dtype |
|----------|-------|
| Positions (engine input) | FP32 |
| Energy (model + return) | FP32 |
| Forces | FP64 |

Prefer FP32 omat artifact (`uma-s-1p2-omat`, not `*-f64`). LAMMPS tag lacks
`KOKKOS_PREC=mixed`; pair path casts explicitly.
