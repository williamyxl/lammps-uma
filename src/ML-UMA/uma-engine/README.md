# uma-engine — GPU-persistent LibTorch UMA inference

Native C++ engine for FairChem UMA, vendored under
`src/ML-UMA/uma-engine/` and linked by package **ML-UMA**.

Produces energy and forces from positions. Forces are
`forces = -dE/dpos` via autograd on a TorchScript energy module.

**Precision:** LAMMPS `pair_style uma` is FP64-only (`*-f64` artifacts).
`precision mixed` is refused.

## Checkpoints vs artifacts

LibTorch cannot load a FairChem multihead Hydra `.pt` checkpoint.
Export traces one energy task head (`omat`, `odac`, `omol`, …) into an
artifact directory (`model_traced.pt` + `metadata.json`). `pair_coeff`
names that directory.

```bash
python src/ML-UMA/uma-engine/python/export_artifact.py \
  --checkpoint $UMA_CHECKPOINT --dtype float64 --task omat \
  --output /path/to/uma-s-1p2-omat-f64
```

Do not commit checkpoints or artifacts (FAIR Chemistry License).

## Build

Usually built as part of LAMMPS (`-D PKG_ML-UMA=on`). Standalone:

```bash
TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUMA_ENGINE_USE_CUDA=ON
cmake --build build
```

Intel GPU: `-DUMA_ENGINE_USE_XPU=ON` (disables CUDA). See
`cmake/Modules/Packages/ML-UMA.cmake` in the LAMMPS tree.
