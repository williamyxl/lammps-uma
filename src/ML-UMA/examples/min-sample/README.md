# Minimal sample: NaCl NVT with `pair_style uma`

10-step Nosé–Hoover NVT @ 300 K on perturbed rocksalt NaCl. Runtime is C++
LibTorch only: `pair_coeff` loads a TorchScript artifact directory.

## Files

| File | Purpose |
|---|---|
| `extract_uma_artifact.sh` | Export an artifact from `$UMA_CHECKPOINT` |
| `make_data.py` | Generate perturbed NaCl `data.nacl` |
| `in.nvt` | NVT input (`pair_style uma precision double`) |

## Prerequisites

- LibTorch (CUDA or XPU) and a built `lmp` with `PKG_ML-UMA=ON`.
- For export: fairchem-core plus `UMA_CHECKPOINT` pointing at `uma-s-1p2.pt`.

```bash
export UMA_CHECKPOINT=/path/to/uma-s-1p2.pt
N=6 bash extract_uma_artifact.sh
python make_data.py
lmp -in in.nvt -var artifact ./artifact_n6
```
