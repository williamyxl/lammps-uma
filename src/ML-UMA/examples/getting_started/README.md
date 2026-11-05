# Getting started: LAMMPS + UMA (`pair_style uma`)

Short path from this tree to 10 NVT steps of NaCl with UMA on one GPU/XPU.

**Precision:** FP64 only (`pair_style uma precision double`).

| File | What it does |
|------|----------------|
| [`in.nacl5_nvt`](in.nacl5_nvt) | NaCl 5×5×5 rocksalt, NVT @ 300 K, 10 steps |
| this README | How to build and run |

Export tool: `../../uma-engine/python/export_artifact.py`

## Before you start

1. A GPU or XPU with LibTorch.
2. A Python env with PyTorch (and fairchem-core only if you need to **export**).
3. Either a pre-exported artifact directory, or `UMA_CHECKPOINT` pointing at
   `uma-s-1p2.pt` (FairChem multihead checkpoint).

No extra data file is required: the input builds the crystal with LAMMPS
`lattice` / `create_atoms`.

## Build

```bash
./scripts/build_lammps_uma.sh
```

Export the FP64 omat artifact if it is not already present:

```bash
ENG=$PWD/src/ML-UMA/uma-engine
PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_artifact.py \
  --checkpoint "$UMA_CHECKPOINT" --dtype float64 --task omat \
  --output $ENG/artifacts/uma-s-1p2-omat-f64
```

`pair_coeff` points at that **directory**.

## Run

From this folder, after `scripts/build_lammps_uma.sh`:

```bash
../../../../build-uma/lmp -in in.nacl5_nvt
```

Uses `pair_style uma precision double`. Do not pass `-k on` or `-sf kk`.
