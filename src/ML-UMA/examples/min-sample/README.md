# Minimal sample: NaCl NVT with FairChem UMA via native LibTorch `pair_style uma`

A minimal, end-to-end example of running the FairChem **UMA** foundation MLIP inside
LAMMPS as a native C++/LibTorch `pair_style`, in **FP64**, on **Intel XPU (Aurora)**
with native **XCCL** graph-parallel across tiles. Runs a **10-step Nosé–Hoover NVT
@ 300 K** on perturbed rocksalt NaCl and dumps first-frame per-atom forces.

No Python at runtime — a single LAMMPS binary loads a TorchScript-traced UMA
artifact and computes energy + autograd forces in C++.

## Files
| File | Purpose |
|---|---|
| `build_lammps_uma.sh` | Build LAMMPS + ML-UMA (XPU + XCCL, FP64) → `build-lmp-xccl/lmp` |
| `extract_uma_artifact.sh` | Export a LAMMPS-loadable UMA artifact from the FairChem checkpoint |
| `make_data.py` | Generate the perturbed NaCl `data.nacl` |
| `in.nvt` | Minimal LAMMPS NVT input (`pair_style uma`, run 0 + 10-step NVT) |
| `run_nvt_aurora.pbs` | ALCF Aurora PBS launcher (single tile or 12-tile GP) |

## Prerequisites
- ALCF Aurora (Intel Max GPU / XPU tiles).
- A conda env with **torch 2.13.0+xpu** (LibTorch), **FairChem**, oneCCL, and `icpx`
  on `PATH`. This example uses `hen/scripts/activate_fxpu.sh`; point `ACTIVATE` at
  your own activator if different.
- The FairChem **UMA-s-1p2** checkpoint (`UMA_CKPT`, default `hen/uma-cache/uma-s-1p2.pt`).
- The `lammps-uma` checkout (`LU`), branch with the ML-UMA package.

## Quick start (single tile, N=6 → 1,728 atoms)
```bash
cd src/ML-UMA/examples/min-sample

# 1) Build LAMMPS with the UMA pair style (~10-15 min)
bash build_lammps_uma.sh

# 2) Export the UMA artifact traced at N=6 (single tile)
N=6 bash extract_uma_artifact.sh          # -> artifact_n6/

# 3) Run the 10-step NVT on one XPU tile
qsub -v N=6 run_nvt_aurora.pbs
```
Output (`run_n6_w1/`): `log.lammps` (thermo, step 0 → step 10), `forces_step0.dump`
(first-frame per-atom forces), `lmp.log`.

## 12-tile graph-parallel (one crystal across all tiles, large N)
The single tile is 64 GiB — capped near N=18 (46,656 atoms). To run one large
crystal across all 12 tiles, export **per-rank** artifacts and launch 12 MPI ranks
(one tile each). Keep `devices 1` in `in.nvt`; the graph-parallel world = number of
MPI ranks.
```bash
# export 12 per-rank artifacts (each ~a few min; repeat for r=0..11)
for R in $(seq 0 11); do
  N=32 EXPORT_WORLD=12 EXPORT_RANK=$R ARTIFACT=$PWD/artifact_gp_n32 \
    bash extract_uma_artifact.sh
done
# run 10-step NVT on 12 tiles
qsub -v N=32,WORLD=12,ARTIFACT=$PWD/artifact_gp_n32 run_nvt_aurora.pbs
```
Verified: full 10-step NVT@300 K at N = 18, 32, 34, 36, **38 (438,976 atoms)** on
12 tiles; N=40 OOMs.

## How it works (1 paragraph)
`extract_uma_artifact.sh` traces the UMA backbone to TorchScript and re-implements
activation checkpointing in exportable form — **per message-passing block**,
**per edge-chunk**, and the **edge-degree prologue** — as custom ops (`uma_ckpt::*`)
whose recompute logic lives in C++. `pair_style uma` (in `src/ML-UMA/pair_uma.cpp`)
loads the artifact, consumes the **LAMMPS neighbor list** (O(N)), runs the traced
forward, and computes forces via `torch::autograd::grad` (FP64). Multi-tile uses
the `uma_peer::*` collectives over **XCCL/oneCCL** (Intel) — a `-D
UMA_ENGINE_USE_CUDA=ON` build uses **NCCL** on NVIDIA instead.

## Notes / caveats
- **Artifacts are traced per cell size N** (the activation-checkpoint chunk count is
  baked at trace time). Export one artifact per N you plan to run.
- **`precision double`** = FP64 (matches the FairChem reference to ~1e-9 meV/atom;
  per-atom forces to the FP64 floor).
- **Element order**: `pair_coeff * * <artifact> Na Cl` — the trailing element list
  maps LAMMPS atom types (1,2) to species. Adjust for other systems.
- **Orthorhombic cells** for the fast neighbor path (the NaCl example is cubic).
- The single-tile N=18 NVT is memory-tight for the traced path; use 12 tiles for
  N ≥ 18 NVT. A rare chunk-count drift under MD (seen once at N=24) is a known,
  fixable limitation (edge-padding to a fixed chunk multiple).

## Validation reference
See `../../uma-engine/docs/REVIEW_libtorch_uma_xpu.md` and
`../../../../docs/REPORT_2path_nvt_comparison.md` for full parity vs the ASE
FairChem API (energy, per-atom forces, AG=FD) and scaling/timing results.
