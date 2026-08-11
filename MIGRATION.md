# HPC migration handoff

**Written:** 2026-08-11 · **Reason:** Delta queue backlog (~2900 pending on `gpuA100x4`); moving to a faster HPC.

## What travels via git

Everything on branch `uma-kokkos-mlip` (currently `3133f3cb5f`+): source, LAMMPS input generators, SLURM scripts, docs (including `src/ML-UMA/uma-engine/docs/multinode_mpi_plan.md`, rev 4), gate scripts, and the two frozen reference structures needed for parity.

## What does NOT travel — regenerate on the new HPC

| Item | Size | Regeneration |
|---|---|---|
| `build-uma/`, `build-uma-v7/`, `build-uma-mn/` | 486M + 425M + 126M | `bash scripts/build_lammps_uma.sh` (each with the right `BUILD_DIR`/`MP_BUILD_DIR` env) |
| `src/ML-UMA/uma-engine/build-cpp-mp{,-v7,-mn}/` | ~80M | built alongside the LAMMPS builds |
| `frozen/v6_5d50357634/` | 83M | md5 snapshot of v6 binaries — recreate by rebuilding at that commit |
| `src/ML-UMA/uma-engine/artifacts/` | **63G** | re-export from the UMA checkpoint |
| `src/ML-UMA/examples/*/results/`, `logs/` | ~50M | job outputs; regenerate by rerunning |
| `src/ML-UMA/examples/nacl_nsweep/artifacts/` | 1.8G | per-N MP exports; regenerate on demand |

`.gitignore` was updated so these do not creep into future commits.

## External data assumed at the same path

| Path | Purpose | Size |
|---|---|---|
| `/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt` | UMA-S 1.2 checkpoint | 2.3 GB |

If the new HPC uses a different path, set `UMA_CHECKPOINT` or edit the runners. The checkpoint is a FairChem model downloadable from Meta/HuggingFace at the noted revision.

## Conda envs

Two envs, mutually exclusive on Delta:

| Env | Purpose | Key pins |
|---|---|---|
| `uma312` | LibTorch UMA path, all campaigns | py 3.12, torch 2.8.0+cu128, fairchem-core 2.21.0 |
| `nvalchemi312` | ALCHEMI path | py 3.12, torch 2.8.0+cu128, fairchem-core 2.21.0, nvalchemi-toolkit 0.2.0, warp-lang 1.16.0 |

The two cannot share one env: fairchem caps `torch<2.9` while `nvalchemi-toolkit-ops` floors it at `>=2.11`.

## Where things stand at migration

**Product recipe: W8nk** — `pair_style uma precision double devices N`, `UMA_USE_KOKKOS=0`, `umas_fast_pytorch`+`merge_mole`. Measured on Delta 4×A100-40GB:

| Cell | NVT Pair ms/step |
|---|---:|
| NaCl6 @2 | 161.94 |
| NaCl6 @4 | 92.10 |
| water888 @2 | 164.82 |
| water888 @4 | 95.74 |

**V7 (same-node engine) — CLOSED** at the hard ceiling: engine-controllable overhead is 1.02 of 89.75 ms = 1.1%; the other 98% is FairChem model execution. See `V7_PLAN.md`.

**Multi-node — IN PROGRESS.** Code + build done, physics gate not yet run. See `src/ML-UMA/uma-engine/docs/multinode_mpi_plan.md` (rev 4). The 2-node parity job died on Delta with `PMIX_ERR_FILE_OPEN_FAILURE`; the workaround is committed but only the new HPC can confirm it.

## Jobs still pending on Delta (do not follow)

| Job | Description | State |
|---|---|---|
| `21038701` | LibTorch mn_parity_2n (M3 gate, 8×8×8, 2 nodes) | PENDING |
| `21038856` | ALCHEMI 4-GPU nacl8 | PENDING |
| `21038857` | ALCHEMI 8-GPU nacl8 (2 nodes) | PENDING |

Resubmit on the new HPC once builds and w8 shards are back.

## Immediate resume steps on the new HPC

1. `git clone https://github.com/williamyxl/lammps-uma.git`
2. Recreate the two conda envs (above).
3. Place `uma-s-1p2.pt` and export `UMA_CHECKPOINT`.
4. Build product: `bash scripts/build_lammps_uma.sh` → `build-uma/lmp`.
5. Build multi-node: `bash scripts/build_lammps_uma_mn.sh` → `build-uma-mn/lmp`.
6. Regenerate the `w8_n4096` shards: `sbatch src/ML-UMA/uma-engine/tests/mn_w8_export.slurm`.
7. Adjust `#SBATCH --account`/`--partition` in all SLURM scripts for the new site.
8. Resume at the M3 gate: `sbatch src/ML-UMA/uma-engine/tests/mn_parity_2node.slurm` (edit the PMIx workaround if the new site's launcher differs).
