# Multi-node UMA — MPI + Kokkos (no Ray)

> **SUPERSEDED (2026-08-09)** by [`multinode_mpi_plan.md`](multinode_mpi_plan.md).
> The v1 all-gather sketch below is still valid as a parity-only oracle, but **v2
> as written is wrong**: a 6–12 Å ghost shell cannot be exact for a 4-layer
> model (receptive field 24 Å), and this doc omits UMA-S's per-layer global
> `balance_channels` reduction and MOLE global-composition dependency. Kept for
> the Phase-G geometry/OOM record only.

**Status:** Phase G complete (N\* = 10); Phase 0 never started  
**Suite:** `src/ML-UMA/examples/multi_node_nacl6/`  
**Plan / session handoff:** `.cursor/plans/multi-node_mpi_uma_c20e7566.plan.md` (read **Session handoff** first)

## Locked design

| Layer | Role |
|-------|------|
| MPI | Multi-node / multi-rank domain decomposition |
| Kokkos | On-node acceleration **per MPI rank** |
| UMA MLIP | Traced LibTorch **`devices=1`** only |

**Forbidden in UMA multi-node:** Ray, `GraphParallelRuntime`, `uma_gp_worker`, `nlocal`-only multi-rank without gather/halo.

ASE FairChem FP64 and FairChem FC LAMMPS multi-node are **prior-art baselines** (FairChem-native stack OK). They do not define the UMA product path.

## Node-count gate

```text
Phase G (geom) → 2-node all paths COMPLETED → 4-node → 8-node
```

Do not open 4- or 8-node until every path (`ase`, `fc`, `uma_double`) finishes the 2-node campaign.
(**`uma_mixed` disabled** — FP64 only for UMA.)


## Dual binaries

| Tree | `BUILD_MPI` | Use |
|------|-------------|-----|
| `build-uma/lmp` | OFF | Same-node Ray GP, Phase G OOM sweep |
| `build-uma-mpi/lmp` | ON | Multi-rank / multi-node traced `devices=1` |

Scripts must hardcode `LMP_UMA`. Refuse UMA MPI jobs if Ray / `UMA_DEVICES>1` is set.

## Force / energy ownership

| Quantity | v1 all-gather | v2 halo |
|----------|---------------|---------|
| Ordering | atom **tag** | locals + ghosts |
| Forces | write **`nlocal` only** | write **`nlocal` only** |
| Energy | **rank 0** full-system PE; other ranks 0 | rule must match serial PE |
| Newton | `newton pair off` | same |

v1 = **parity-only** (no speedup tables). v2 halo required before any multi-node UMA speedup claim. Start `comm_modify cutoff` ≥ 6 Å; bump (e.g. 12 Å) if parity vs ASE FP64@1 fails.

## Results I/O (no merge races)

| When | What |
|------|------|
| Each SLURM job | Write **only** its own dir, e.g. `results/geom_sweep/N08/ase/` or `results/nnodes2/uma_double/` |
| After wave | Explicit merge → SUMMARY / reports / canvas |
| Submit | Parallel OK within a wave; **no `afterok` for I/O** |

## Timing

SLURM wall of the timed `run_*` region is the sole reported `ms_per_eval` (`1000 * wall_s / N_TIMING`), matching the multi_gpu_nacl6 policy.

## Oracle

ASE FairChem FP64, `workers=1`, no ParallelMLIP, on the **frozen Phase-G geometry only**. Do not reuse the nacl6 (1728-atom) oracle.

## Phase G (geometry)

1. OOM sweep N×N×N @ 4×A100, all four paths → **N\*** = max N all pass.
2. Rattle δ=0.1 Å seed=0 → freeze `nacl{N*}_rattle_fixed.extxyz` + manifest.
3. Record ASE FP64@1 oracle on that file.

Atom count: rocksalt conventional cell × (N,N,N) → **`natoms = 8·N³`**.

## Delta SLURM (UMA MPI)

- Binary: `build-uma-mpi/lmp`
- `srun -n R --gpus-per-task=1` (1 GPU per rank)
- `pair_style uma/kk precision double devices 1`
- Frozen `nacl{N*}_rattle_fixed.extxyz`
- Gates vs Phase-G ASE FP64@1 (+ engineering vs 1-rank MPI)
