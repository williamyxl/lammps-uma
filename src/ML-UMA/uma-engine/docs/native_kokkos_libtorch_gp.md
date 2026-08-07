# Native Kokkos + LibTorch multi-GPU (no Ray)

**Stamp:** 2026-08-07 ~18:20 CDT  
**Plan:** `.cursor/plans/libtorch_multi-gpu_kokkos_20de0136.plan.md`  
**Agent stamps:** `examples/multi_gpu_nacl6/agent_stamps/`

## Goal

Same-node `pair_style uma/kk precision double devices N` with:

- `--ntasks=1`, `lmp -k on g N -sf kk` on Delta **`gpuA100x4`**
- Full scientific UMA (no smaller cutoff / fewer neighbors / skipped layers)
- **Vesin** full NL → FairChem partition **key** → LibTorch shards
- Inter-GPU traffic via **Kokkos** (peer `deep_copy` / fences) — not MPI, not Ray, not c10d-torchrun
- Gates: energy + per-atom forces vs ASE FP64@1 oracle; SLURM timing like jobs `20910346/50/54`
- Land **devices=2** before **devices=4**

## Why the morning plan “succeeded” but failed

[`uma_kk_multi-gpu_23f22729`](../../../../.cursor/plans/uma_kk_multi-gpu_23f22729.plan.md) marked todos done after shipping **FairChem Ray** behind `GraphParallelRuntime`. That matched ASE/FC timing because it *was* FairChem GP. Native LibTorch+Kokkos was never built.

## Current vs target load path

| `devices` | **Legacy (today)** | **Target (this plan)** |
|-----------|--------------------|------------------------|
| `1` | TorchScript `model_traced.pt` + vesin CUDA NL | unchanged |
| `N>1` | Fork `uma_gp_worker.py` → FairChem Ray `workers=N`; launch drops Kokkos | Eager/non-opaque LibTorch shards + vesin full graph + Kokkos peer reduce; **keep** `uma/kk` + `-k on g N` |

## Phase 0b — BLOCKED (escalation)

**Fact:** LibTorch C++ cannot load `uma-cache/uma-s-1p2.pt` (FairChem Hydra `MLIPInferenceCheckpoint`). Only `artifacts/*/model_traced.pt` loads via `torch::jit::load`. Traced modules are **opaque** — no mid-forward atom-shard collectives.

**Forbidden pivots:** FairChem Ray, smaller NL, N× independent traced forwards on atom subsets (wrong MP physics).

**Unblock options (COORD must pick; WRITE must not silently choose Ray):**

1. **Preferred research:** Export a non-opaque FP64 module (`torch.export` / AOTInductor / layer-wise scriptable graph) that accepts engine vesin graphs and can host Kokkos/LibTorch peer allreduces at FairChem `gp_utils` sites.
2. **Heavy lift:** C++ reimplementation of UMA MP + load `state_dict` tensors from a new export sidecar.
3. **Stop:** Keep `devices=1` traced only until (1) or (2) lands; multi-node stays `devices=1` + halo.

Single-GPU traced vs ASE FP64@1 is already green on NaCl6 (campaign RESULTS). That does **not** satisfy Phase 0b for multi-GPU.

Env guard for development: set `UMA_FORBID_RAY_GP=1` so `devices>1` throws instead of forking Ray (see `predictor.cpp`).

## Phase 0c — Vesin strategy (default)

1. Build **one** full graph with existing `vesin_nl::vesin_build_graph_cuda` + FairChem edge flip + `max_neighbors=300` (same as `devices=1`).
2. Partition: `node_partition = tensor_split(arange(n_atoms), N)[d]`; keep edges with `edge_index[1] ∈ partition` (`graph_shard.h`).
3. Peer-copy / assign shard tensors to `cuda:d`; local forward (once non-opaque module exists); Kokkos reduce energy / gather forces.
4. Vesin itself may stay single-device build-once; Kokkos moves graph shards — do not invent a second NL.

## Agent loop (ops)

- Roles: WRITE / REVIEW / TEST-PREP / COORD (see plan).
- Queue: `#SBATCH --partition=gpuA100x4 --account=bbpl-delta-gpu`.
- COORD: **5 min ticks** while jobs live → `agent_stamps/coord_tick.log`.
- Cadence: devices=2 green (E+F) → then devices=4.

## File index

| Path | Role |
|------|------|
| `docs/native_kokkos_libtorch_gp.md` | This note |
| `include/uma/graph_shard.h` | FairChem-key edge/atom shard helpers |
| `docs/multi_gpu_graph_parallel.md` | NL contract + legacy vs target table |
| `examples/multi_gpu_nacl6/agent_stamps/` | WRITE/REVIEW/TEST/COORD stamps |
