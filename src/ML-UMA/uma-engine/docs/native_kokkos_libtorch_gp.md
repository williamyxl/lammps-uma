# Native Kokkos + LibTorch multi-GPU (no Ray)

**Stamp:** 2026-08-08 ~09:00 CDT  
**Plan:** `.cursor/plans/libtorch_multi-gpu_kokkos_20de0136.plan.md`  
**Perf plan:** `.cursor/plans/mp_performance_campaign_e2397088.plan.md`  
**Agent stamps:** `examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/`  
**Results:** `examples/multi_gpu_nacl6/results/RESULTS.md`

## Goal (product)

**Deliverable:** a Kokkos-driven multi-GPU inference engine for **LAMMPS** integration — not a standalone Python GP path and not FairChem Ray behind `uma`.

Ship as same-node `pair_style uma/kk precision double devices N` with:

- `--ntasks=1`, `lmp -k on g N -sf kk` on Delta **`gpuA100x4`**
- Full scientific UMA (no smaller cutoff / fewer neighbors / skipped layers)
- **Vesin** full NL → FairChem partition **key** → LibTorch shards
- Inter-GPU traffic via **CUDA IPC** device payloads + process-shared control (default `UMA_PEER_TRANSPORT=cuda_ipc`); host `/dev/shm` payload remains as `shm` fallback — not MPI, not Ray, not c10d-torchrun
- Gates: energy + per-atom forces vs ASE FP64@1 / devices=1; honest pair ms from `uma64 E=…`
- **Landed:** devices=2 and devices=4 E+F green · **self-scale green** (320 / 265 / 193 ms @1/2/4, job `20932975`)

## Status (2026-08-08)

| Item | State |
|------|--------|
| Phase 2b engine CLI E+F @2/@4 | GREEN |
| Phase 3 LAMMPS `uma/kk` E+F @2/@4 | GREEN (max\|ΔF\|=0) |
| Perf P0 (drop `CUDA_LAUNCH_BLOCKING`) | DONE — still no self-scale |
| Perf P1 (CUDA IPC collectives) | **DONE — self-scale PASS** |
| Phase 5 multi-node | out of scope |

## Current vs target load path

| `devices` | **Legacy (opt-in)** | **Default (product)** |
|-----------|--------------------|------------------------|
| `1` | TorchScript `model_traced.pt` + vesin CUDA NL | unchanged |
| `N>1` | `UMA_PYTHON_GP_WORKER=1` → Python; `UMA_ALLOW_RAY_GP=1` → Ray | **C++** `LibtorchMpRuntime`: `model_mp_wN_n*_r*.pt` + `uma_peer` + CUDA IPC + vesin; `uma/kk` + `-k on g N` |

Env: `UMA_FORBID_RAY_GP=1` rejects Ray; `UMA_PYTHON_GP_WORKER=1` opt-in Python only; `UMA_PEER_TRANSPORT=shm|cuda_ipc`.

## Why an earlier plan “succeeded” but failed

[`uma_kk_multi-gpu_23f22729`](../../../../.cursor/plans/uma_kk_multi-gpu_23f22729.plan.md) marked todos done after shipping **FairChem Ray** behind `GraphParallelRuntime`. That matched ASE/FC timing because it *was* FairChem GP. Native LibTorch+Kokkos is the product path above.

## Phase 0c — Vesin strategy (landed)

1. Build **one** full graph with `vesin_nl::vesin_build_graph_cuda` + FairChem edge flip + `max_neighbors=300`.
2. Partition: FairChem key via `graph_shard.h`.
3. Process-per-rank workers (JIT mid-forward deadlock forbids threaded multi-module); collectives via CUDA IPC.
4. Do not invent a second NL.

## File index

| Path | Role |
|------|------|
| `docs/native_kokkos_libtorch_gp.md` | This note |
| `include/uma/shared_peer.h` | shm control + CUDA IPC / host payload transports |
| `include/uma/graph_shard.h` | FairChem-key edge/atom shard helpers |
| `docs/multi_gpu_graph_parallel.md` | NL contract + legacy vs target table |
| `examples/multi_gpu_nacl6/results/RESULTS.md` | Canonical gates + pair ms |
| `examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/` | Campaign stamps |
