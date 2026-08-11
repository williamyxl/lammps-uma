# C++ LibTorch multi-GPU track — agent brief

**Track id:** `cpp_libtorch`  
**Stamp dir:** `examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/`  
**Sibling track:** Python process-GP (`kokkos_gp_runtime.py` / `uma_native_gp_worker.py`) — **do not extend**; owned by the other agent.

## Product goal

Kokkos-driven **LibTorch** multi-GPU UMA inference behind LAMMPS:

```text
lmp -k on g N -sf kk
pair_style uma/kk precision double devices N
```

- Single MPI rank (`--ntasks=1`)
- Inter-GPU: Kokkos peer (`deep_copy` / fence / `cudaMemcpyPeer`) — **not** MPI, Ray, c10d, or Python Manager
- FP64 only
- Vesin full NL → FairChem partition key → LibTorch shards
- Gate: E + per-atom F vs ASE FP64@1 / `devices=1`; land **devices=2** then **devices=4** on Delta `gpuA100x4` (`bbpl-delta-gpu`)

Plan: `/u/xyan11/.cursor/plans/libtorch_multi-gpu_kokkos_20de0136.plan.md`  
Design: `uma-engine/docs/native_kokkos_libtorch_gp.md`

## Starting assets (already landed)

| Asset | Path |
|-------|------|
| Eager weights | `uma-engine/artifacts/uma-s-1p2-omat-f64/model_state.pt` (~2.2G) |
| Arch sidecar | `…/eager_arch.json` |
| Peer math (C++) | `include/uma/kokkos_peer.h` + `tests/kokkos_peer_smoke.cpp` |
| Shard helpers | `include/uma/graph_shard.h` + `tests/graph_shard_smoke.cpp` |
| Vesin NL | `include/uma/vesin_nl.h`, `Predictor::rebuild_neighbors()` |
| GP shell | `include/uma/graph_parallel.h`, `src/graph_parallel.cpp` (today still spawns Python — **replace**) |
| Traced devices=1 | `Predictor` + `model_traced.pt` (reference only; **not** the MP forward) |

## Forbidden

- FairChem Ray / `uma_gp_worker.py` / `workers=N`
- Extending `kokkos_gp_runtime.py` / `uma_native_gp_worker.py` (sibling track)
- Dropping `-k on g N` or switching to plain `pair_style uma`
- Smaller cutoff / max_neighbors / skipped layers / FP32
- MPI ranks or c10d-torchrun as GP transport

## WRITE sequence (this track)

1. **C++ LibTorch MP skeleton**  
   Load `model_state.pt` (or reconstruct module from `eager_arch.json` + state_dict) on `cuda:0..N-1`.  
   Implement gather/reduce using device buffers + fences (evolve `kokkos_peer.h` beyond host-staged `.to()`).

2. **Vesin → shard injection**  
   One full vesin graph (cutoff 6.0, max_neighbors 300) → `graph_shard.h` → per-device edge/atom shards.

3. **Wire `GraphParallelRuntime`**  
   Default `devices>1` path = C++ LibTorch MP (no Python subprocess). Keep `UMA_ALLOW_RAY_GP` / native-python only as explicit legacy if needed.

4. **SLURM gate**  
   `RECOMPILE=1`, `ONLY_PATHS=uma_double`, devices=2 on `gpuA100x4`. E then F vs ASE/d1. Then devices=4.

## Stamps (append-only)

| File | Meaning |
|------|---------|
| `coord_tick.log` | 5m ops ticks for this track’s jobs |
| `active_jobid.txt` | Current SLURM job for this track |
| `write_progress.md` | What landed each WRITE burst |
| `review_notes.md` | Architecture self-check vs plan REVIEW checklist |
| `BLOCKERS.md` | Hard blockers (eager load impossible, etc.) — escalate, don’t pivot to Ray |

## Ops

- Account/partition: `bbpl-delta-gpu` / `gpuA100x4`
- Always `RECOMPILE=1` before gRASPA/LAMMPS jobs that touch engine (workspace rule)
- Tick cadence while jobs live: **5 min** → this dir’s `coord_tick.log`
- Do not steal the Python track’s `active_jobid.txt` (parent `agent_stamps/`)

## Sibling Python track (context only)

Best force so far ~Fmax 0.036 vs d1 with process-GP + force all_reduce; energy exact.  
Not the product. Ignore unless citing as semantic reference for FairChem gather/reduce.

## Report ownership (no data race)

**Parent** owns and tick-refreshes: `results/{RESULTS,SUMMARY,MULTIGPU_REPORT}.md`, `SUMMARY.json`, README results table, canvases.
**This agent** must **not** edit those files. Write stamps only under `agent_stamps/cpp_libtorch/` (+ engine source).
