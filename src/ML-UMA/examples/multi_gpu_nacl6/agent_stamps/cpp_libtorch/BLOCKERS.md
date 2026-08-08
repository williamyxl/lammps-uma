# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~08:50 CDT  
**Track:** `cpp_libtorch`  
**Status:** Phase 3 DONE · Phase 4 report landed · **Perf P1 (CUDA IPC) in flight**

## B1 / B2 / B4–B7 — as before (product = C++ LibTorch MP, not Ray)

## Perf campaign

Hard gate: `pair_ms(2) < pair_ms(1)` and `pair_ms(4) < pair_ms(2)` on NaCl6 1728 FP64, E+F green vs d1.

### P0 — DONE (job `20932843`)

CUDA_LAUNCH_BLOCKING off. Honest pair ms: **320.64 / 330.52 / 382.86** @1/2/4. E+F green (max|ΔF|=0). Self-scale **FAIL**.

### P1 — CUDA IPC device payloads (in flight)

Replace host-staged `SharedPeerGatherSlot` payload with `cudaIpcMemHandle_t` device buffers; shm keeps mutex/cond/gen + handles + nbytes.  
Env: `UMA_PEER_TRANSPORT=shm|cuda_ipc` (default `cuda_ipc` when CUDA available).  
Script: `perf_p1.slurm` · stamps under `agent_stamps/cpp_libtorch/perf/`.

## Phase 5 — multi-node

Out of scope for this perf track.
