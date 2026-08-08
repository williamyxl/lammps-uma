# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~09:00 CDT  
**Track:** `cpp_libtorch`  
**Status:** Phase 3 DONE · Phase 4 report · **Perf P1 CUDA IPC DONE (self-scale GREEN)**

## B1 / B2 / B4–B7 — resolved or mitigated (product = C++ LibTorch MP, not Ray)

See prior bursts. MP TS remains n_atoms-specific (B7 mitigation: `UMA_MP_NATOMS` + `model_mp_w*_n*_r*.pt`).

## Perf campaign — HARD GATE PASS

Hard gate: `pair_ms(2) < pair_ms(1)` and `pair_ms(4) < pair_ms(2)` on NaCl6 1728 FP64, E+F green vs d1.

### P0 — DONE (job `20932843`)

CUDA_LAUNCH_BLOCKING off. Honest pair ms: **320.64 / 330.52 / 382.86** @1/2/4. E+F green. Self-scale **FAIL**.

### P1 — CUDA IPC — DONE (job `20932975`)

| devices | pair ms | dE_d1 | max\|ΔF\| |
|--------:|--------:|------:|----------:|
| 1 | **320.34** | 1.8e-12 | **0** |
| 2 | **264.96** | 0 | **0** |
| 4 | **193.32** | 1.8e-12 | **0** |

Self-scale **GREEN** (0.83× / 0.60× vs d1). Transport: `UMA_PEER_TRANSPORT=cuda_ipc` (default). Commit `8e7e6a0d27`.

### Optional next (not blocking)

- **P2:** shrink parent↔worker pipe/mmap tax (further close on ASE/FC ~194 / ~115 @2/@4).
- Unbake `gp_node_offset` for size-agnostic MP artifacts.

## Phase 5 — multi-node

Out of scope.
