# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~12:20 CDT  
**Track:** `cpp_libtorch`  
**Status:** P3c `20935770` false-FAIL · `20940372` cancelled · **active `20940376`** · campaign OPEN  
**REPORT_OWNER=parent** (no RESULTS/SUMMARY/MULTIGPU/canvas edits)

## HARD bar
@4 honest ms **≤ ASE 115.2 and ≤ FC 118** + E+F green. Soft ≤150 is not success.

## Perf

| Phase | Job | Notes |
|-------|-----|-------|
| P3a | `20934280` | 320.6 / 183.6 / 117.6 — FAIL vs ASE +2.4 |
| P3c | `20935770` | FAILED sanity (pipefail false neg; NCCL was linked) |
| P3c | `20940372` | cancelled (replaced) |
| **P3c** | **`20940376`** | explicit worker `UMA_ENGINE_USE_NCCL` + libnccl; nm/ldd gate |

Fix: CMake links libnccl + defs on `uma_libtorch_mp_worker` directly; sanity uses `nm -D` + `ldd`.
