# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~19:00 CDT  
**Track:** `cpp_libtorch`  
**Status:** **P3c PASS** job `20940474` · campaign **PASS** hard bar @4  
**REPORT_OWNER=parent** (no RESULTS/SUMMARY/MULTIGPU/canvas edits)

## HARD campaign success

| Required | Result |
|----------|--------|
| E+F green | PASS (max\|ΔF\|=0 @1/2/4) |
| @2 ≤ ASE & FC | PASS (183.30 ≤ 193.9 / 193.2) |
| @4 ≤ ASE & FC | **PASS (112.04 ≤ 115.2 / 118.0)** |

## Perf

| Phase | Job | ms @1/2/4 | Notes |
|-------|-----|-----------|-------|
| P3a cuda_ipc | `20934280` | 320.6 / 183.6 / 117.6 | FAIL vs ASE @4 +2.4 |
| P3c hang | `20940376` | 321.6 / — / — | teardown deadlock (fixed) |
| **P3c nccl** | **`20940474`** | **321.0 / 183.3 / 112.0** | **HARD_ASE_FC OK** · beat P3a |

Self-scale green. Teardown clean (all ranks shutdown). Summary: `perf/summary_p3c_20940474.json`.
