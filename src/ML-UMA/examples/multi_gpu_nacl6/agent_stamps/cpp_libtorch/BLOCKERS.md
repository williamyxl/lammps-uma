# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~12:35 CDT  
**Track:** `cpp_libtorch`  
**Status:** P3c `20940376` CANCELLED (NCCL teardown deadlock) · **resubmit `20940474`** · campaign OPEN  
**REPORT_OWNER=parent**

## HARD bar
@4 ≤ ASE 115.2 and ≤ FC 118 + E+F green.

## Perf

| Phase | Job | ms @1/2/4 | Notes |
|-------|-----|-----------|-------|
| P3a cuda_ipc | `20934280` | 320.6 / 183.6 / 117.6 | FAIL vs ASE @4 +2.4 |
| P3c | `20940376` | 321.6 / — / — | CANCELLED teardown deadlock |
| **P3c** | **`20940474`** | TBD | broadcast shutdown + nccl destroy rendezvous |

See `P3C_HANG_20940376.md`.
