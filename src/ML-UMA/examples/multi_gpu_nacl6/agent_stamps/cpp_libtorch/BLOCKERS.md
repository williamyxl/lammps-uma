# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~12:18 CDT  
**Track:** `cpp_libtorch`  
**Status:** P3a DONE · P3b DONE · P3c `20935770` false-FAIL · **resubmit `20940372`** · campaign OPEN  
**REPORT_OWNER=parent** (no RESULTS/SUMMARY/MULTIGPU/canvas edits)

## HARD campaign success (required)

| Required | Bar |
|----------|-----|
| Accuracy | E+F green vs d1 + ASE oracle bands |
| Perf | honest ms **≤ ASE and ≤ FC** at **devices=2 and 4** |
| Soft | under/near ASE @4 (~115) |

## Perf campaign

| Phase | Job | ms @1/2/4 | Notes |
|-------|-----|-----------|-------|
| P3a | `20934280` | 320.6 / 183.6 / 117.6 | ≤FC @4; FAIL vs ASE (+2.4) |
| P3b | — | — | residual in fwd/bwd (+nl) |
| P3c | `20935770` | — | FAILED false sanity (NCCL linked) |
| **P3c** | **`20940372`** | TBD | resubmit after sanity fix |

Campaign PASS only if @4 ≤ ASE 115.2 and ≤ FC 118 with E+F green.
