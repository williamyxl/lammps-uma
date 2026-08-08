# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~10:05 CDT  
**Track:** `cpp_libtorch`  
**Status:** P2 DONE · **P3 in flight** (close @4 vs ASE ~115)  
**REPORT_OWNER=parent** (no RESULTS/SUMMARY/MULTIGPU/canvas edits)

## Perf campaign

| Phase | Job | ms @1/2/4 | Notes |
|-------|-----|-----------|-------|
| P1 CUDA IPC | `20932975` | 320.3 / 265.0 / 193.3 | self-scale PASS |
| P2 pipe tax | `20933393` | 321.5 / 190.8 / 140.9 | @2 beats ASE/FC; @4 +26 vs ASE |
| **P3** | (active) | TBD | fast CPU shard + fewer CUDA syncs |

Hard gates: E+F green · self-scale · beat P2 @4. Soft: @4 ≲120 → ~115.
