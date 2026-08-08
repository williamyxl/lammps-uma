# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~09:51 CDT  
**Track:** `cpp_libtorch`  
**Status:** Perf P2 DONE — E+F green · self-scale green · beats ASE/FC @2 GPU

## Perf campaign

| Phase | Job | ms @1/2/4 | Notes |
|-------|-----|-----------|-------|
| P1 CUDA IPC | `20932975` | 320.3 / 265.0 / 193.3 | self-scale PASS |
| **P2 pipe tax** | **`20933393`** | **321.5 / 190.8 / 140.9** | beat P1; **@2 beats ASE/FC (~194)**; @4 ≲150 soft target PASS, still ~26 ms behind ASE (~115) |

Hard gates: E+F green (max|ΔF|=0 vs d1) · ms(2)<ms(1)<ms(4) order · beat P1 @2/@4 — **all PASS**.

### Optional next
Further close devices=4 toward ASE/FC ~115–118 ms (remaining ~23–26 ms).
