# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~11:17 CDT  
**Track:** `cpp_libtorch`  
**Status:** P3a DONE · P3b DONE · **P3c queued `20935770`** · campaign OPEN (@4 vs ASE)  
**REPORT_OWNER=parent** (no RESULTS/SUMMARY/MULTIGPU/canvas edits)

## HARD campaign success (required)

Campaign **fails** if `uma/kk` honest pair ms is slower than ASE FP64 **or** FC LAMMPS at the same GPU count (NaCl6 1728), even with E+F + self-scale green.

| Required | Bar |
|----------|-----|
| Accuracy | E+F green vs d1 + ASE oracle bands |
| Perf | honest ms **≤ ASE and ≤ FC** at **devices=2 and 4** |
| Soft | under/near ASE @4 (~115) |

## Perf campaign

| Phase | Job | ms @1/2/4 | Notes |
|-------|-----|-----------|-------|
| P1 CUDA IPC | `20932975` | 320.3 / 265.0 / 193.3 | self-scale PASS |
| P2 pipe tax | `20933393` | 321.5 / 190.8 / 140.9 | @4 FAIL vs ASE/FC |
| P3a | `20934280` | 320.6 / 183.6 / 117.6 | ≤FC @4; **FAIL vs ASE** (+2.4) |
| P3b | — | — | wait≈compute; force_ar≪1ms; residual in fwd/bwd |
| **P3c** | **`20935770`** | TBD | `UMA_PEER_TRANSPORT=nccl` A/B vs P3a (PENDING) |

P3b: `P3B_ATTRIBUTION.md`. Prior false “P3c in flight” cleared — real sbatch at ~11:17 → `20935770`.
