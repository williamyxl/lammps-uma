# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~10:25 CDT  
**Track:** `cpp_libtorch`  
**Status:** P2 @4 **FAIL vs ASE/FC** · **P3a queued** (`20934280`) · campaign OPEN  
**REPORT_OWNER=parent** (no RESULTS/SUMMARY/MULTIGPU/canvas edits)

## HARD campaign success (required)

Campaign **fails** if `uma/kk` honest pair ms is slower than ASE FP64 **or** FC LAMMPS at the same GPU count (NaCl6 1728), even with E+F + self-scale green. Soft ≤150 does **not** count as success.

| Required | Bar |
|----------|-----|
| Accuracy | E+F green vs d1 + ASE oracle bands |
| Perf | honest ms **≤ ASE and ≤ FC** at **devices=2 and 4** |
| Soft | under/near ASE @4 (~115) |

P2: PASS @1/@2 · **FAIL @4** (140.9 vs ASE~115 / FC~118).

## Perf campaign

| Phase | Job | ms @1/2/4 | Notes |
|-------|-----|-----------|-------|
| P1 CUDA IPC | `20932975` | 320.3 / 265.0 / 193.3 | self-scale PASS |
| P2 pipe tax | `20933393` | 321.5 / 190.8 / 140.9 | @2 ≤ASE/FC; **@4 FAIL** |
| **P3a** | `20934280` | TBD | fast CPU shard + sync trim; cuda_ipc |
| P3b | — | — | PERF_PARENT vs PERF_TICK attribution |
| P3c | — | — | opt-in `UMA_PEER_TRANSPORT=nccl` (after P3b) |

Also keep: E+F green · self-scale · beat prior @4. Success only when @4 ≤ASE and ≤FC.

### Queue
P3a `20934280` PENDING (Priority); est. StartTime ~2026-08-08T22:02 CDT. No P3c coding until P3a lands + P3b stamped.
