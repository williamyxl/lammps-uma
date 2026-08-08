# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~10:17 CDT  
**Track:** `cpp_libtorch`  
**Status:** P2 DONE · **P3a queued** (`20934280`) · P3b/P3c gated  
**REPORT_OWNER=parent** (no RESULTS/SUMMARY/MULTIGPU/canvas edits)

## Perf campaign

| Phase | Job | ms @1/2/4 | Notes |
|-------|-----|-----------|-------|
| P1 CUDA IPC | `20932975` | 320.3 / 265.0 / 193.3 | self-scale PASS |
| P2 pipe tax | `20933393` | 321.5 / 190.8 / 140.9 | @2 beats ASE/FC; @4 +26 vs ASE |
| **P3a** | `20934280` | TBD | fast CPU shard + sync trim; cuda_ipc |
| P3b | — | — | PERF_PARENT vs PERF_TICK attribution (after P3a) |
| P3c | — | — | opt-in `UMA_PEER_TRANSPORT=nccl` (after P3b) |

Hard gates: E+F green · self-scale · beat P2 @4. Soft: @4 ≲120 → ~115.

### Queue
P3a `20934280` PENDING (Priority). scontrol StartTime estimate ~2026-08-08T22:02 CDT. No P3c coding until P3a lands + P3b stamped.
