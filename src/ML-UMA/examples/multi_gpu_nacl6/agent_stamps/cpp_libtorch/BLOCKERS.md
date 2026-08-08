# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~09:15 CDT  
**Track:** `cpp_libtorch`  
**Status:** Perf P1 **PASS** · **P2 pipe-tax cut in flight**

## Non-negotiable gates

1. E+F vs devices=1: \|ΔE\| ≤ 1e-8, max\|ΔF\| ≤ 1e-6 (prefer ~1e-10 / ~5e-7 vs ASE).
2. Self-scale: `pair_ms(2) < pair_ms(1)` and `pair_ms(4) < pair_ms(2)`.

## Perf campaign

| Phase | Job | ms @1/2/4 | Notes |
|-------|-----|-----------|-------|
| P0 | `20932843` | 320.64 / 330.52 / 382.86 | CUDA_LAUNCH_BLOCKING off; scale FAIL |
| P1 | `20932975` | **320.34 / 264.96 / 193.32** | CUDA IPC; scale **PASS**; E+F green |
| P2 | (active) | TBD | payload shm fan-out + rank0-only forces |

### Soft targets (P2+)

Close gap vs ASE (193.9 / 115.2 @2/@4) and FC (193.2 / 118.0): aim devices=2 ≲200→~194, devices=4 ≲150→~115–120 **without** weakening (1)–(2).

### P2 residual (from P1 profile)

@4: compute ≈103 ms (fwd+bwd) vs pair 193 ms → ~90 ms parent/pipe/NL tax.  
Cut: shared payload mmap for pos/z/edges; return forces from rank0 only; quiet worker logs; cache partition check.
