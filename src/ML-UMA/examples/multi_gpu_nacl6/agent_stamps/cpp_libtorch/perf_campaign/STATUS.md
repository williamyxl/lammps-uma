# uma/kk perf campaign — living status

**Stamp:** 2026-08-09T16:07:47 CDT · Loop **armed** · **Tier1 COMPLETE — turbo PASS vs merge ASE (NaCl+water)**
**State:** `STATE.json` · Plan: `v5_max_perf_push_82db7365.plan.md` (**CURRENT**)

## Dual-oracle policy

| Oracle | When |
|--------|------|
| ASE FP64 **general** (`merge_mole=False`) | Tier0 / product `*-f64` |
| ASE FP64 **merge_mole** (job `20983514`) | Tier1 turbo `*-f64-fast` / `*-f64-merge` |

`|ΔE|~2.1e-5` vs **general** ASE is the expected MOLE-fuse residual, **not** a uma bug. vs **merge** ASE: `|ΔE|~1.7e-10`.

## Tier0 — general art (product)

| Suite | @2 | @4 | E/F vs general ASE |
|-------|---:|---:|:------------------:|
| NaCl6 | 172.9 | 100.2 | PASS |
| water888 | 178.3 | 104.2 | PASS |

## Tier1 turbo — speed + E/F vs merge ASE

| Suite | @2 ms | @4 ms | vs merge ASE |
|-------|------:|------:|:------------:|
| NaCl6 (fast+merge) | **161** | **92** | **PASS** (retarget; NaCl@2 probe) |
| NaCl6 (merge-only) | **170** | — | **PASS** dE~1.7e-10, fmax=5.00e-07 |
| water888 (fast+merge) | **173** | **96** | **PASS** vs merge ASE (job 20984160; |ΔE|~1e-11) |

ASE@1 merge oracle ms: general+merge **367**, umas_fast+merge **350** (single GPU).

## Next

1. **DONE:** water merge-ASE force gate PASS @2/@4.
2. Keep pushing speed (W5 / Tier2) with merge-ASE as E/F bar for turbo path.
3. devices=1 fast re-export still optional.

## Constraints

FP64 · `uma/kk` + Kokkos · 1 MPI · no Ray · full parent NL · no force-reduce skip.
