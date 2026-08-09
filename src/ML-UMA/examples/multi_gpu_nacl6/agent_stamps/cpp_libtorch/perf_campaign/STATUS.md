# uma/kk perf campaign — living status

**Stamp:** 2026-08-09T16:07:47 CDT · Loop **armed** · **Tier1 COMPLETE — turbo PASS vs merge ASE (NaCl+water)**
**State:** `STATE.json` · Plan: `v5_max_perf_push_82db7365.plan.md` (**CURRENT**)

## Locked speed baselines — do **not** re-run ASE/FC

Campaign speed gates (`le_ase` / `le_fc`) **reuse** these already-documented FP64 runs. Re-run only if geometry, checkpoint, or FairChem stack changes.

| Suite | path | @1 | @2 | @4 | Source |
|-------|------|---:|---:|---:|--------|
| NaCl6 | ASE FP64 | 396.5 | 193.9 | 115.2 | `STATE.json` v0 / P3c campaign |
| NaCl6 | FC LAMMPS | 345.5 | 193.2 | 118.0 | same |
| water888 | ASE FP64 NVT | 382.09 | 198.19 | 117.98 | `water888/results/COMPARE.md` (jobs 20948821 / 20949177 / 20949180) |
| water888 | FC LAMMPS NVT | 359.40 | 200.54 | 118.94 | same (20949064 / 20949178 / 20949181) |

Uma-only jobs are what we submit for perf iterations.

## Dual-oracle policy (E/F only)

| Oracle | When |
|--------|------|
| ASE FP64 **general** (`merge_mole=False`) | Tier0 / product `*-f64` — reuse cached ASE@1 E+F |
| ASE FP64 **merge_mole** (NaCl `20983514`, water `20984160`) | Tier1 turbo — one-shot oracles, reuse stamps |

`|ΔE|~2.1e-5` vs **general** ASE is the expected MOLE-fuse residual, **not** a uma bug. vs **merge** ASE: `|ΔE|~1e-10`–`1e-11`.

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
