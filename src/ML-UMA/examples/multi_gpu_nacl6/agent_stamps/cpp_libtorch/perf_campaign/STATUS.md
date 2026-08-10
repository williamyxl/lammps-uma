# uma/kk perf campaign — living status

**Stamp:** 2026-08-09T19:45:51 CDT · Loop **armed** · **W6 COMPLETE PASS**
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

1. **Wave A COMPLETE**. **W6 COMPLETE PASS** (flat vs Wave A). Next: **W7**.
2. Then Tier2 W6→W12 one+gate each until hard ceiling.
3. devices=1 umas_fast+merge export required for product completeness.

## Constraints

FP64 · `uma/kk` + Kokkos · 1 MPI · no Ray · full parent NL · no force-reduce skip.

## Wave A unblock (tick71)
- Water@2 `20985385` still **PENDING (Resources)** — holds water@4 `20985403` (`afterok`).
- Cancelled dependent `20985386`/`20985387`; **NaCl@4** resubmitted independently as `20985402` (RECOMPILE=0; binary already good from `20984870`).
- Active queue: `20985385` Resources | `20985402` Priority | `20985403` Dependency.

## Wave A gates (live)
| Suite | @2 | @4 | E/F vs merge ASE | notes |
|-------|---:|---:|:----------------:|-------|
| NaCl6 | **160.9** (`20984870`) | **92.7** (`20985402`) | PASS | W5/W9 |
| water888 | **164.28** (`20985385`) | **96.25** (`20985403`) | PASS | attach max_e fix |

Wave A **COMPLETE PASS** — both systems @2/@4 speed ≤ ASE/FC + E/F vs merge ASE.


## Queue (live)
- W6 gates done. No pending jobs.


## Tier2 W6 (COMPLETE PASS)
| Suite | @2 | @4 | Δ vs Wave A @2/@4 |
|-------|---:|---:|-------------------:|
| NaCl6 | **160.2** (`20989045`) | **92.9** (`20989046`) | -0.7 / +0.2 |
| water888 | **165.47** (`20989047`) | **95.83** (`20989048`) | +1.19 / -0.42 |

Full-graph publish + GPU shard: E/F green; speed ≤ ASE/FC. Wins mostly **&lt;1 ms** (flat) → continue Tier2 W7.

