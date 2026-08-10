# uma/kk perf campaign — living status

**Stamp:** 2026-08-09T23:11 CDT · Loop **armed** · **W8-fix COMPLETE** (NaCl@4 91.6; water@4 96.2 still > ASE 94.5)
**Matrix:** [`MATRIX.md`](MATRIX.md) · **Settings docs:** [`settings_docs/`](settings_docs/README.md) · **State:** `STATE.json` · Plan: `v5_max_perf_push_82db7365.plan.md` (**CURRENT** — Tier K after W8-fix @4)

## Locked speed baselines — `general` only (do **not** re-run)

Historical floor (`execution_mode=general`, `merge_mole=False`). Still required;
**not** sufficient for Tier1+ once matching-settings bars land.

| Suite | path | @1 | @2 | @4 | Source |
|-------|------|---:|---:|---:|--------|
| NaCl6 | ASE FP64 | 396.5 | 193.9 | 115.2 | `STATE.json` v0 / P3c campaign |
| NaCl6 | FC LAMMPS | 345.5 | 193.2 | 118.0 | same |
| water888 | ASE FP64 NVT | 382.09 | 198.19 | 117.98 | `water888/results/COMPARE.md` (jobs 20948821 / 20949177 / 20949180) |
| water888 | FC LAMMPS NVT | 359.40 | 200.54 | 118.94 | same (20949064 / 20949178 / 20949181) |

## Matching-settings speed bars (measure → lock)

Honest metric: NaCl `ms_per_eval_python` (ignore SLURM-wall `ms_per_eval`). Water NVT Pair pending resubmit (`--chdir` fix).

| Suite | path | settings | @2 | @4 | status |
|-------|------|----------|---:|---:|--------|
| NaCl6 | ASE | `general`+`merge_mole` | **195.7** (`20989338`) | **167.9** (`20989346`) | locked |
| NaCl6 | ASE | `umas_fast_pytorch`+`merge_mole` | **191.6** (`20989339`) | **164.5** (`20989347`) | locked |
| NaCl6 | FC | `merge_mole=True` (any mode) | — | — | **FAIL** FP64 `Float`/`Double` in `merge_MOLE` (`fc_merge_mole_fp64_FAIL.json`) |
| water888 | ASE | both settings @2/@4 | `20989439`+ | `20989443`+ | queued (resubmit) |
| water888 | FC | `merge_mole=True` | — | — | expect same FP64 FAIL as NaCl FC |

**Tier1+ hard gate:** uma ≤ **matching ASE**. Matching FC is **blocked** until FairChem supports `merge_mole` under FP64 (ASE path works). Uma W7 already ≤ NaCl matching ASE (`*-f64-fast` @2 159.4 &lt; 191.6; @4 92.4 &lt; 164.5). Locked `general` FC remains a secondary floor only.

## Settings (required)

**Canonical:** [`GLOSSARY.md`](GLOSSARY.md). Write FairChem knobs + artifact dirs; do not invent synonyms.

| Artifact | `execution_mode` | `merge_mole` |
|----------|------------------|--------------|
| `*-f64` | `general` | `False` |
| `*-f64-merge` | `general` | `True` |
| `*-f64-fast` | `umas_fast_pytorch` | `True` |

FairChem `InferenceSettings` preset **`turbo`** is unused here (≠ `umas_fast_pytorch`).

## Dual-oracle policy (E/F only)

| Oracle | When |
|--------|------|
| ASE `general` (`merge_mole=False`) | Tier0 / product `*-f64` — reuse cached ASE@1 E+F |
| ASE `merge_mole=True` | `*-f64-fast` / `*-f64-merge` — stamps NaCl `20983514`, water `20984160` |

`|ΔE|~2.1e-5` vs ASE `general` is the expected **MOLE-fuse residual**, **not** a uma bug. vs ASE `merge_mole=True`: `|ΔE|~1e-10`–`1e-11`.

## Tier0 — `*-f64` (`general`, `merge_mole=False`)

| Suite | @2 | @4 | E/F vs ASE `general` |
|-------|---:|---:|:--------------------:|
| NaCl6 | 172.9 | 100.2 | PASS |
| water888 | 178.3 | 104.2 | PASS |

## Tier1 — `*-f64-fast` / `*-f64-merge` vs ASE `merge_mole=True`

| Suite | settings | @2 ms | @4 ms | vs ASE `merge_mole=True` |
|-------|----------|------:|------:|:------------------------:|
| NaCl6 | `umas_fast_pytorch`+`merge_mole` | **161** | **92** | **PASS** (retarget; NaCl@2 probe) |
| NaCl6 | `general`+`merge_mole` | **170** | — | **PASS** dE~1.7e-10, fmax=5.00e-07 |
| water888 | `umas_fast_pytorch`+`merge_mole` | **173** | **96** | **PASS** (oracle `20984160`; \|ΔE\|~1e-11) |

ASE@1 (`merge_mole=True`) ms: `general`+merge **367**, `umas_fast_pytorch`+merge **350** (single GPU).

## Next

1. Finish NaCl FC matching bars; **resubmit water ASE/FC** with `--chdir` to `water888`.
2. Tier2 **W8** (NCCL stream overlap) — W7 flat (&lt;1 ms / water@4 +1 ms).
3. devices=1 `umas_fast_pytorch`+`merge_mole` export required for product completeness.

## Constraints

FP64 · 1 MPI · no Ray · full parent NL · no force-reduce skip.  
**Until Tier K:** gate on `uma/kk` + `-k on g N` for continuity. **After Tier K:** product recipe is `pair_style uma … devices N` (see plan §6).

## Wave A gates
| Suite | @2 | @4 | E/F vs ASE `merge_mole=True` | notes |
|-------|---:|---:|:----------------------------:|-------|
| NaCl6 | **160.9** (`20984870`) | **92.7** (`20985402`) | PASS | W5/W9 |
| water888 | **164.28** (`20985385`) | **96.25** (`20985403`) | PASS | attach max_e fix |

Wave A **COMPLETE PASS**.

## Queue (live)
- W8-fix **COMPLETE**: NaCl@2 160.08 / @4 **91.59** (E/F PASS; @4 −0.8 vs W7). Water@2 164.75 / @4 **96.21** (E/F PASS; −0.6 vs W7; still **FAIL** ASE ufast 94.5 by 1.7 ms).
- Next: continue Tier2 toward water@4 ASE floor, then **Tier K** (drop Kokkos).


## Tier2 W6 (COMPLETE PASS)
| Suite | @2 | @4 | Δ vs Wave A @2/@4 |
|-------|---:|---:|-------------------:|
| NaCl6 | **160.2** (`20989045`) | **92.9** (`20989046`) | -0.7 / +0.2 |
| water888 | **165.47** (`20989047`) | **95.83** (`20989048`) | +1.19 / -0.42 |

## Tier2 W7 (COMPLETE PASS)
| Suite | @2 | @4 | Δ vs W6 |
|-------|---:|---:|--------:|
| NaCl6 | **159.4** (`20989184`) | **92.37** (`20989185`) | -0.9 / -0.56 |
| water888 | **165.10** (`20989186`) | **96.84** (`20989187`) | -0.37 / **+1.01** |

Two-phase publish: E/F vs ASE `merge_mole=True` PASS; speed ≤ locked `general` ASE/FC. Wins mostly &lt;1 ms (water@4 regression) → continue W8.
