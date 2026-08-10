# Settings matrix + BEST_BARS (living)

**Plan:** ASE/FC best bars = **minimum floor**; uma max-push continues to hard ceiling.
**Stamp:** 2026-08-09T21:03 CDT

## Policy

1. Accuracy: FP64; uma E/F vs matching ASE oracle.
2. Measure ASE/FC under legal author knobs; lock **BEST_BARS** per `(system, ngpu)`.
3. Floor: `uma ≤ best_ASE` and `uma ≤ best_FC` (working cells only).
4. Max-push: after floor green, keep Tier2 until hard ceiling (two consecutive &lt;1 ms wins). Few-ms under ASE/FC is **not** done.

### Tags

| Tag | `execution_mode` | `merge_mole` | ASE | FC | uma art |
|-----|------------------|--------------|-----|-----|---------|
| `gen` | `general` | False | run | run | `*-f64` |
| `gmerge` | `general` | True | run | SKIP_KNOWN_CRASH | `*-f64-merge` |
| `ufast` | `umas_fast_pytorch` | True | run | SKIP_KNOWN_CRASH | `*-f64-fast` |
| `ufast_nomole` | `umas_fast_pytorch` | False | SKIP illegal | SKIP | — |

FC+`merge_mole`+FP64: FairChem Float/Double crash — do not fix.

### Metrics

- NaCl: `ms_per_eval_python`
- Water: `nvt_pair_ms_per_step`

## BEST_BARS (floor — update as cells land)

| System | ngpu | best ASE (tag, ms) | best FC (tag, ms) | uma (tag, ms) | margin ASE | margin FC | floor |
|--------|-----:|--------------------|-------------------|---------------|----------:|----------:|:-----:|
| NaCl6 | 1 | ufast **350.3** | gen **345.5** | gen **315.6** (ufast 533 FAIL) | 34.7 | 29.9 | **PASS** (gen) |
| NaCl6 | 2 | ufast **191.6** | gen **193.2** | ufast **159.4** (W7) | 32.2 | 33.8 | **PASS** |
| NaCl6 | 4 | ufast **164.5** | gen **118.0** | ufast **92.4** (W7) | 72.1 | 25.6 | **PASS** |
| water888 | 1 | ufast **337.6** | gen **359.4** | ufast **337.7** (E~gen) | −0.1 | 21.7 | **FAIL**/suspect @1 |
| water888 | 2 | ufast **165.5** | gen **200.5** | ufast **164.75** (W8-fix) | 0.75 | 35.75 | **PASS** |
| water888 | 4 | ufast **94.5** | gen **118.9** | ufast **96.8** (W7) | **-2.3** | 22.1 | **FAIL** |

Living E/F/timing tables: [`settings_docs/`](settings_docs/README.md) — refresh with `python regenerate_settings_docs.py --ingest-matrix`.

### W8 probe (not promoted)

NaCl ufast W8 (`20989797`/`20989798`): timing ~160.2 / 90.5 ms but **INVALID_FORCE** (absmax ~1e11 / 5e6) — NCCL on dedicated stream raced Torch default producers. Stream-order fix in `shared_peer.h`; re-gate before promoting over W7.

## Grid (seed / live)

See `MATRIX.json`. Illegal and FC-merge crashes stamped there.

## Next

1. Keep [`settings_docs/`](settings_docs/README.md) regenerated after every gather.
2. Re-gate W8 after stream fix; beat water ASE ufast @4 (94.5).
3. Fill remaining water uma gmerge / devices=1 product gaps.
