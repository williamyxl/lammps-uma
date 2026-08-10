# Settings matrix + BEST_BARS (living)

**Plan:** ASE/FC best bars = **minimum floor**; uma max-push continues to hard ceiling.
**Stamp:** 2026-08-09T20:26 CDT

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
| NaCl6 | 1 | ufast **350.3** | gen **345.5** (locked) | ufast **533.1** | -183 | -188 | **FAIL** |
| NaCl6 | 2 | ufast **191.6** | gen **193.2** | ufast **159.4** (W7) | 32.2 | 33.8 | **PASS** |
| NaCl6 | 4 | ufast **164.5** | gen **118.0** | ufast **92.4** (W7) | 72.1 | 25.6 | **PASS** |
| water888 | 1 | ufast **337.6** | gen **359.4** (locked) | — | — | — | TBD (uma@1) |
| water888 | 2 | ufast **165.5** | gen **200.5** | ufast **165.1** (W7) | 0.4 | 35.4 | **PASS** (thin) |
| water888 | 4 | gen **118.0** (locked; ASE ufast pending) | gen **118.9** | ufast **96.8** (W7) | 21.2* | 22.1 | **PASS*** |

\*Water ASE `gmerge`/`ufast` not locked yet — floor vs locked `gen` until those land; re-lock BEST_ASE if ufast is faster.

## Grid (seed / live)

See `MATRIX.json`. Illegal and FC-merge crashes stamped there.

## Next

1. Fix water EX (absolute) → measure water ASE gmerge/ufast @1/@2/@4; NaCl ASE @1 gmerge/ufast.
2. Fill missing uma `gen`/`gmerge`/`ufast` @1 and water/NaCl gaps.
3. Recompute BEST_BARS; then Tier2 **W8** max-push (floor already green on NaCl @2/@4 and water vs gen).
