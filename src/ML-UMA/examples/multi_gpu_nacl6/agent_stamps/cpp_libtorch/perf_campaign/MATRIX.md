# Settings matrix + BEST_BARS (living)

**Plan:** ASE/FC best bars = **minimum floor**; uma max-push continues to hard ceiling.
**Stamp:** 2026-08-10T08:45 CDT · W8nk product (`pair_style uma`)

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
| NaCl6 | 1 | ufast **350.3** | gen **345.5** | ufast **296.5** (W15) | 53.8 | 49.0 | **PASS** |
| NaCl6 | 2 | ufast **191.6** | gen **193.2** | ufast **159.4** (W7) | 32.2 | 33.8 | **PASS** |
| NaCl6 | 4 | ufast **164.5** | gen **118.0** | ufast **91.6** (W8-fix) | 72.9 | 26.4 | **PASS** |
| water888 | 1 | ufast **337.6** | gen **359.4** | ufast **359.7** (W15) | -22.1 | -0.3 | **FAIL** |
| water888 | 2 | ufast **165.5** | gen **200.5** | ufast **164.75** (W8-fix) | 0.75 | 35.75 | **PASS** |
| water888 | 4 | ufast **94.5** | gen **118.9** | ufast **96.2** (W8-fix) | **-1.7** | 22.7 | **FAIL** |

Living E/F/timing tables: [`settings_docs/`](settings_docs/README.md) — refresh with `python regenerate_settings_docs.py --ingest-matrix`.


### W8nk (product — no Kokkos)

`pair_style uma` + W8-fix NCCL. NVT Pair ms: NaCl@2 **161.94**, NaCl@4 **92.10**, water@2 **164.82**, water@4 **95.74** (still FAIL ASE ufast 94.5).

### W8-fix (promoted where E/F PASS)

Stream-ordered NCCL (default→nccl precede). NaCl@4 **91.59** ms clears ASE; water@4 **96.21** still fails ASE ufast 94.5. Prior broken W8 kept as INVALID_FORCE probes only.


## Grid (seed / live)

See `MATRIX.json`. Illegal and FC-merge crashes stamped there.

### W15 (devices=1 traced-fast)

Re-exported `model_traced.pt` with `umas_fast_pytorch`+`merge_mole` from NaCl atoms (was identical to general f64). NaCl@1 NVT **296.5** ms, E/F vs ASE merge PASS (was 533 FAIL, E~general).

## Next

1. Keep [`settings_docs/`](settings_docs/README.md) regenerated after every gather.
2. Re-gate W8 after stream fix; beat water ASE ufast @4 (94.5).
3. Fill remaining water uma gmerge / devices=1 product gaps.
