# COORD ANALYSIS — NaCl6 multi-GPU

> Prefer [`RESULTS.md`](RESULTS.md). Mixed disabled.

**Stamp:** 2026-08-07 ~17:55 CDT · Geometry: `structures/nacl6_rattle_fixed.extxyz`

## Timing (today path-isolated ASE/FC + gp_round uma)

| path | 1× ms | 2× ms | 4× ms | 1→2 | 1→4 | jobs |
|------|------:|------:|------:|----:|----:|------|
| ASE FP64 | 396.5 | 193.9 | 115.2 | 2.04× | 3.44× | `20910344`/`48`/`52` |
| FC | 345.5 | 193.2 | 118.0 | 1.79× | 2.93× | `20910345`/`49`/`53` |
| uma double GP | 322.2 | 192.4 | 112.6 | 1.67× | 2.86× | gp_round |

## Findings

1. ASE `workers=N` still ~**2× @2** and ~**3.4× @4** on re-measure.
2. FC ~**1.8× @2** and ~**2.9× @4** (η@4 ≈ 73%).
3. uma GP double ~**2.9× @4**.
