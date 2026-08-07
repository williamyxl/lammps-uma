# COORD ANALYSIS — NaCl6 multi-GPU

> Prefer [`RESULTS.md`](RESULTS.md). Mixed disabled.

**Stamp:** 2026-08-07 ~18:00 CDT · Geometry: `structures/nacl6_rattle_fixed.extxyz`

## Timing (path-isolated ASE / FC / uma_double)

| path | 1× ms | 2× ms | 4× ms | 1→2 | 1→4 | jobs |
|------|------:|------:|------:|----:|----:|------|
| ASE FP64 | 396.5 | 193.9 | 115.2 | 2.04× | 3.44× | `20910344`/`48`/`52` |
| FC | 345.5 | 193.2 | 118.0 | 1.79× | 2.93× | `20910345`/`49`/`53` |
| uma double | 320.4 | 192.0 | 112.6 | 1.67× | 2.85× | `20910346`/`50`/`54` |

## Accuracy (vs ASE FP64@1)

| path | \|ΔE\| | max \|ΔFᵢ\| | notes |
|------|-------:|------------:|-------|
| ASE @1/2/4 | ~0 | 0 | self / workers=N |
| FC @1/2/4 | 4.9×10⁻⁶ | 7.1×10⁻⁶ | FP32 cell in FC bridge |
| uma double @1 | 1.2×10⁻¹⁰ | 5.0×10⁻⁷ | |
| uma double @2/4 | ~10⁻¹² | 5.0×10⁻⁷ | GP vs d1 PASS (ΔF=0) |

## Findings

1. ASE scales best (~3.4× @4). FC and uma double both ~2.9× @4.
2. uma double matches ASE to ~10⁻¹⁰ eV; device-parallel forces bitwise-identical vs d1.
3. Campaign complete — no pending path-isolated jobs.
