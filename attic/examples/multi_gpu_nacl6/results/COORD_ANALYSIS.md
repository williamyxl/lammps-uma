# COORD ANALYSIS — NaCl6 multi-GPU

> Prefer [`RESULTS.md`](RESULTS.md) for canonical product numbers. Mixed disabled.

**Stamp:** 2026-08-08 ~19:00 CDT · Geometry: `structures/nacl6_rattle_fixed.extxyz`

## Timing (honest ms/eval)

| path | 1× ms | 2× ms | 4× ms | 1→2 | 1→4 | jobs |
|------|------:|------:|------:|----:|----:|------|
| ASE FP64 | 396.5 | 193.9 | 115.2 | 2.04× | 3.44× | `20910344`/`48`/`52` |
| FC | 345.5 | 193.2 | 118.0 | 1.79× | 2.93× | `20910345`/`49`/`53` |
| **uma/kk (P3c NCCL)** | **321.04** | **183.30** | **112.04** | **1.75×** | **2.87×** | **`20940474`** |

Historical: path-isolated Ray-era uma_double was 320.4 / 192.0 / 112.6 (`20910346`/`50`/`54`); P1 cuda_ipc product was 320.3 / 265.0 / 193.3 (`20932975`).

## Accuracy (vs ASE FP64@1)

| path | \|ΔE\| | max \|ΔFᵢ\| | notes |
|------|-------:|------------:|-------|
| ASE @1/2/4 | ~0 | 0 | self / workers=N |
| FC @1/2/4 | 4.9×10⁻⁶ | 7.1×10⁻⁶ | FP32 cell in FC bridge |
| uma/kk @1/2/4 | ≈1.2×10⁻¹⁰ | ≈5.0×10⁻⁷ | P3c; GP vs d1 max\|ΔF\|=0 |

## Findings

1. Campaign **PASS**: uma/kk ≤ ASE and ≤ FC at @1/@2/@4 with E+F green.
2. ASE still scales most efficiently (~3.44× @4); uma/kk 2.87× but wins absolute time.
3. Remaining headroom: default NCCL, parent NL+publish (~8–10 ms @4).
