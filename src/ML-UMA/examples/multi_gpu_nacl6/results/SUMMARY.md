# NaCl 6×6×6 multi-GPU parity summary

**Stamp:** 2026-08-07 ~18:00 CDT · Canonical detail: [`RESULTS.md`](RESULTS.md)

Fixed geometry `structures/nacl6_rattle_fixed.extxyz` (1728 atoms). ASE/FC: FairChem `workers=N`. uma/kk: `lmp -k on g N -sf kk`, `pair_style uma/kk precision double devices N`. Mixed disabled.

- **Oracle:** ASE FP64 @1 = `−5830.9237201666` eV
- **uma vs devices=1 gates:** **3/3 PASS**
- **Jobs:** ASE `20910344/48/52` · FC `20910345/49/53` · uma_double `20910346/50/54`

## Honest timing (ms/eval)

| Path | 1 GPU | 2 GPU | 4 GPU | 1→2 | 1→4 |
|------|------:|------:|------:|----:|----:|
| ASE FairChem FP64 | 396.5 | 193.9 | 115.2 | 2.04× | 3.44× |
| FairChem FC | 345.5 | 193.2 | 118.0 | 1.79× | 2.93× |
| uma/kk double | 320.4 | 192.0 | 112.6 | 1.67× | 2.85× |

## Energy + forces vs ASE FP64@1

| Path | ngpu | Energy (eV) | \|ΔE\| | Force MAE | Force RMSE | max \|ΔFᵢ\| | Cosine |
|------|-----:|-------------:|-------:|----------:|-----------:|------------:|-------:|
| ASE FP64 | 1/2/4 | −5830.9237201666 | ~0 | 0 | 0 | 0 | 1.0 |
| FairChem FC | 1/2/4 | −5830.9237152511 | 4.915×10⁻⁶ | 1.002×10⁻⁶ | 1.343×10⁻⁶ | 7.123×10⁻⁶ | 1.0 |
| uma/kk double | 1 | −5830.9237201667 | 1.23×10⁻¹⁰ | 1.516×10⁻⁷ | 2.179×10⁻⁷ | 5.000×10⁻⁷ | 1.0 |
| uma/kk double | 2 | −5830.9237201666 | 1.82×10⁻¹² | 1.516×10⁻⁷ | 2.179×10⁻⁷ | 5.000×10⁻⁷ | 1.0 |
| uma/kk double | 4 | −5830.9237201666 | 9.09×10⁻¹³ | 1.516×10⁻⁷ | 2.179×10⁻⁷ | 5.000×10⁻⁷ | 1.0 |

## uma/kk double vs devices=1

| ngpu | \|ΔE\| vs d1 | max \|ΔF\| vs d1 | cosine | gate |
|-----:|-------------:|-----------------:|-------:|:----:|
| 1 | — | 0 | 1.0 | PASS |
| 2 | 1.27×10⁻¹⁰ | 0 | 1.0 | PASS |
| 4 | 1.27×10⁻¹⁰ | 0 | 1.0 | PASS |

Thresholds (double): \|ΔE\| ≤ 1×10⁻⁸ · max\|ΔF\| ≤ 1×10⁻⁶.

> Auto-merged `SUMMARY.json` / SLURM wall `ms_per_eval` values are contaminated — ignore them for scaling.
