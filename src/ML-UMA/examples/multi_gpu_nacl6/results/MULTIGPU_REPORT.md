# UMA multi-GPU NaCl 6×6×6 parity report

**Stamp:** 2026-08-07 ~18:00 CDT · Full write-up: [`RESULTS.md`](RESULTS.md)

## Setup

- **System:** NaCl 6×6×6 rocksalt, 1728 atoms
- **Structure:** `structures/nacl6_rattle_fixed.extxyz` (δ=0.1 Å seed=0, immutable)
- **Reference:** ASE FairChem FP64 @ ngpu=1 → `−5830.9237201666` eV
- **ASE/FC:** `workers=N` in one process
- **uma/kk:** `lmp -k on g N -sf kk`, `pair_style uma/kk precision double devices N`
- **Precision:** FP64 / double only (mixed disabled)
- **Parity gates (uma vs devices=1):** **3/3 PASS**

## Honest timing (ms/eval)

| Path | 1 GPU | 2 GPU | 4 GPU | 1→2 | 1→4 |
|------|------:|------:|------:|----:|----:|
| ASE FairChem FP64 | 396.5 | 193.9 | 115.2 | 2.04× | 3.44× |
| FairChem FC | 345.5 | 193.2 | 118.0 | 1.79× | 2.93× |
| uma/kk double | 320.4 | 192.0 | 112.6 | 1.67× | 2.85× |

Jobs: ASE `20910344/48/52` · FC `20910345/49/53` · uma_double `20910346/50/54`.

## uma vs devices=1 (graph-parallel gate)

| Path | ngpu | devices | \|ΔE\| vs d1 | max \|ΔF\| vs d1 | cosine vs d1 | gate |
|------|-----:|--------:|-------------:|-----------------:|-------------:|:----:|
| uma/kk double | 1 | 1 | — | 0 | 1.000000 | PASS |
| uma/kk double | 2 | 2 | 1.273×10⁻¹⁰ | 0 | 1.000000 | PASS |
| uma/kk double | 4 | 4 | 1.273×10⁻¹⁰ | 0 | 1.000000 | PASS |

Thresholds (double): \|ΔE\| ≤ 1×10⁻⁸ · max\|ΔF\| ≤ 1×10⁻⁶ · cosine ≥ 1−ε.

## Path × ngpu vs ASE FP64@1

| Path | ngpu | Energy (eV) | ms/eval | \|ΔE\| | Force MAE | Force RMSE | max \|ΔFᵢ\| | Cosine |
|------|-----:|-------------:|--------:|-------:|----------:|-----------:|------------:|-------:|
| ASE FP64 | 1 | −5830.9237201666 | 396.5 | — | 0 | 0 | 0 | 1.0 |
| ASE FP64 | 2 | −5830.9237201666 | 193.9 | ~0 | 0 | 0 | 0 | 1.0 |
| ASE FP64 | 4 | −5830.9237201666 | 115.2 | ~0 | 0 | 0 | 0 | 1.0 |
| FairChem FC | 1 | −5830.9237152511 | 345.5 | 4.915×10⁻⁶ | 1.002×10⁻⁶ | 1.343×10⁻⁶ | 7.123×10⁻⁶ | 1.0 |
| FairChem FC | 2 | −5830.9237152511 | 193.2 | 4.915×10⁻⁶ | 1.002×10⁻⁶ | 1.343×10⁻⁶ | 7.123×10⁻⁶ | 1.0 |
| FairChem FC | 4 | −5830.9237152511 | 118.0 | 4.915×10⁻⁶ | 1.002×10⁻⁶ | 1.343×10⁻⁶ | 7.123×10⁻⁶ | 1.0 |
| uma/kk double | 1 | −5830.9237201667 | 320.4 | 1.23×10⁻¹⁰ | 1.516×10⁻⁷ | 2.179×10⁻⁷ | 5.000×10⁻⁷ | 1.0 |
| uma/kk double | 2 | −5830.9237201666 | 192.0 | 1.82×10⁻¹² | 1.516×10⁻⁷ | 2.179×10⁻⁷ | 5.000×10⁻⁷ | 1.0 |
| uma/kk double | 4 | −5830.9237201666 | 112.6 | 9.09×10⁻¹³ | 1.516×10⁻⁷ | 2.179×10⁻⁷ | 5.000×10⁻⁷ | 1.0 |

## Notes

- Honest ms from ASE/uma64 log lines and `fc_result_early.json` — not SLURM wall/`N_TIMING`.
- FC \|ΔE\| ~5×10⁻⁶ from FP32 cell construction in the LAMMPS bridge.
- uma forces vs ASE max\|ΔF\| = 5×10⁻⁷; forces identical across devices=1/2/4 (max\|ΔF\| vs d1 = 0).
