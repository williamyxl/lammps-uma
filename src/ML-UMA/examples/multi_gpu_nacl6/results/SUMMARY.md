# NaCl 6×6×6 multi-GPU parity summary

> **Note (2026-08-07):** Mixed precision (`uma/kk mixed`) is **disabled**.

**Stamp:** 2026-08-07 ~17:50 CDT · Canonical: [`RESULTS.md`](RESULTS.md)

Fixed geometry `structures/nacl6_rattle_fixed.extxyz` (1728 atoms).

- **ASE FP64 E/F oracle @1:** `-5830.9237201666` eV (`oracle_ase_fp64_w1`)
- **ASE timing 1/2/4:** **DONE** today (`20910344`/`48`/`52`)
- **FC timing 1/2:** **DONE** (`20910345`/`49`) · **@4 PENDING** (`20910353`)
- **uma double GP 1/2/4:** DONE

### Timing (ms/eval)

| Path | 1 GPU | 2 GPU | 4 GPU | 1→2 | 1→4 |
|------|------:|------:|------:|----:|----:|
| ASE FairChem FP64 | 396.5 | 193.9 | 115.2 | 2.04× | 3.44× |
| FairChem FC | 345.5 | 193.2 | — | 1.79× | — |
| uma double (GP) | 322.2 | 192.4 | 112.6 | 1.67× | 2.86× |

### uma double vs ASE FP64@1

| ngpu | \|ΔE\| | max \|ΔF\| | cosine |
|------|---------|------------|--------|
| 1 | ~1e-10 | 5.0e-7 | 1.0 |
| 2 | ~1e-12 | 5.0e-7 | 1.0 |
| 4 | ~1e-12 | 5.0e-7 | 1.0 |
