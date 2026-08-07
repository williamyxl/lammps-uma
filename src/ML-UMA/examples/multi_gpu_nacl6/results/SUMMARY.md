# NaCl 6×6×6 multi-GPU parity summary

> **Note (2026-08-07):** Mixed precision (`uma/kk mixed`) is **disabled**.

**Stamp:** 2026-08-07 ~17:42 CDT · Canonical detail: [`RESULTS.md`](RESULTS.md)

Fixed geometry `structures/nacl6_rattle_fixed.extxyz` (1728 atoms).  
**uma GP** timings from `gp_round/` (GraphParallelRuntime). ASE/FC from path-isolated jobs.

- **Reference energy:** ASE FP64 @1 = `-5830.9237201666` eV (`oracle_ase_fp64_w1`)
- **uma double 1/2/4:** DONE (E + forces vs ASE; gates PASS)
- **ASE 1/2/4 timing:** DONE (396.1 / 191.9 / **117.7** ms)
- **FC 1/2:** DONE · **FC @4:** PENDING

### Timing (ms/eval) — canonical

| Path | 1 GPU | 2 GPU | 4 GPU | 1→4 |
|------|------:|------:|------:|----:|
| ASE FairChem FP64 | 396.1 | 191.9 | 117.7 | 3.37× |
| FairChem FC | 345.0 | 194.8 | — | — |
| uma double (GP) | 322.2 | 192.4 | 112.6 | 2.86× |

### uma double vs ASE FP64@1

| ngpu | Energy (eV) | \|ΔE\| | max \|ΔF\| | cosine |
|------|-------------|---------|------------|--------|
| 1 | −5830.9237201667 | ~1e-10 | 5.0e-7 | 1.0 |
| 2 | −5830.9237201666 | ~1e-12 | 5.0e-7 | 1.0 |
| 4 | −5830.9237201666 | ~1e-12 | 5.0e-7 | 1.0 |

### Thresholds (uma vs devices=1)

| Mode | \|ΔE\| max | max \|ΔF\| | cosine min |
|------|----------|------------|------------|
| double | 1.000e-08 | 1.000e-06 | 1.00e+00 |

Machine-readable: `SUMMARY.json` (may lag this markdown until next collect).
