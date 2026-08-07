# UMA multi-GPU NaCl 6×6×6 parity report

> **Note (2026-08-07):** Mixed precision (`uma/kk mixed`) is **disabled**.

**Stamp:** 2026-08-07 ~17:42 CDT · Prefer [`RESULTS.md`](RESULTS.md) as canonical.

## Setup

- **System:** NaCl 6×6×6 rocksalt, 1728 atoms
- **Structure (immutable):** `structures/nacl6_rattle_fixed.extxyz`
- **Reference:** ASE FairChem FP64 @ ngpu=1 (`oracle_ase_fp64_w1`)
- **ASE@ngpu1 energy:** `-5830.9237201666` eV
- **uma GP:** GraphParallelRuntime devices=N (same-node)
- **Precision:** ASE + uma/kk double = FP64

## Canonical timing (ms/eval)

| Path | 1 GPU | 2 GPU | 4 GPU | 1→4 |
|------|------:|------:|------:|----:|
| ASE FairChem FP64 | 396.1 | 191.9 | 117.7 | 3.37× |
| FairChem FC | 345.0 | 194.8 | PENDING | — |
| uma double (GP) | 322.2 | 192.4 | 112.6 | 2.86× |

## uma double vs ASE FP64@1

| ngpu | \|ΔE\| | max \|ΔF\| | cosine | gate |
|------|---------|------------|--------|------|
| 1 | ~1e-10 | 5.0e-7 | 1.0 | PASS |
| 2 | ~1e-12 | 5.0e-7 | 1.0 | PASS |
| 4 | ~1e-12 | 5.0e-7 | 1.0 | PASS |

## Status

- uma GP 1/2/4: **DONE**
- ASE @4 (`20909845`): **DONE**
- FC @4: **PENDING** (`20910353`; `20909846` cancelled)
