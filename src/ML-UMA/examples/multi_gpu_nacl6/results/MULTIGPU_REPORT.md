# UMA multi-GPU NaCl 6×6×6 parity report

> **Note (2026-08-07):** Mixed precision disabled. Prefer [`RESULTS.md`](RESULTS.md).

**Stamp:** 2026-08-07 ~17:50 CDT

## Setup

- **System:** NaCl 6×6×6, 1728 atoms · `structures/nacl6_rattle_fixed.extxyz`
- **ASE/FC:** path-isolated jobs today (`20910344`–`20910352`)
- **uma GP:** `gp_round` GraphParallelRuntime

## Timing (ms/eval)

| Path | 1 GPU | 2 GPU | 4 GPU | 1→4 |
|------|------:|------:|------:|----:|
| ASE FairChem FP64 | 396.5 | 193.9 | 115.2 | 3.44× |
| FairChem FC | 345.5 | 193.2 | PENDING | — |
| uma double (GP) | 322.2 | 192.4 | 112.6 | 2.86× |

## Status

- ASE 1/2/4: **DONE**
- FC 1/2: **DONE** · FC @4: **PENDING** (`20910353`)
- uma double 1/2/4: **DONE**
