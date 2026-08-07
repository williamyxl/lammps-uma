# COORD ANALYSIS — NaCl6 multi-GPU

> **Note (2026-08-07):** Mixed precision disabled. See [`RESULTS.md`](RESULTS.md) for the canonical stamp.

**Geometry:** `structures/nacl6_rattle_fixed.extxyz` (1728 atoms)

## Timing (canonical GP / FairChem workers)

| path | 1×GPU ms | 2×GPU ms | 4×GPU ms | 1→4 |
|------|--------:|--------:|--------:|----:|
| ASE FP64 | 396.1 | 191.9 | **117.7** | **3.37×** |
| FC fix-ext | 345.0 | 194.8 | PENDING | — |
| uma/kk double (GP) | 322.2 | 192.4 | 112.6 | 2.86× |

## Findings

1. ASE Ray workers scale ~2.1× @2 and **~3.4× @4**.
2. uma GraphParallelRuntime double: **~2.9× @4** (not flat Kokkos-only).
3. FC @4 still pending to close the prior-art table.
4. OOM / max-N for larger cells: **N\*=10** (see `multi_node_nacl6` SWEEP).
