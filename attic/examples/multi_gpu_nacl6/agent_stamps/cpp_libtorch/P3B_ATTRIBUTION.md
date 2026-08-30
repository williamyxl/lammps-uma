# P3b — @4 residual attribution (job `20934280`)

**Stamp:** 2026-08-08 ~10:56 CDT  
**REPORT_OWNER=parent** (no RESULTS/SUMMARY/MULTIGPU/canvas edits)

## P3a honest ms (cuda_ipc)

| devices | ms | vs ASE | vs FC |
|--------:|---:|-------:|------:|
| 1 | 320.60 | — | — |
| 2 | 183.57 | −10.3 | −9.6 |
| 4 | **117.63** | **+2.43 FAIL** | −0.37 PASS |

E+F green · self-scale green · beat P2 @4. Hard campaign bar still **FAIL @4 vs ASE**.

Machine-readable: `perf/p3b_attribution_20934280.json`.

## Warm @4 breakdown (gen≥3 / skip 2 cold ticks)

| Source | Component | mean ms |
|--------|-----------|--------:|
| PERF_PARENT | `ms_nl` | 8.32 |
| PERF_PARENT | `ms_pub` | 1.51 |
| PERF_PARENT | `ms_wait_workers` | 106.74 |
| PERF_PARENT | `ms_total` | 116.59 |
| PERF_TICK r0 | `ms_fwd` | 47.02 |
| PERF_TICK r0 | `ms_bwd` | 58.45 |
| PERF_TICK r0 | `ms_force_ar` | 0.39 |
| PERF_TICK r0 | `ms_compute` | 105.86 |
| honest pair | `uma64` | 117.63 |

## Findings

1. **`ms_wait_workers ≈ ms_compute`** — parent wait is worker compute-bound, not pipe-idle.
2. **`ms_force_ar ~0.4 ms`** — CUDA IPC force allreduce is not the ASE gap.
3. **Parent `ms_nl+ms_pub ~9.8 ms`** on top of wait; honest pair tracks `ms_total`.
4. **~2.4 ms vs ASE** must come from cutting mid-graph peer traffic inside `ms_fwd`/`ms_bwd` and/or `ms_nl`.
5. **P3c NCCL** targets `uma_peer` all_gather/all_reduce (inside fwd/bwd); keep cuda_ipc default/fallback.
