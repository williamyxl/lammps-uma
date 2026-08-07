# Report close-out

> **Precision:** double/FP64 only. **Stamp:** 2026-08-07 ~17:55 CDT

| # | Item | Jobs | Status |
|---|------|------|--------|
| 1 | ASE FP64@1 E/F oracle | oracle + `20910344` timing | **DONE** |
| 2 | ASE timing/E @ 1/2/4 | `20910344` / `48` / `52` | **DONE** (396.5 / 193.9 / 115.2 ms) |
| 3 | FC timing @ 1/2/4 | `20910345` / `49` / `53` | **DONE** (345.5 / 193.2 / 118.0 ms) |
| 4 | uma double GP 1/2/4 | gp_round | **DONE** |
| 5 | uma @4 requeue | `20910354` | RUNNING (optional; gp_round already has @4) |
| G1 | OOM sweep | see multi_node | **N\*=10** |

Path-isolated timing campaign complete for ASE/FC/uma GP. Prefer `RESULTS.md` + canvas over SLURM-wall `SUMMARY.json` `ms_per_eval`.
