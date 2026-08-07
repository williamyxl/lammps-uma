# Report close-out

> **Precision:** double/FP64 only. **Stamp:** 2026-08-07 ~18:00 CDT

| # | Item | Jobs | Status |
|---|------|------|--------|
| 1 | ASE FP64@1 E/F oracle | oracle + `20910344` | **DONE** |
| 2 | ASE timing/E/F @ 1/2/4 | `20910344` / `48` / `52` | **DONE** (396.5 / 193.9 / 115.2 ms) |
| 3 | FC timing/E/F @ 1/2/4 | `20910345` / `49` / `53` | **DONE** (345.5 / 193.2 / 118.0 ms) |
| 4 | uma_double timing/E/F @ 1/2/4 | `20910346` / `50` / `54` | **DONE** (320.4 / 192.0 / 112.6 ms; GP 3/3 PASS) |
| G1 | OOM sweep | multi_node | **N\*=10** |

Path-isolated campaign complete. Prefer [`RESULTS.md`](RESULTS.md) + canvas over SLURM-wall `SUMMARY.json` timings.
