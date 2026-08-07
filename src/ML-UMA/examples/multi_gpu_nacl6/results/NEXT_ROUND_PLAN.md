# PENDING — report close-out

> **Precision:** double/FP64 only. **Stamp:** 2026-08-07 ~17:50 CDT

| # | Item | Jobs | Status |
|---|------|------|--------|
| 1 | ASE FP64@1 E/F oracle | oracle + `20910344` timing | **DONE** |
| 2 | ASE timing/E @ 1/2/4 | `20910344` / `48` / `52` | **DONE** (396.5 / 193.9 / 115.2 ms) |
| 3 | FC timing @ 1/2 | `20910345` / `49` | **DONE** (345.5 / 193.2 ms) |
| 4 | FC @ 4 | `20910353` | **PENDING** |
| 5 | uma double GP 1/2/4 | gp_round | **DONE** |
| 6 | uma @4 requeue | `20910354` | PENDING (afterok #4; optional) |
| G1 | OOM sweep | see multi_node | **N\*=10** |

When #4 completes: fill FC @4 into RESULTS/SUMMARY/canvas; stop poll loops.
