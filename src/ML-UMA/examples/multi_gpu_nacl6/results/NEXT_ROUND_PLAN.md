# Report close-out

> **Precision:** double/FP64 only. **Stamp:** 2026-08-08 ~19:00 CDT  
> Canonical product: [`RESULTS.md`](RESULTS.md) — **321.04 / 183.30 / 112.04** (P3c NCCL `20940474`) · campaign **PASS**.

| # | Item | Jobs | Status |
|---|------|------|--------|
| 1 | ASE FP64@1 E/F oracle | oracle + `20910344` | **DONE** |
| 2 | ASE timing/E/F @ 1/2/4 | `20910344` / `48` / `52` | **DONE** (396.5 / 193.9 / 115.2 ms) |
| 3 | FC timing/E/F @ 1/2/4 | `20910345` / `49` / `53` | **DONE** (345.5 / 193.2 / 118.0 ms) |
| 4 | uma_double timing/E/F @ 1/2/4 (historical Ray-era) | `20910346` / `50` / `54` | **DONE** (320.4 / 192.0 / 112.6 ms) |
| 5 | Product Kokkos+LibTorch P3c NCCL | `20940474` | **DONE — campaign PASS** (321.0 / 183.3 / 112.0) |
| G1 | OOM sweep | multi_node | **N\*=10** |

Prefer [`RESULTS.md`](RESULTS.md) + canvas. Honest ms from `uma64` — not SLURM wall/`N_TIMING`.
