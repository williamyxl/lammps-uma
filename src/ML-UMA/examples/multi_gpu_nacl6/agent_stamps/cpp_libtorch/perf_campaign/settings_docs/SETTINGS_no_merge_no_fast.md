# Settings: no merge, no fast (`general`, `merge_mole=False`)

**Stamp:** 2026-08-09T23:24  
**FairChem:** `execution_mode=general`, `merge_mole=False`  
**uma artifact:** `uma-s-1p2-omat-f64`  
**E/F reference:** ASE `general`  

Paths: **ASE** FairChem FP64 · **FC** LAMMPS (FairChem fix external) · **uma** `uma/kk` precision double.
GPUs: 1 / 2 / 4 (ASE/FC `workers=N`; uma `devices N`, 1 MPI).

### Metrics

| Suite | Timing | Force columns |
|-------|--------|---------------|
| NaCl6 | `ms_per_eval_python` (SP) | vs reference forces; `max‖ΔF‖_atom` = max per-atom ‖ΔF‖ |
| water888 | `nvt_pair_ms_per_step` (NVT Pair) | same |

`|ΔE|` and forces are vs the settings reference above (not vs a different settings row).

## NaCl6 (1728 atoms)

| path | ngpu | status | E (eV) | \|ΔE\| (eV) | force MAE | force max\|Δ\| | max‖ΔF‖/atom | time (ms) | notes |
|------|-----:|--------|-------:|-----------:|----------:|---------------:|-------------:|----------:|-------|
| ase | 1 | DONE | -5830.9237 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 396.4798 |  |
| ase | 2 | REUSED | -5830.9237 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 193.9000 |  |
| ase | 4 | REUSED | -5830.9237 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 115.2000 |  |
| fc | 1 | DONE | -5830.9237 | 4.916e-06 | 1.002e-06 | 7.123e-06 | 7.127e-06 | 345.5284 |  |
| fc | 2 | REUSED | -5830.9237 | 4.915e-06 | — | — | — | 193.2000 | FC force vs ASE not re-extracted; E residual ~5e-6 |
| fc | 4 | REUSED | -5830.9237 | 4.915e-06 | — | — | — | 118.0000 | FC force vs ASE not re-extracted; E residual ~5e-6 |
| uma | 1 | DONE | -5830.9237 | 1.819e-12 | 1.516e-07 | 5.000e-07 | 7.673e-07 | 315.5800 | 20989773 |
| uma | 2 | REUSED_Tier0 | -5830.9237 | 0.000e+00 | ~1.5e-7 | ~5e-7 | ~7.7e-7 | 172.9000 | Tier0 vs ASE general |
| uma | 4 | REUSED_Tier0 | -5830.9237 | 0.000e+00 | ~1.5e-7 | ~5e-7 | ~7.7e-7 | 100.2000 | Tier0 vs ASE general |

### Timing summary (NaCl6)

| path | @1 | @2 | @4 |
|------|---:|---:|---:|
| ase | 396.4798 | 193.9000 | 115.2000 |
| fc | 345.5284 | 193.2000 | 118.0000 |
| uma | 315.5800 | 172.9000 | 100.2000 |

## water888 (648 atoms, NVT)

| path | ngpu | status | E (eV) | \|ΔE\| (eV) | force MAE | force max\|Δ\| | max‖ΔF‖/atom | time (ms) | notes |
|------|-----:|--------|-------:|-----------:|----------:|---------------:|-------------:|----------:|-------|
| ase | 1 | REUSED | -3143.3894 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 382.0900 |  |
| ase | 2 | REUSED | -3143.3894 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 198.1900 |  |
| ase | 4 | REUSED | -3143.3894 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 117.9800 |  |
| fc | 1 | REUSED | -3143.3894 | 1.568e-06 | — | 9.350e-05 | — | 359.4000 | max|ΔF| from COMPARE vs ASE@1 |
| fc | 2 | REUSED | -3143.3894 | 1.568e-06 | — | 9.350e-05 | — | 200.5400 | max|ΔF| from COMPARE vs ASE@1 |
| fc | 4 | REUSED | -3143.3894 | 1.568e-06 | — | 9.350e-05 | — | 118.9400 | max|ΔF| from COMPARE vs ASE@1 |
| uma | 1 | REUSED | -3143.3894 | 8.200e-12 | — | 4.962e-06 | — | 332.5100 | @2/@4 ms from Tier0; @1 from COMPARE |
| uma | 2 | REUSED | -3143.3894 | 8.200e-12 | — | 4.962e-06 | — | 178.3200 | @2/@4 ms from Tier0; @1 from COMPARE |
| uma | 4 | REUSED | -3143.3894 | 8.200e-12 | — | 4.962e-06 | — | 104.2000 | @2/@4 ms from Tier0; @1 from COMPARE |

### Timing summary (water888)

| path | @1 | @2 | @4 |
|------|---:|---:|---:|
| ase | 382.0900 | 198.1900 | 117.9800 |
| fc | 359.4000 | 200.5400 | 118.9400 |
| uma | 332.5100 | 178.3200 | 104.2000 |

## Legend

- `SKIP_ILLEGAL` — FairChem rejects this settings combo
- `SKIP_KNOWN_CRASH` — FC+`merge_mole`+FP64 crashes in FairChem MOLE merge (not fixed here)
- `REUSED` / `REUSED_Tier0` — locked campaign baseline
- `INVALID_FORCE` probe — timing recorded but E/F not trusted (e.g. W8 NCCL stream race); primary row stays prior valid gate
- Force self-parity for ASE vs its own oracle is ~1e-16 (numerical noise)

See also [`MATRIX.md`](../MATRIX.md), [`GLOSSARY.md`](../GLOSSARY.md).
