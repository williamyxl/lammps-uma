# Settings: merge, no fast (`general`, `merge_mole=True`)

**Stamp:** 2026-08-09T20:41  
**FairChem:** `execution_mode=general`, `merge_mole=True`  
**uma artifact:** `uma-s-1p2-omat-f64-merge`  
**E/F reference:** ASE `general`+`merge_mole`  

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
| ase | 1 | DONE | -5830.9237 | 0.000e+00 | 1.174e-16 | 5.829e-16 | 6.111e-16 | 380.7138 | 20989758 |
| ase | 2 | DONE | -5830.9237 | 4.547e-12 | 1.113e-16 | 5.551e-16 | 6.799e-16 | 195.7000 | 20989338 |
| ase | 4 | DONE | -5830.9237 | 5.457e-12 | 1.200e-16 | 6.106e-16 | 6.650e-16 | 167.9000 | 20989346 |
| fc | 1 | SKIP_KNOWN_CRASH | — | — | — | — | — | — | FC+merge_mole+FP64: FairChem Float vs Double in merge_MOLE |
| fc | 2 | SKIP_KNOWN_CRASH | — | — | — | — | — | — | FC+merge_mole+FP64: FairChem Float vs Double in merge_MOLE |
| fc | 4 | SKIP_KNOWN_CRASH | — | — | — | — | — | — | FC+merge_mole+FP64: FairChem Float vs Double in merge_MOLE |
| uma | 1 | DONE | -5830.9237 | 2.120e-05 | — | — | — | 315.4800 | 20989768 E matches general not merge — devices=1 merge art suspect |
| uma | 2 | REUSED_Tier1c | -5830.9237 | 1.670e-10 | — | 5.000e-07 | — | 169.8500 | 20983451 vs merge ASE |
| uma | 4 | DONE | -5830.9237 | 1.230e-10 | 1.520e-07 | 5.000e-07 | 7.790e-07 | 96.4700 | 20989769 vs merge ASE forces |

### Timing summary (NaCl6)

| path | @1 | @2 | @4 |
|------|---:|---:|---:|
| ase | 380.7138 | 195.7000 | 167.9000 |
| fc | SKIP_KNOWN_CRASH | SKIP_KNOWN_CRASH | SKIP_KNOWN_CRASH |
| uma | 315.4800 | 169.8500 | 96.4700 |

## water888 (648 atoms, NVT)

| path | ngpu | status | E (eV) | \|ΔE\| (eV) | force MAE | force max\|Δ\| | max‖ΔF‖/atom | time (ms) | notes |
|------|-----:|--------|-------:|-----------:|----------:|---------------:|-------------:|----------:|-------|
| ase | 1 | DONE | -3143.3894 | 4.547e-13 | 5.092e-16 | 2.831e-15 | 3.659e-15 | 353.9000 | 20989760 |
| ase | 2 | DONE | -3143.3894 | 4.547e-13 | 6.007e-16 | 4.996e-15 | 5.358e-15 | 174.0000 | 20989762 |
| ase | 4 | DONE | -3143.3894 | 4.547e-13 | 6.216e-16 | 4.663e-15 | 4.930e-15 | 98.5000 | 20989764 |
| fc | 1 | SKIP_KNOWN_CRASH | — | — | — | — | — | — | FC+merge_mole+FP64: FairChem Float vs Double in merge_MOLE |
| fc | 2 | SKIP_KNOWN_CRASH | — | — | — | — | — | — | FC+merge_mole+FP64: FairChem Float vs Double in merge_MOLE |
| fc | 4 | SKIP_KNOWN_CRASH | — | — | — | — | — | — | FC+merge_mole+FP64: FairChem Float vs Double in merge_MOLE |
| uma | 1 | DONE | -3143.3894 | 2.135e-05 | — | — | — | 326.5000 | 20989772 @1 E~general ASE (merge residual absent) |
| uma | 2 | DONE | -3143.3894 | 8.185e-12 | 7.437e-07 | 4.993e-06 | 6.639e-06 | 186.4500 | 20989770 |
| uma | 4 | DONE | -3143.3894 | 8.640e-12 | 7.437e-07 | 4.993e-06 | 6.639e-06 | 101.0500 | 20989771 |

### Timing summary (water888)

| path | @1 | @2 | @4 |
|------|---:|---:|---:|
| ase | 353.9000 | 174.0000 | 98.5000 |
| fc | SKIP_KNOWN_CRASH | SKIP_KNOWN_CRASH | SKIP_KNOWN_CRASH |
| uma | 326.5000 | 186.4500 | 101.0500 |

## Legend

- `SKIP_ILLEGAL` — FairChem rejects this settings combo
- `SKIP_KNOWN_CRASH` — FC+`merge_mole`+FP64 crashes in FairChem MOLE merge (not fixed here)
- `REUSED` / `REUSED_Tier0` — locked campaign baseline
- `INVALID_FORCE` probe — timing recorded but E/F not trusted (e.g. W8 NCCL stream race); primary row stays prior valid gate
- Force self-parity for ASE vs its own oracle is ~1e-16 (numerical noise)

See also [`MATRIX.md`](../MATRIX.md), [`GLOSSARY.md`](../GLOSSARY.md).
