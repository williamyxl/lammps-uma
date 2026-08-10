# Settings: fast and merge (`umas_fast_pytorch`, `merge_mole=True`)

**Stamp:** 2026-08-09T21:04  
**FairChem:** `execution_mode=umas_fast_pytorch`, `merge_mole=True`  
**uma artifact:** `uma-s-1p2-omat-f64-fast`  
**E/F reference:** ASE `umas_fast_pytorch`+`merge_mole`  

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
| ase | 1 | DONE | -5830.9237 | 0.000e+00 | 1.164e-16 | 5.551e-16 | 6.795e-16 | 350.3438 | 20989759 |
| ase | 2 | DONE | -5830.9237 | 1.819e-12 | 1.181e-16 | 6.661e-16 | 6.794e-16 | 191.6000 | 20989339 |
| ase | 4 | DONE | -5830.9237 | 2.728e-12 | 1.701e-16 | 8.604e-16 | 8.968e-16 | 164.5000 | 20989347 |
| fc | 1 | SKIP_KNOWN_CRASH | — | — | — | — | — | — | FC+merge_mole+FP64 crash |
| fc | 2 | SKIP_KNOWN_CRASH | — | — | — | — | — | — | FC+merge_mole+FP64 crash |
| fc | 4 | SKIP_KNOWN_CRASH | — | — | — | — | — | — | FC+merge_mole+FP64 crash |
| uma | 1 | DONE_FLOOR_FAIL | -5830.9237 | 2.120e-05 | — | — | — | 533.1000 | 20989766 devices=1: slow + E~general |
| uma | 2 | DONE_W8 | -5830.9237 | 1.255e-10 | 1.518e-07 | 5.000e-07 | 7.793e-07 | 160.0800 | 20989976 W8 stream-ordered NCCL W8 INVALID_FORCE ms=160.19 (20989797): forces absmax=6.213e+11; NCCL dedicated-stream race; forces garbage — do not promote |
| uma | 4 | DONE_W7 | -5830.9237 | 1.230e-10 | 1.520e-07 | 5.000e-07 | 7.790e-07 | 92.3700 | 20989185 W8 INVALID_FORCE ms=90.45 (20989798): forces absmax=5.165e+06; NCCL dedicated-stream race; forces garbage — do not promote |

### Timing summary (NaCl6)

| path | @1 | @2 | @4 |
|------|---:|---:|---:|
| ase | 350.3438 | 191.6000 | 164.5000 |
| fc | SKIP_KNOWN_CRASH | SKIP_KNOWN_CRASH | SKIP_KNOWN_CRASH |
| uma | 533.1000 | 160.0800 | 92.3700 |

## water888 (648 atoms, NVT)

| path | ngpu | status | E (eV) | \|ΔE\| (eV) | force MAE | force max\|Δ\| | max‖ΔF‖/atom | time (ms) | notes |
|------|-----:|--------|-------:|-----------:|----------:|---------------:|-------------:|----------:|-------|
| ase | 1 | DONE | -3143.3894 | 4.547e-13 | 5.891e-16 | 2.665e-15 | 3.686e-15 | 337.6000 | 20989761 |
| ase | 2 | DONE | -3143.3894 | 0.000e+00 | 7.105e-16 | 4.552e-15 | 4.980e-15 | 165.5000 | 20989763 |
| ase | 4 | DONE | -3143.3894 | 4.547e-13 | 7.192e-16 | 5.218e-15 | 5.629e-15 | 94.5000 | 20989765 |
| fc | 1 | SKIP_KNOWN_CRASH | — | — | — | — | — | — | FC+merge_mole+FP64 crash |
| fc | 2 | SKIP_KNOWN_CRASH | — | — | — | — | — | — | FC+merge_mole+FP64 crash |
| fc | 4 | SKIP_KNOWN_CRASH | — | — | — | — | — | — | FC+merge_mole+FP64 crash |
| uma | 1 | DONE | -3143.3894 | 2.140e-05 | — | — | — | 337.7000 | 20989767 E~general |
| uma | 2 | DONE_W7 | -3143.3894 | 8.200e-12 | 7.440e-07 | 4.990e-06 | 6.640e-06 | 165.1000 | 20989186 |
| uma | 4 | DONE_W7_FLOOR_FAIL_vs_ASE | -3143.3894 | 8.200e-12 | 7.440e-07 | 4.990e-06 | 6.640e-06 | 96.8400 | 20989187 ms 96.84 > ASE ufast 94.5 |

### Timing summary (water888)

| path | @1 | @2 | @4 |
|------|---:|---:|---:|
| ase | 337.6000 | 165.5000 | 94.5000 |
| fc | SKIP_KNOWN_CRASH | SKIP_KNOWN_CRASH | SKIP_KNOWN_CRASH |
| uma | 337.7000 | 165.1000 | 96.8400 |

## Legend

- `SKIP_ILLEGAL` — FairChem rejects this settings combo
- `SKIP_KNOWN_CRASH` — FC+`merge_mole`+FP64 crashes in FairChem MOLE merge (not fixed here)
- `REUSED` / `REUSED_Tier0` — locked campaign baseline
- `INVALID_FORCE` probe — timing recorded but E/F not trusted (e.g. W8 NCCL stream race); primary row stays prior valid gate
- Force self-parity for ASE vs its own oracle is ~1e-16 (numerical noise)

See also [`MATRIX.md`](../MATRIX.md), [`GLOSSARY.md`](../GLOSSARY.md).
