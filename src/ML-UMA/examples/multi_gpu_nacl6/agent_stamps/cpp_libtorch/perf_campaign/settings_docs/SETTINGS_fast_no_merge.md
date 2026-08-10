# Settings: fast, no merge (`umas_fast_pytorch`, `merge_mole=False`)

**Stamp:** 2026-08-10T01:26  
**FairChem:** `execution_mode=umas_fast_pytorch`, `merge_mole=False`  
**uma artifact:** `n/a (illegal)`  
**E/F reference:** n/a  

Paths: **ASE** FairChem FP64 · **FC** LAMMPS (FairChem fix external) · **uma** `uma/kk` precision double.
GPUs: 1 / 2 / 4 (ASE/FC `workers=N`; uma `devices N`, 1 MPI).

> **All cells SKIP — illegal FairChem combination.** `umas_fast_pytorch` requires `merge_mole=True`. No measurements.

### Metrics

| Suite | Timing | Force columns |
|-------|--------|---------------|
| NaCl6 | `ms_per_eval_python` (SP) | vs reference forces; `max‖ΔF‖_atom` = max per-atom ‖ΔF‖ |
| water888 | `nvt_pair_ms_per_step` (NVT Pair) | same |

`|ΔE|` and forces are vs the settings reference above (not vs a different settings row).

## NaCl6 (1728 atoms)

| path | ngpu | status | E (eV) | \|ΔE\| (eV) | force MAE | force max\|Δ\| | max‖ΔF‖/atom | time (ms) | notes |
|------|-----:|--------|-------:|-----------:|----------:|---------------:|-------------:|----------:|-------|
| ase | 1 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| ase | 2 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| ase | 4 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| fc | 1 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| fc | 2 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| fc | 4 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| uma | 1 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| uma | 2 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| uma | 4 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |

### Timing summary (NaCl6)

| path | @1 | @2 | @4 |
|------|---:|---:|---:|
| ase | SKIP_ILLEGAL | SKIP_ILLEGAL | SKIP_ILLEGAL |
| fc | SKIP_ILLEGAL | SKIP_ILLEGAL | SKIP_ILLEGAL |
| uma | SKIP_ILLEGAL | SKIP_ILLEGAL | SKIP_ILLEGAL |

## water888 (648 atoms, NVT)

| path | ngpu | status | E (eV) | \|ΔE\| (eV) | force MAE | force max\|Δ\| | max‖ΔF‖/atom | time (ms) | notes |
|------|-----:|--------|-------:|-----------:|----------:|---------------:|-------------:|----------:|-------|
| ase | 1 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| ase | 2 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| ase | 4 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| fc | 1 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| fc | 2 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| fc | 4 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| uma | 1 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| uma | 2 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |
| uma | 4 | SKIP_ILLEGAL | — | — | — | — | — | — | FairChem: umas_fast_pytorch requires merge_mole=True |

### Timing summary (water888)

| path | @1 | @2 | @4 |
|------|---:|---:|---:|
| ase | SKIP_ILLEGAL | SKIP_ILLEGAL | SKIP_ILLEGAL |
| fc | SKIP_ILLEGAL | SKIP_ILLEGAL | SKIP_ILLEGAL |
| uma | SKIP_ILLEGAL | SKIP_ILLEGAL | SKIP_ILLEGAL |

## Legend

- `SKIP_ILLEGAL` — FairChem rejects this settings combo
- `SKIP_KNOWN_CRASH` — FC+`merge_mole`+FP64 crashes in FairChem MOLE merge (not fixed here)
- `REUSED` / `REUSED_Tier0` — locked campaign baseline
- `INVALID_FORCE` probe — timing recorded but E/F not trusted (e.g. W8 NCCL stream race); primary row stays prior valid gate
- Force self-parity for ASE vs its own oracle is ~1e-16 (numerical noise)

See also [`MATRIX.md`](../MATRIX.md), [`GLOSSARY.md`](../GLOSSARY.md).
