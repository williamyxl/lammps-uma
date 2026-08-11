# Four-path comparison — energy, per-atom force, timing

**Generated:** 2026-08-10 22:16:28  
**Precision:** FP64 · **Ensemble:** NVT 300 K · **Oracle:** ASE FP64 `umas_fast_pytorch`+`merge_mole`

Parity gate: `|dE| <= 1e-6 eV` **and** per-atom `max|dF| <= 1e-5 eV/A`. Net force `|sum F|` is deliberately not used: it is bit-identical under sign inversion.

## nacl6 (1728 atoms)

Oracle energy: `-5830.923741338` eV

### Energy + per-atom force parity

| Path | GPUs | Energy (eV) | \|dE\| | max/atom \|dF\| | mean \|dF\| | >tol | Verdict |
|---|---:|---:|---:|---:|---:|---:|:---:|
| ALCHEMI (nvalchemi) | 1 | -5830.923741 | 1.70e-10 | 1.69e-14 | 3.36e-15 | 0 | **PASS** |
| ALCHEMI (nvalchemi) | 2 | -5830.923741 | 1.67e-10 | 1.69e-14 | 3.36e-15 | 0 | **PASS** |
| LibTorch UMA LAMMPS (W8nk) | 1 | -5830.923741 | 4.46e-11 | 7.79e-07 | 3.33e-07 | 0 | **PASS** |
| LibTorch UMA LAMMPS (W8nk) | 2 | -5830.923741 | 4.64e-11 | 7.79e-07 | 3.33e-07 | 0 | **PASS** |
| LibTorch UMA LAMMPS (W8nk) | 4 | -5830.923741 | 4.55e-11 | 7.79e-07 | 3.33e-07 | 0 | **PASS** |
| LibTorch UMA V7 [w18] | 1 | -5830.923741 | 4.37e-11 | 7.79e-07 | 3.33e-07 | 0 | **PASS** |
| LibTorch UMA V7 [w18] | 2 | -5830.923741 | 4.64e-11 | 7.79e-07 | 3.33e-07 | 0 | **PASS** |
| LibTorch UMA V7 [w18] | 4 | -5830.923741 | 4.64e-11 | 7.79e-07 | 3.33e-07 | 0 | **PASS** |

### Timing (ms/step or ms/eval)

| Path | @1 | @2 | @4 | Metric |
|---|---:|---:|---:|---|
| ASE FC FP64 (general) | 396.5 | 193.9 | 115.2 | locked baseline (frozen) |
| ASE FC FP64 (ufast+merge) | 350.0 | 191.6 | 164.5 | locked baseline (frozen) |
| FC LAMMPS (general) | 345.5 | 193.2 | 118.0 | locked baseline (frozen) |
| ALCHEMI (nvalchemi) | 355.4 | - | - | in-code NVT ms/step (warmup excl.) |
| LibTorch UMA LAMMPS (W8nk) | 296.5 | 161.9 | 92.1 | LAMMPS Pair ms/step |
| LibTorch UMA V7 [w18] | 295.7 | 160.8 | 94.8 | LAMMPS Pair ms/step (+shell) |

## water888 (648 atoms)

Oracle energy: `-3143.389377472` eV

### Energy + per-atom force parity

| Path | GPUs | Energy (eV) | \|dE\| | max/atom \|dF\| | mean \|dF\| | >tol | Verdict |
|---|---:|---:|---:|---:|---:|---:|:---:|
| ALCHEMI (nvalchemi) | 1 | -3143.382963 | 6.41e-03 | 1.01e-02 | 5.66e-04 | 648 | **FAIL** |
| ALCHEMI (nvalchemi) | 2 | -3143.382963 | 6.41e-03 | 1.01e-02 | 5.66e-04 | 648 | **FAIL** |
| LibTorch UMA LAMMPS (W8nk) | 2 | -3143.389377 | 8.19e-12 | 6.64e-06 | 1.82e-06 | 0 | **PASS** |
| LibTorch UMA LAMMPS (W8nk) | 4 | -3143.389377 | 8.19e-12 | 6.64e-06 | 1.82e-06 | 0 | **PASS** |
| LibTorch UMA V7 [w18] | 1 | -3143.389377 | 8.19e-12 | 6.64e-06 | 1.82e-06 | 0 | **PASS** |
| LibTorch UMA V7 [w18] | 2 | -3143.389377 | 8.19e-12 | 6.64e-06 | 1.82e-06 | 0 | **PASS** |
| LibTorch UMA V7 [w18] | 4 | -3143.389377 | 8.19e-12 | 6.64e-06 | 1.82e-06 | 0 | **PASS** |

### Timing (ms/step or ms/eval)

| Path | @1 | @2 | @4 | Metric |
|---|---:|---:|---:|---|
| ASE FC FP64 (general) | 382.1 | 198.2 | 118.0 | locked baseline (frozen) |
| ASE FC FP64 (ufast+merge) | - | 165.5 | 94.5 | locked baseline (frozen) |
| FC LAMMPS (general) | 359.4 | 200.5 | 118.9 | locked baseline (frozen) |
| ALCHEMI (nvalchemi) | 303.9 | - | - | in-code NVT ms/step (warmup excl.) |
| LibTorch UMA LAMMPS (W8nk) | - | 164.8 | 95.7 | LAMMPS Pair ms/step |
| LibTorch UMA V7 [w18] | 304.5 | 167.1 | 99.0 | LAMMPS Pair ms/step (+shell) |

## V7 worker step breakdown (median ms/rank)

| wave | sys | GPUs | h2d | shard | pad | prep | fwd | barrier | bwd | post | accounted | wait | resid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w18 | nacl6 | 2 | 0.08 | 1.04 | 0.00 | 0.04 | 69.03 | 0.03 | 16.87 | 70.14 | 157.25 | 157.82 | 0.57 |
| w18 | nacl6 | 4 | 0.08 | 0.84 | 0.00 | 0.04 | 40.36 | 0.03 | 17.24 | 30.41 | 89.22 | 89.75 | 0.53 |
| w18 | water888 | 2 | 0.07 | 1.10 | 0.00 | 0.04 | 71.61 | 0.03 | 17.06 | 72.66 | 162.40 | 162.82 | 0.42 |
| w18 | water888 | 4 | 0.08 | 1.13 | 0.00 | 0.04 | 42.02 | 0.03 | 17.17 | 31.94 | 92.60 | 93.09 | 0.50 |

## Notes and caveats

- **ASE FC / FC LAMMPS are frozen.** Their code is unchanged, so locked bars are reused rather than re-run.
- **FC LAMMPS has no FP64+`merge_mole` row**: `merge_MOLE` raises a Float/Double error, so the matching-settings bar is blocked upstream.
- **Metrics are not interchangeable.** LAMMPS paths report the internal Pair timer; V7 additionally reports a shell-level differential measured in the SLURM script, external to LAMMPS; ALCHEMI reports an in-code timer. Compare within a column.
- **ALCHEMI multi-GPU** is spatial domain decomposition (halo); LibTorch UMA is model-parallel with NCCL. A speed gap between them is not a like-for-like implementation comparison.
- **ALCHEMI water888 fails parity** (dE 6.4e-3, all 648 atoms over tol) while nacl6 passes at 1.7e-14. Isolated to nvalchemi's own input adaptation: plain fairchem on the identical structure passes at 1.4e-7, and TF32 was ruled out. Upstream defect.
