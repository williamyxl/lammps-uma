# UMA multi-GPU NaCl 6×6×6 parity report

## Setup

- **System:** NaCl 6×6×6 rocksalt, 1728 atoms
- **Structure (immutable):** `src/ML-UMA/examples/delta_parity/structures/nacl6_rattle_fixed.extxyz`
- **Perturbation:** uniform_box δ=0.1 Å seed=0 (fixed extxyz — never re-rattle)
- **Reference:** ASE FairChem FP64 @ ngpu=1
- **ASE@ngpu1 energy:** `—` eV
- **uma/kk launch:** `lmp -k on g ${NGPUS} -sf kk (ntasks=1, no MPI multi-GPU)`
- **uma pair_style:** `pair_style uma/kk precision <mode> devices ${UMA_DEVICES}`
- **ASE/FC:** `workers=${NGPUS} in one process`
- **Precision:** ASE + uma/kk double = FP64
- **Parity gates (uma vs devices=1):** 4/4 passed (all_passed=True)

Force metrics vs ASE are secondary oracle. **Primary GP gate:** uma/kk `devices=N` vs `devices=1` at same precision.

### Parity thresholds (vs devices=1)

| Mode | |ΔE| max | max |ΔF| | cosine min |
|------|----------|------------|------------|
| double | 1.000e-08 | 1.000e-06 | 1.00e+00 |
| mixed | 5.000e-04 | 1.000e-05 | 1.00e+00 |

## uma vs devices=1 (graph-parallel gate)

| Path | ngpu | devices | |dE| vs d1 | max |ΔF| vs d1 | cosine vs d1 | gate |
|------|------|---------|------------|----------------|--------------|------|
| uma/kk double | 1 | 1 | — | 0 | 1.000000 | PASS |
| uma/kk double | 2 | 2 | 1.282e-10 | 0 | 1.000000 | PASS |
| uma/kk double | 4 | 4 | 1.282e-10 | 0 | 1.000000 | PASS |
| uma/kk mixed | 1 | 1 | — | 0 | 1.000000 | PASS |
| uma/kk mixed | 2 | 2 | 1.925e-04 | 6.958e-07 | 1.000000 | PASS |
| uma/kk mixed | 4 | 4 | 7.663e-05 | 7.309e-07 | 1.000000 | PASS |

## Path × ngpu table (vs ASE)

| Path | ngpu | Energy (eV) | ms/eval | |dE| vs ASE@ngpu1 | Force MAE | Force RMSE | max abs dF_i | Cosine |
|------|------|-------------|---------|------------------|----------|-----------|--------------|--------|
| uma/kk double | 1 | -5830.9237201667 | 322.2 | — | — | — | — | — |
| uma/kk double | 2 | -5830.9237201666 | 192.4 | — | — | — | — | — |
| uma/kk double | 4 | -5830.9237201666 | 112.6 | — | — | — | — | — |
| uma/kk mixed | 1 | -5830.9819335938 | 246.4 | — | — | — | — | — |
| uma/kk mixed | 2 | -5830.9234143138 | 148.7 | — | — | — | — | — |
| uma/kk mixed | 4 | -5830.9235703731 | 91.2 | — | — | — | — | — |

## Timing (ms/eval)

| Path | ngpu1 | ngpu2 | ngpu4 |
|------|------|------|------|
| uma/kk double | 322.2 | 192.4 | 112.6 |
| uma/kk mixed | 246.4 | 148.7 | 91.2 |

## Energy |dE| vs ASE@ngpu1 (eV)

| Path | ngpu1 | ngpu2 | ngpu4 |
|------|------|------|------|
| uma/kk double | — | — | — |
| uma/kk mixed | — | — | — |

## Notes

- Kokkos multi-GPU is **same-node** via `-k on g N`; SLURM uses `--ntasks=1` (no `srun -n N` / mpirun across GPUs).
- uma/kk graph-parallel: `pair_style uma/kk precision <mode> devices N` shards UMA inference across GPUs (engine GP runtime).
- ASE and FairChem FC use FairChem `workers=N` in a single process.
- Machine-readable merge: `gp_round/SUMMARY.json`.
