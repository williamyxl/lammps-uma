# UMA multi-GPU NaCl 6×6×6 parity report

## Honest path-isolated timing (2026-08-07 ~17:55 CDT)

Prefer these over auto-merged SLURM wall/`N_TIMING` values in the tables below. Full write-up: [`RESULTS.md`](RESULTS.md).

| Path | 1 GPU | 2 GPU | 4 GPU | 1→2 | 1→4 |
|------|------:|------:|------:|----:|----:|
| ASE FairChem FP64 | 396.5 ms | 193.9 ms | 115.2 ms | 2.04× | 3.44× |
| FairChem FC | 345.5 ms | 193.2 ms | 118.0 ms | 1.79× | 2.93× |
| uma double (GP) | 322.2 ms | 192.4 ms | 112.6 ms | 1.67× | 2.86× |

FC @4 job `20910353` COMPLETED · `ngpu4/work/fc/fc_result_early.json` = **118.0 ms**.

## Setup

- **System:** NaCl 6×6×6 rocksalt, 1728 atoms
- **Structure (immutable):** `src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz`
- **Perturbation:** uniform_box δ=0.1 Å seed=0 (fixed extxyz — never re-rattle)
- **Reference:** ASE FairChem FP64 @ ngpu=1
- **ASE@ngpu1 energy:** `-5830.9237201666` eV
- **uma/kk launch:** `lmp -k on g ${NGPUS} -sf kk (ntasks=1, no MPI multi-GPU)`
- **uma pair_style:** `pair_style uma/kk precision <mode> devices ${UMA_DEVICES}`
- **ASE/FC:** `workers=${NGPUS} in one process`
- **Precision:** ASE + uma/kk double = FP64
- **Parity gates (uma vs devices=1):** 2/2 passed (all_passed=True)

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
| uma/kk double | 2 | 2 | 1.273e-10 | 0 | 1.000000 | PASS |
| uma/kk mixed | 1 | 1 | — | 0 | 1.000000 | PASS |
| uma/kk mixed | 2 | 2 | 2.968e-04 | 7.010e-07 | 1.000000 | PASS |

## Path × ngpu table (vs ASE)

| Path | ngpu | Energy (eV) | ms/eval | |dE| vs ASE@ngpu1 | Force MAE | Force RMSE | max abs dF_i | Cosine |
|------|------|-------------|---------|------------------|----------|-----------|--------------|--------|
| ASE FP64 | 1 | -5830.9237201666 | 8.71 s | — | — | — | — | — |
| ASE FP64 | 2 | -5830.9237201666 | 17.55 s | 2.728e-12 | 0 | 0 | 0 | 1.000000 |
| ASE FP64 | 4 | -5830.9237201666 | 11.26 s | 3.638e-12 | 0 | 0 | 0 | 1.000000 |
| FairChem FC | 1 | -5830.9237152511 | 4.65 s | 4.915e-06 | — | — | — | — |
| FairChem FC | 2 | -5830.9237152511 | 10.36 s | 4.915e-06 | — | — | — | — |
| FairChem FC | 4 | -5830.9237152511 | 9.86 s | 4.915e-06 | — | — | — | — |
| uma/kk double | 1 | -5830.9237201667 | 6.15 s | 1.228e-10 | 1.516e-07 | 2.179e-07 | 5.000e-07 | 1.000000 |
| uma/kk double | 2 | -5830.9237201666 | 24.00 s | 4.547e-12 | 1.516e-07 | 2.179e-07 | 5.000e-07 | 1.000000 |
| uma/kk mixed | 1 | -5830.9824218750 | 5.84 s | 5.870e-02 | 1.055e-06 | 1.397e-06 | 7.243e-06 | 1.000000 |
| uma/kk mixed | 2 | -5830.9234688666 | 24.28 s | 2.513e-04 | 1.061e-06 | 1.404e-06 | 7.243e-06 | 1.000000 |

## Timing (ms/eval)

| Path | ngpu1 | ngpu2 | ngpu4 |
|------|------|------|------|
| ASE FP64 | 8.71 s | 17.55 s | 11.26 s |
| FairChem FC | 4.65 s | 10.36 s | 9.86 s |
| uma/kk double | 6.15 s | 24.00 s | — |
| uma/kk mixed | 5.84 s | 24.28 s | — |

## Energy |dE| vs ASE@ngpu1 (eV)

| Path | ngpu1 | ngpu2 | ngpu4 |
|------|------|------|------|
| ASE FP64 | — | 2.728e-12 | 3.638e-12 |
| FairChem FC | 4.915e-06 | 4.915e-06 | 4.915e-06 |
| uma/kk double | 1.228e-10 | 4.547e-12 | — |
| uma/kk mixed | 5.870e-02 | 2.513e-04 | — |

## Notes

- Kokkos multi-GPU is **same-node** via `-k on g N`; SLURM uses `--ntasks=1` (no `srun -n N` / mpirun across GPUs).
- uma/kk graph-parallel: `pair_style uma/kk precision <mode> devices N` shards UMA inference across GPUs (engine GP runtime).
- ASE and FairChem FC use FairChem `workers=N` in a single process.
- Machine-readable merge: `results/SUMMARY.json`.
