# NaCl 6×6×6 multi-GPU parity summary

Fixed geometry `nacl6_rattle_fixed.extxyz` (1728 atoms, δ=0.1 Å seed=0). uma/kk uses Kokkos same-node multi-GPU (`lmp -k on g N -sf kk`, `pair_style uma/kk ... devices N`, `--ntasks=1`). ASE/FC use FairChem `workers=N` in one process. ASE + uma/kk double are FP64.

- **Reference energy:** ASE FP64 @ ngpu1 = `-5830.9237201666` eV
- **uma d1 gate reference:** devices=1 @ ngpu1
- **ngpus present:** [1, 2, 4]
- **parity gates (uma vs d1):** 1/3 passed (all_passed=False)

### Thresholds (uma vs devices=1)

| Mode | |ΔE| max | max |ΔF| | cosine min |
|------|----------|------------|------------|
| double | 1.000e-08 | 1.000e-06 | 1.00e+00 |
| mixed | 5.000e-04 | 1.000e-05 | 1.00e+00 |

| Path | ngpu | devices | Energy (eV) | ms/eval | |dE| vs ASE@ngpu1 | |dE| vs uma@d1 | max |ΔF| vs d1 | cosine vs d1 | gate |
|------|------|---------|-------------|---------|-----------------------|---------------|----------------|--------------|------|
| ASE FP64 | 1 | 4 | -5830.9237201666 | 8.71 s | — | — | — | — | — |
| ASE FP64 | 2 | 4 | -5830.9237201666 | 17.55 s | 2.728e-12 | — | — | — | — |
| ASE FP64 | 4 | 4 | -5830.9237201666 | 11.26 s | 3.638e-12 | — | — | — | — |
| FairChem FC | 1 | 4 | -5830.9237152511 | 4.65 s | 4.915e-06 | — | — | — | — |
| FairChem FC | 2 | 4 | -5830.9237152511 | 10.36 s | 4.915e-06 | — | — | — | — |
| FairChem FC | 4 | 4 | -5830.9237152511 | 9.86 s | 4.915e-06 | — | — | — | — |
| uma/kk double | 1 | 1 | -5830.9237201667 | 6.15 s | 1.228e-10 | — | 0 | 1.000000 | PASS |
| uma/kk double | 2 | 2 | -5830.9237413382 | 10.42 s | 2.117e-05 | 2.117e-05 | 1.000e-06 | 1.000000 | FAIL |
| uma/kk double | 4 | 4 | -5830.9237413382 | 12.65 s | 2.117e-05 | 2.117e-05 | 1.000e-06 | 1.000000 | FAIL |
| uma/kk mixed | 1 | 1 | -5830.9824218750 | 5.84 s | 5.870e-02 | — | 0 | 1.000000 | PASS |
| uma/kk mixed | 2 | 2 | -5830.9234688666 | 24.28 s | 2.513e-04 | 2.968e-04 | 7.010e-07 | 1.000000 | PASS |

## Legacy force table (vs ASE)

| Path | ngpu | Energy (eV) | ms/eval | |dE| vs ASE@ngpu1 (eV) | Force MAE | Force RMSE | max abs dF_i | max norm dF_atom | Cosine |
|------|------|-------------|---------|-----------------------|----------|-----------|--------------|------------------|--------|
| ASE FP64 | 1 | -5830.9237201666 | 8.71 s | — | — | — | — | — | — |
| ASE FP64 | 2 | -5830.9237201666 | 17.55 s | 2.728e-12 | 0 | 0 | 0 | 0 | 1.000000 |
| ASE FP64 | 4 | -5830.9237201666 | 11.26 s | 3.638e-12 | 0 | 0 | 0 | 0 | 1.000000 |
| FairChem FC | 1 | -5830.9237152511 | 4.65 s | 4.915e-06 | — | — | — | — | — |
| FairChem FC | 2 | -5830.9237152511 | 10.36 s | 4.915e-06 | — | — | — | — | — |
| FairChem FC | 4 | -5830.9237152511 | 9.86 s | 4.915e-06 | — | — | — | — | — |
| uma/kk double | 1 | -5830.9237201667 | 6.15 s | 1.228e-10 | 1.516e-07 | 2.179e-07 | 5.000e-07 | 7.673e-07 | 1.000000 |
| uma/kk double | 2 | -5830.9237413382 | 10.42 s | 2.117e-05 | 1.561e-07 | 2.208e-07 | 5.881e-07 | 7.673e-07 | 1.000000 |
| uma/kk double | 4 | -5830.9237413382 | 12.65 s | 2.117e-05 | 1.561e-07 | 2.208e-07 | 5.881e-07 | 7.673e-07 | 1.000000 |
| uma/kk mixed | 1 | -5830.9824218750 | 5.84 s | 5.870e-02 | 1.055e-06 | 1.397e-06 | 7.243e-06 | 7.479e-06 | 1.000000 |
| uma/kk mixed | 2 | -5830.9234688666 | 24.28 s | 2.513e-04 | 1.061e-06 | 1.404e-06 | 7.243e-06 | 7.479e-06 | 1.000000 |

## Timing matrix (ms/eval)

| Path | ngpu1 | ngpu2 | ngpu4 |
|------|------|------|------|
| ASE FP64 | 8.71 s | 17.55 s | 11.26 s |
| FairChem FC | 4.65 s | 10.36 s | 9.86 s |
| uma/kk double | 6.15 s | 10.42 s | 12.65 s |
| uma/kk mixed | 5.84 s | 24.28 s | — |

Machine-readable: `SUMMARY.json`.
