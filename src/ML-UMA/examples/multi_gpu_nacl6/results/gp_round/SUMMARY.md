# NaCl 6×6×6 multi-GPU parity summary

Fixed geometry `nacl6_rattle_fixed.extxyz` (1728 atoms, δ=0.1 Å seed=0). uma/kk uses Kokkos same-node multi-GPU (`lmp -k on g N -sf kk`, `pair_style uma/kk ... devices N`, `--ntasks=1`). ASE/FC use FairChem `workers=N` in one process. ASE + uma/kk double are FP64.

- **Reference energy:** ASE FP64 @ ngpu1 = `—` eV
- **uma d1 gate reference:** devices=1 @ ngpu1
- **ngpus present:** [1, 2, 4]
- **parity gates (uma vs d1):** 4/4 passed (all_passed=True)

### Thresholds (uma vs devices=1)

| Mode | |ΔE| max | max |ΔF| | cosine min |
|------|----------|------------|------------|
| double | 1.000e-08 | 1.000e-06 | 1.00e+00 |
| mixed | 5.000e-04 | 1.000e-05 | 1.00e+00 |

| Path | ngpu | devices | Energy (eV) | ms/eval | |dE| vs ASE@ngpu1 | |dE| vs uma@d1 | max |ΔF| vs d1 | cosine vs d1 | gate |
|------|------|---------|-------------|---------|-----------------------|---------------|----------------|--------------|------|
| uma/kk double | 1 | 1 | -5830.9237201667 | 322.2 | — | — | 0 | 1.000000 | PASS |
| uma/kk double | 2 | 2 | -5830.9237201666 | 192.4 | — | 1.282e-10 | 0 | 1.000000 | PASS |
| uma/kk double | 4 | 4 | -5830.9237201666 | 112.6 | — | 1.282e-10 | 0 | 1.000000 | PASS |
| uma/kk mixed | 1 | 1 | -5830.9819335938 | 246.4 | — | — | 0 | 1.000000 | PASS |
| uma/kk mixed | 2 | 2 | -5830.9234143138 | 148.7 | — | 1.925e-04 | 6.958e-07 | 1.000000 | PASS |
| uma/kk mixed | 4 | 4 | -5830.9235703731 | 91.2 | — | 7.663e-05 | 7.309e-07 | 1.000000 | PASS |

## Legacy force table (vs ASE)

| Path | ngpu | Energy (eV) | ms/eval | |dE| vs ASE@ngpu1 (eV) | Force MAE | Force RMSE | max abs dF_i | max norm dF_atom | Cosine |
|------|------|-------------|---------|-----------------------|----------|-----------|--------------|------------------|--------|
| uma/kk double | 1 | -5830.9237201667 | 322.2 | — | — | — | — | — | — |
| uma/kk double | 2 | -5830.9237201666 | 192.4 | — | — | — | — | — | — |
| uma/kk double | 4 | -5830.9237201666 | 112.6 | — | — | — | — | — | — |
| uma/kk mixed | 1 | -5830.9819335938 | 246.4 | — | — | — | — | — | — |
| uma/kk mixed | 2 | -5830.9234143138 | 148.7 | — | — | — | — | — | — |
| uma/kk mixed | 4 | -5830.9235703731 | 91.2 | — | — | — | — | — | — |

## Timing matrix (ms/eval)

| Path | ngpu1 | ngpu2 | ngpu4 |
|------|------|------|------|
| uma/kk double | 322.2 | 192.4 | 112.6 |
| uma/kk mixed | 246.4 | 148.7 | 91.2 |

Machine-readable: `SUMMARY.json`.
