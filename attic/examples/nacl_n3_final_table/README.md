# NaCl, Si, and Al MLIP single-point compares (lattice × 1.01 + rattle)

> **Note (2026-08-07):** Mixed precision (`uma/kk mixed`) is **disabled**. Mixed rows below are commented out (historical only).


Rattled geometries on a Titan V. Single-point energy + forces only (no MD / minim).

Reports live under `lammps/src/ML-UMA/examples/nacl_n3_final_table/`.

## Perturbation

<!-- DISABLED mixed: Build ideal crystal → scale lattice×positions by `1.01` → add Unif[-0.10,0.10] Å per Cartesian component (PCG64 seed=0) once → wrap into cell → freeze npz → all 4 paths (ASE / FairChem / uma double / uma mixed) load that same npz. -->

Reference: ASE `FAIRChemCalculator` FP64. Timings are repeated force evaluations; uma/kk timings use the LAMMPS Pair section over five NVE steps after warmup. FairChem `fix external` is **not** labeled FP64 — `lammps_fc` builds the cell in FP32.

**Force columns are error metrics vs ASE**, not reference force magnitudes: `max |ΔF_i|` = max component-wise |F − F_ASE|; `max ‖ΔF‖_atom` = max per-atom ‖F − F_ASE‖.

## NaCl 3×3×3 rocksalt

NaCl 3x3x3 rocksalt a=5.696400 Å (=5.64*1.01), uniform-box rattle δ=0.1 Å seed=0; 216 atoms. ASE |F|_max = 3.828e-01 eV/Å.

| Path | Energy (eV) | ms/eval | |ΔE| vs ASE FP64 (eV) | Force MAE (eV/Å) | Force RMSE (eV/Å) | max |ΔF_i| (eV/Å) | max ‖ΔF‖_atom (eV/Å) | Cosine |
|------|-------------|---------|------------------------|------------------|-------------------|---------------------|------------------------|--------|
| ASE FP64 | -729.5978639736 | 136.7 | — | — | — | — | — | 1.000000 |
| FairChem fix external | -729.5978694316 | 146.6 | 5.458e-06 | 5.740e-07 | 7.567e-07 | 3.036e-06 | 3.640e-06 | 1.000000 |
| uma/kk precision double | -729.5978639736 | 127.2 | 1.592e-12 | 1.458e-07 | 2.135e-07 | 4.990e-07 | 7.405e-07 | 1.000000 |
<!-- DISABLED mixed: | uma/kk precision mixed | -729.5968627930 | 70.1 | 1.001e-03 | 6.572e-07 | 8.454e-07 | 3.023e-06 | 3.363e-06 | 1.000000 | -->

Relative force error (uma/kk double): |ΔF|_max / |F|_ref_max = 1.304e-06 (max |ΔF_i| = 4.990e-07 eV/Å).

## Si 3×3×3 diamond

Si 3x3x3 diamond a=5.484300 Å (=5.43*1.01), uniform-box rattle δ=0.1 Å seed=0; 216 atoms. ASE |F|_max = 2.631e+00 eV/Å.

| Path | Energy (eV) | ms/eval | |ΔE| vs ASE FP64 (eV) | Force MAE (eV/Å) | Force RMSE (eV/Å) | max |ΔF_i| (eV/Å) | max ‖ΔF‖_atom (eV/Å) | Cosine |
|------|-------------|---------|------------------------|------------------|-------------------|---------------------|------------------------|--------|
| ASE FP64 | -1156.6502293830 | 166.7 | — | — | — | — | — | 1.000000 |
| FairChem fix external | -1156.6502319978 | 182.1 | 2.615e-06 | 2.760e-06 | 3.499e-06 | 1.098e-05 | 1.318e-05 | 1.000000 |
| uma/kk precision double | -1156.6502293830 | 157.8 | 0.000e+00 | 8.305e-07 | 1.507e-06 | 4.997e-06 | 6.110e-06 | 1.000000 |
<!-- DISABLED mixed: | uma/kk precision mixed | -1156.6470947266 | 83.9 | 3.135e-03 | 4.884e-06 | 6.141e-06 | 2.247e-05 | 2.435e-05 | 1.000000 | -->

Relative force error (uma/kk double): |ΔF|_max / |F|_ref_max = 1.899e-06 (max |ΔF_i| = 4.997e-06 eV/Å).

## Si 4×4×4 diamond

Si 4x4x4 diamond a=5.484300 Å (=5.43*1.01), uniform-box rattle δ=0.1 Å seed=0; 512 atoms. ASE |F|_max = 2.571e+00 eV/Å.

| Path | Energy (eV) | ms/eval | |ΔE| vs ASE FP64 (eV) | Force MAE (eV/Å) | Force RMSE (eV/Å) | max |ΔF_i| (eV/Å) | max ‖ΔF‖_atom (eV/Å) | Cosine |
|------|-------------|---------|------------------------|------------------|-------------------|---------------------|------------------------|--------|
| ASE FP64 | -2740.8713511548 | 44162.9 | — | — | — | — | — | 1.000000 |
| FairChem fix external | -2740.8713442983 | 52155.8 | 6.857e-06 | 4.450e-06 | 5.860e-06 | 2.388e-05 | 3.760e-05 | 1.000000 |
| uma/kk precision double | -2740.8713511548 | 26320.0 | 0.000e+00 | 8.176e-07 | 1.480e-06 | 4.998e-06 | 7.482e-06 | 1.000000 |
<!-- DISABLED mixed: | uma/kk precision mixed | -2740.8757324219 | 178.5 | 4.381e-03 | 6.108e-06 | 7.789e-06 | 3.381e-05 | 4.453e-05 | 1.000000 | -->

Relative force error (uma/kk double): |ΔF|_max / |F|_ref_max = 1.944e-06 (max |ΔF_i| = 4.998e-06 eV/Å).

**Timing note (Si 4×4×4):** FP64 paths jump to **~26–52 s/eval** on Titan V 12 GB
(HBM thrashing / host RAM spill). Prefer a larger GPU for FP64 at this size.
<!-- DISABLED mixed: Prefer `precision mixed` at this size (~178 ms). -->


## Al 3×3×3 FCC

Al 3x3x3 fcc a=4.090500 Å (=4.05*1.01), uniform-box rattle δ=0.1 Å seed=0; 108 atoms. ASE |F|_max = 8.135e-01 eV/Å.

| Path | Energy (eV) | ms/eval | |ΔE| vs ASE FP64 (eV) | Force MAE (eV/Å) | Force RMSE (eV/Å) | max |ΔF_i| (eV/Å) | max ‖ΔF‖_atom (eV/Å) | Cosine |
|------|-------------|---------|------------------------|------------------|-------------------|---------------------|------------------------|--------|
| ASE FP64 | -401.0795098177 | 122.8 | — | — | — | — | — | 1.000000 |
| FairChem fix external | -401.0795131207 | 127.3 | 3.303e-06 | 7.893e-07 | 9.918e-07 | 2.908e-06 | 3.668e-06 | 1.000000 |
| uma/kk precision double | -401.0795098177 | 111.4 | 5.684e-14 | 1.839e-07 | 2.440e-07 | 4.992e-07 | 6.943e-07 | 1.000000 |
<!-- DISABLED mixed: | uma/kk precision mixed | -401.0796203613 | 62.3 | 1.105e-04 | 4.534e-06 | 5.745e-06 | 1.724e-05 | 2.104e-05 | 1.000000 | -->

Relative force error (uma/kk double): |ΔF|_max / |F|_ref_max = 6.136e-07 (max |ΔF_i| = 4.992e-07 eV/Å).

## Notes

- `max |ΔF_i|` and `max ‖ΔF‖_atom` are both **force-error** metrics vs ASE.
- `uma/kk precision double` vs ASE FP64: NaCl 3×3×3 rocksalt |ΔE|=1.592e-12; Si 3×3×3 diamond exact 0; Si 4×4×4 diamond exact 0; Al 3×3×3 FCC |ΔE|=5.684e-14.
- Mixed |ΔE| vs ASE FP64: NaCl 1.001 meV, Si 3.135 meV, Si 4.381 meV, Al 0.111 meV.
- Cosine vs ASE forces ≈ 1 for all paths.
- Mixed ms/eval: NaCl 3×3×3 rocksalt 70.1, Si 3×3×3 diamond 83.9, Si 4×4×4 diamond 178.5, Al 3×3×3 FCC 62.3.
- **Si 4×4×4 FP64 thrashing:** ASE / FairChem / uma double multi-second vs mixed ~178 ms (no hard OOM). Prefer mixed on 12 GB.
- FairChem `fix external` is not labeled FP64 (`lammps_fc` builds cell in FP32).
- `lammps/src/ML-UMA/uma-engine/src/postprocess.cpp` preserves FP32/FP64 compute dtype after the `denorm_energy` fix.

## Files

- `final_table.json` — machine-readable report
- `*_block.json` — resumable partial reports
- `forces_*_rattle.npz` / `structure_*_rattle.npz`
- `run_all4.log` — latest full 4×4 GPU verification
- `run_final_table_compare.py` — driver (`ONLY_SYSTEM=nacl|si|si4|al|all`)
