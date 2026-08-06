# NaCl, Si, and Al MLIP single-point compares (lattice × 1.01 + rattle)

Rattled geometries on a Titan V. Single-point energy + forces only (no MD / minim).

## Perturbation

Build ideal crystal → scale lattice×positions by `1.01` → add Unif[-0.10,0.10] Å per Cartesian component (PCG64 seed=0) once → wrap into cell → freeze npz → all 4 paths (ASE / FairChem / uma double / uma mixed) load that same npz.

Rule (from `final_table.json`): Build ideal crystal → scale lattice×positions by lattice_scale → add Unif[-delta,delta] Å per Cartesian component (PCG64 seed) once → wrap into cell → freeze npz → all 4 paths load that npz.

Reference: ASE `FAIRChemCalculator` FP64. Timings are repeated force evaluations; uma/kk timings use the LAMMPS Pair section over five NVE steps after warmup. Cosine similarity is meaningful here because rattling yields non-zero ASE reference forces. FairChem `fix external` is **not** labeled FP64 — `lammps_fc` builds the cell in FP32 even when the Predictor is configured for float64.

**Force columns are error metrics vs ASE**, not reference force magnitudes: `max |ΔF_i|` = max component-wise |F − F_ASE|; `max ‖ΔF‖_atom` = max per-atom ‖F − F_ASE‖. Both are the same order of magnitude. Highlight ASE `|F|_max` (= `f_ref_max_abs`) and the relative error `|ΔF|_max / |F|_ref_max` for uma/kk double.

## NaCl 3×3×3 rocksalt

NaCl 3x3x3 rocksalt a=5.696400 Å (=5.64*1.01), uniform-box rattle δ=0.1 Å seed=0; 216 atoms. ASE |F|_max = 3.828e-01 eV/Å.

| Path | Energy (eV) | ms/eval | |ΔE| vs ASE FP64 (eV) | Force MAE (eV/Å) | Force RMSE (eV/Å) | max |ΔF_i| (eV/Å) | max ‖ΔF‖_atom (eV/Å) | Cosine |
|------|-------------|---------|------------------------|------------------|-------------------|---------------------|------------------------|--------|
| ASE FP64 | -729.5978639736 | 144.0 | — | — | — | — | — | 1.000000 |
| FairChem fix external | -729.5978694316 | 151.2 | 5.458e-06 | 5.740e-07 | 7.567e-07 | 3.036e-06 | 3.640e-06 | 1.000000 |
| uma/kk precision double | -729.5978639736 | 129.8 | 1.592e-12 | 1.458e-07 | 2.135e-07 | 4.990e-07 | 7.405e-07 | 1.000000 |
| uma/kk precision mixed | -729.5969238281 | 70.7 | 9.401e-04 | 6.536e-07 | 8.393e-07 | 3.023e-06 | 3.385e-06 | 1.000000 |

Relative force error (uma/kk double): |ΔF|_max / |F|_ref_max = 1.304e-06 (max |ΔF_i| = 4.990e-07 eV/Å).

## Si 3×3×3 diamond

Si 3x3x3 diamond a=5.484300 Å (=5.43*1.01), uniform-box rattle δ=0.1 Å seed=0; 216 atoms. ASE |F|_max = 2.631e+00 eV/Å.

| Path | Energy (eV) | ms/eval | |ΔE| vs ASE FP64 (eV) | Force MAE (eV/Å) | Force RMSE (eV/Å) | max |ΔF_i| (eV/Å) | max ‖ΔF‖_atom (eV/Å) | Cosine |
|------|-------------|---------|------------------------|------------------|-------------------|---------------------|------------------------|--------|
| ASE FP64 | -1156.6502293830 | 179.9 | — | — | — | — | — | 1.000000 |
| FairChem fix external | -1156.6502319978 | 201.9 | 2.615e-06 | 2.760e-06 | 3.499e-06 | 1.098e-05 | 1.318e-05 | 1.000000 |
| uma/kk precision double | -1156.6502293830 | 218.9 | 0.000e+00 | 8.305e-07 | 1.507e-06 | 4.997e-06 | 6.110e-06 | 1.000000 |
| uma/kk precision mixed | -1156.6470947266 | 83.7 | 3.135e-03 | 4.849e-06 | 6.118e-06 | 2.247e-05 | 2.435e-05 | 1.000000 |

Relative force error (uma/kk double): |ΔF|_max / |F|_ref_max = 1.899e-06 (max |ΔF_i| = 4.997e-06 eV/Å).

## Si 4×4×4 diamond

Si 4x4x4 diamond a=5.484300 Å (=5.43*1.01), uniform-box rattle δ=0.1 Å seed=0; 512 atoms. ASE |F|_max = 2.571e+00 eV/Å.

| Path | Energy (eV) | ms/eval | |ΔE| vs ASE FP64 (eV) | Force MAE (eV/Å) | Force RMSE (eV/Å) | max |ΔF_i| (eV/Å) | max ‖ΔF‖_atom (eV/Å) | Cosine |
|------|-------------|---------|------------------------|------------------|-------------------|---------------------|------------------------|--------|
| ASE FP64 | -2740.8713511548 | 13244.7 | — | — | — | — | — | 1.000000 |
| FairChem fix external | -2740.8713442983 | 24221.8 | 6.857e-06 | 4.450e-06 | 5.860e-06 | 2.388e-05 | 3.760e-05 | 1.000000 |
| uma/kk precision double | -2740.8713511548 | 29838.0 | 0.000e+00 | 8.176e-07 | 1.480e-06 | 4.998e-06 | 7.482e-06 | 1.000000 |
| uma/kk precision mixed | -2740.8759765625 | 177.3 | 4.625e-03 | 6.120e-06 | 7.819e-06 | 3.381e-05 | 4.453e-05 | 1.000000 |

Relative force error (uma/kk double): |ΔF|_max / |F|_ref_max = 1.944e-06 (max |ΔF_i| = 4.998e-06 eV/Å).

**Timing note (Si 4×4×4):** FP64 paths jump to **~13–30 s/eval** while mixed stays **~177 ms** (~75× faster than ASE here; Si 3×3×3 FP64 was only ~180–220 ms). Parity is still excellent (uma double |ΔE|=0). This is **GPU memory thrashing / oversubscription**, not a correctness failure and not a clean OOM: the FP64 working set no longer fits comfortably in the Titan V’s **12 GB HBM**, so the CUDA/WSL stack starts **paging or migrating tensors through host (CPU) RAM**. That is GPU↔CPU traffic under memory pressure — not classic OS swap-to-disk thrashing, but yes, **CPU RAM is involved**. Mixed (FP32 model weights/activations, FP64 forces) stays HBM-resident and remains fast. Prefer `precision mixed` at this size on 12 GB.
## Al 3×3×3 FCC

Al 3x3x3 fcc a=4.090500 Å (=4.05*1.01), uniform-box rattle δ=0.1 Å seed=0; 108 atoms. ASE |F|_max = 8.135e-01 eV/Å.

| Path | Energy (eV) | ms/eval | |ΔE| vs ASE FP64 (eV) | Force MAE (eV/Å) | Force RMSE (eV/Å) | max |ΔF_i| (eV/Å) | max ‖ΔF‖_atom (eV/Å) | Cosine |
|------|-------------|---------|------------------------|------------------|-------------------|---------------------|------------------------|--------|
| ASE FP64 | -401.0795098177 | 116.9 | — | — | — | — | — | 1.000000 |
| FairChem fix external | -401.0795131207 | 126.6 | 3.303e-06 | 7.893e-07 | 9.918e-07 | 2.908e-06 | 3.668e-06 | 1.000000 |
| uma/kk precision double | -401.0795098177 | 111.2 | 5.684e-14 | 1.839e-07 | 2.440e-07 | 4.992e-07 | 6.943e-07 | 1.000000 |
| uma/kk precision mixed | -401.0796203613 | 64.0 | 1.105e-04 | 4.522e-06 | 5.725e-06 | 1.724e-05 | 2.100e-05 | 1.000000 |

Relative force error (uma/kk double): |ΔF|_max / |F|_ref_max = 6.136e-07 (max |ΔF_i| = 4.992e-07 eV/Å).

## Notes

- `max |ΔF_i|` and `max ‖ΔF‖_atom` are both **force-error** metrics vs ASE (same order expected); ASE `|F|_max` is the reference force scale.
- `uma/kk precision double` matches ASE FP64 energy to machine precision (NaCl |ΔE|=1.592e-12 eV; Si 3×3×3 exact 0; Si 4×4×4 exact 0; Al |ΔE|=5.684e-14 eV).
- Mixed |ΔE| vs ASE FP64: 0.940 meV (NaCl), 3.135 meV (Si 3×3×3), 4.625 meV (Si 4×4×4), 0.111 meV (Al).
- Cosine vs ASE forces ≈ 1 for all paths.
- Relative |ΔF|_max/|F|_ref_max (uma double): 1.304e-06 (NaCl), 1.899e-06 (Si 3×3×3), 1.944e-06 (Si 4×4×4), 6.136e-07 (Al).
- Mixed ms/eval: 70.7 (NaCl), 83.7 (Si 3×3×3), 177.3 (Si 4×4×4), 64.0 (Al).
- **Si 4×4×4 FP64 thrashing:** ASE / FairChem / uma double ~13–30 s/eval vs mixed ~177 ms (no hard OOM). Working set exceeds comfortable fit in 12 GB HBM; CUDA/WSL oversubscription then moves traffic through **host CPU RAM** (GPU↔CPU paging/migration), collapsing throughput. Prefer mixed at N=4 on Titan V.
- FairChem `fix external` is not labeled FP64 (`lammps_fc` builds cell in FP32).
- `uma-engine/src/postprocess.cpp` preserves FP32/FP64 compute dtype after the `denorm_energy` fix.

## Files

- `final_table.json` — machine-readable report (rattle systems, δ=0.10 Å)
- `nacl_block.json` / `si_block.json` / `si4_block.json` / `al_block.json` — resumable partial reports
- `forces_*_rattle.npz` — per-atom forces
- `structure_*_rattle.npz` — frozen rattled geometries shared by all 4 paths
- `run_si4.log` — Si 4×4×4 GPU run log
- `run_final_table_compare.py` — driver (`ONLY_SYSTEM=nacl|si|si4|al|all`)
