# Multi-GPU NaCl 6×6×6 — results (latest)

**Stamp:** 2026-08-06 (legacy ASE/FC); **gp_round** 2026-08-07  
**Suite:** `src/ML-UMA/examples/multi_gpu_nacl6/`  
**Status:** Legacy ASE/FC below. **Graph-parallel uma (`results/gp_round/`) DONE** — devices 1/2/4 double+mixed gates PASS (see `results/gp_round/RESULTS.md`).

---

## Geometry (immutable)

| Field | Value |
|-------|--------|
| File | `../delta_parity/structures/nacl6_rattle_fixed.extxyz` |
| System | NaCl rocksalt 6×6×6, a = 5.64 Å |
| Atoms | 1728 |
| Cell | 33.84 × 33.84 × 33.84 Å (cubic, PBC) |
| Perturbation | Unif[−0.10, +0.10] Å per Cartesian, seed = 0, wrap |
| Coordinate format | 12 significant digits (`.12g`) |
| Policy | **Never re-rattle** — all paths share this file |

---

## Jobs

| Config | Job ID | Outcome |
|--------|--------|---------|
| 1×A100 full (4 paths) | `20898588` | COMPLETED |
| 2×A100 full (4 paths) | `20898818` | COMPLETED; FC row invalid (see below) |
| 2×A100 FC-only (merge) | `20900529` | COMPLETED — valid FC@2GPU merged |
| 4×A100 | — | Deferred |

Artifacts: `results/ngpu1/parity.json`, `results/ngpu2/parity.json`, `results/SUMMARY.json`.

---

## Recipes

| Stack | Multi-GPU mechanism |
|-------|---------------------|
| ASE FairChem FP64 | `load_predict_unit(..., workers=NGPUS)` → Ray `ParallelMLIPPredictUnit` |
| FairChem fix-external | Same `workers=NGPUS` predictor; LAMMPS single Python process |
| uma/kk double / mixed | Single MPI rank: `lmp -k on g ${NGPUS} -sf kk` (no mpirun) |

Notes:

- ASE with `workers>1` uses **internal** graph gen (`external_graph=False`); external graph + graph-parallel hit CUDA index asserts.
- uma/kk LibTorch UMA forward remains **single-device**; Kokkos `g N` does not speed this SP.
- FC after ASE in one process can leave torch.distributed state; validated FC@2 used `ONLY_PATHS=fc` + `HARD_EXIT_AFTER_FC=1`.

Precision: ASE + uma/kk double = **FP64**. Mixed is explicit FP32-pos/energy path. FC may build cell in FP32 inside `lammps_fc`.

---

## Timing (ms / eval)

| Path | 1×A100 | 2×A100 | Speedup |
|------|-------:|-------:|--------:|
| ASE FairChem FP64 | 396.1 | 191.9 | **2.06×** |
| FairChem LAMMPS fix-external | 345.0 | 194.8 | **1.77×** |
| uma/kk precision double | 320.4 | 321.0 | 1.00× |
| uma/kk precision mixed | 245.4 | 246.3 | 1.00× |

---

## Total energy (eV) — same geometry

Reference = ASE@1GPU: **−5830.9237201666 eV**

| Path | 1×A100 E | 2×A100 E | \|ΔE\| vs ASE@1 |
|------|---------:|---------:|----------------:|
| ASE FP64 | −5830.9237201666 | −5830.9237201666 | 0 / ~1.8×10⁻¹² |
| FairChem FC | −5830.9237152511 | −5830.9237152511 | 4.915×10⁻⁶ |
| uma/kk double | −5830.9237201667 | −5830.9237201667 | ~1.2×10⁻¹⁰ |
| uma/kk mixed | −5830.9814453125 | −5830.9814453125 | 5.773×10⁻² (~57.7 meV) |

Force agreement vs ASE@1 (from SUMMARY): uma/kk double force cosine ≈ 1.0, max |ΔF| ~5×10⁻⁷ eV/Å; FC max |ΔF| ~7×10⁻⁶ eV/Å; mixed similar force scale with larger energy offset.

---

## Findings

1. **ASE Ray scales** ~2× on 2 GPUs for this 1728-atom cell (FP64, internal graph).
2. **FC Ray scales** ~1.8× once run in isolation (same energy as 1-GPU FC).
3. **uma/kk does not scale** with `-k on g 2` for single-point energy/forces — expected until LibTorch UMA is multi-device.
4. **Mixed** is faster than double but ~58 meV off ASE FP64 (known precision tradeoff).
5. **4-GPU** not measured yet.

---

## Known issues / workarounds

| Issue | Workaround |
|-------|------------|
| ASE `workers>1` + `external_graph=True` → CUDA index assert | `external_graph=(workers<=1)` |
| ASE then FC in one process → PG double-init / E=0 | Separate FC job or hard-exit after FC |
| `dist.destroy_process_group` hang after FC | `SKIP_DIST_DESTROY=1`, `HARD_EXIT_AFTER_FC=1` |
| `ONLY_PATHS` subset wiped `ngpu2/` | Merge mode (subset no longer `rmtree`s) |

---

## File index

| Path | Role |
|------|------|
| `results/RESULTS.md` | **This document** (canonical latest) |
| `results/ngpu{1,2}/parity.json` | Per-GPU machine-readable rows |
| `results/ngpu{1,2}/forces.npz` | Per-atom forces |
| `results/SUMMARY.json` / `.md` | Merged collector tables |
| `results/MULTIGPU_REPORT.md` | Auto report from collector |
| `results/COORD_ANALYSIS.md` | Coordinator summary |
| `MULTI_GPU_API_NOTES.md` | API notes |
| `CODE_REVIEW.md` | Code review |
