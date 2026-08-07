# Multi-GPU NaCl 6×6×6 — results (canonical)

> **Note (2026-08-07):** Mixed precision (`uma/kk mixed`) is **disabled**. Tables below are ASE / FC / uma double (FP64) only.

**Stamp:** 2026-08-07 ~18:00 CDT (Delta A100-SXM4-40GB, `gpuA100x4`)  
**Suite:** `src/ML-UMA/examples/multi_gpu_nacl6/`  
**Status:** path-isolated ASE / FC / uma_double @ 1/2/4 **ALL COMPLETED**

Canvas: [`uma-multigpu-nacl6-results`](/u/xyan11/.cursor/projects/work-nvme-bfzx-xyan11-workdir-lammps-uma/canvases/uma-multigpu-nacl6-results.canvas.tsx)

---

## Ground truth — ASE FairChem FP64 (`workers=1`)

| Field | Value |
|-------|--------|
| Artifact | [`gp_round/oracle_ase_fp64_w1.json`](gp_round/oracle_ase_fp64_w1.json) + [`.npz`](gp_round/oracle_ase_fp64_w1.npz) |
| Energy | **−5830.9237201666 eV** |
| Forces | `(1728, 3)` float64 · RMS 0.159 · max\|F\| 0.493 eV/Å |
| Timing @1 GPU | **396.5 ms/eval** (job `20910344`) |
| API | ASE FairChem FP64, `workers=1`, **no** ParallelMLIPPredictUnit |

---

## Geometry (immutable)

| Field | Value |
|-------|--------|
| File | `structures/nacl6_rattle_fixed.extxyz` |
| Atoms | 1728 · cell 33.84³ Å · Unif[−0.1,0.1] Å seed=0 · never re-rattle |

---

## Timing (ms/eval) — path-isolated batch

Prefer log / `fc_result_early.json` lines. Do **not** use `parity.json` `ms_per_eval` when it equals SLURM wall/`N_TIMING`.

| Path | 1 GPU | 2 GPU | 4 GPU | 1→2 | 1→4 | η@4 |
|------|------:|------:|------:|----:|----:|----:|
| ASE FairChem FP64 | **396.5** | **193.9** | **115.2** | **2.04×** | **3.44×** | 86% |
| FairChem FC LAMMPS | **345.5** | **193.2** | **118.0** | **1.79×** | **2.93×** | 73% |
| uma/kk double (GP) | **320.4** | **192.0** | **112.6** | **1.67×** | **2.85×** | 71% |

| Path | Jobs |
|------|------|
| ASE | `20910344` / `20910348` / `20910352` |
| FC | `20910345` / `20910349` / `20910353` |
| uma_double | `20910346` / `20910350` / `20910354` |

Sources: `ASE E=… XXX ms` in job outs · `ngpu*/work/fc/fc_result_early.json` · `uma64 E=… XXX ms` in job outs.

---

## Energy + force accuracy vs ASE FP64@1

Forces from `ngpu{1,2,4}/forces.npz` vs `forces_ase` at the same ngpu (ASE self = 0). Energies vs oracle −5830.9237201666 eV.

| Path | ngpu | Energy (eV) | \|ΔE\| (eV) | Force MAE | Force RMSE | max \|ΔFᵢ\| | max ‖ΔF‖_atom | Cosine |
|------|-----:|-------------:|------------:|----------:|-----------:|------------:|--------------:|-------:|
| ASE FP64 | 1 | −5830.9237201666 | ~0 | 0 | 0 | 0 | 0 | 1.0 |
| ASE FP64 | 2 | −5830.9237201666 | ~0 | 0 | 0 | 0 | 0 | 1.0 |
| ASE FP64 | 4 | −5830.9237201666 | ~0 | 0 | 0 | 0 | 0 | 1.0 |
| FairChem FC | 1 | −5830.9237152511 | 4.915×10⁻⁶ | 1.002×10⁻⁶ | 1.343×10⁻⁶ | 7.123×10⁻⁶ | 7.127×10⁻⁶ | 1.0 |
| FairChem FC | 2 | −5830.9237152511 | 4.915×10⁻⁶ | 1.002×10⁻⁶ | 1.343×10⁻⁶ | 7.123×10⁻⁶ | 7.127×10⁻⁶ | 1.0 |
| FairChem FC | 4 | −5830.9237152511 | 4.915×10⁻⁶ | 1.002×10⁻⁶ | 1.343×10⁻⁶ | 7.123×10⁻⁶ | 7.127×10⁻⁶ | 1.0 |
| uma/kk double | 1 | −5830.9237201667 | 1.23×10⁻¹⁰ | 1.516×10⁻⁷ | 2.179×10⁻⁷ | 5.000×10⁻⁷ | 7.673×10⁻⁷ | 1.0 |
| uma/kk double | 2 | −5830.9237201666 | 1.82×10⁻¹² | 1.516×10⁻⁷ | 2.179×10⁻⁷ | 5.000×10⁻⁷ | 7.673×10⁻⁷ | 1.0 |
| uma/kk double | 4 | −5830.9237201666 | 9.09×10⁻¹³ | 1.516×10⁻⁷ | 2.179×10⁻⁷ | 5.000×10⁻⁷ | 7.673×10⁻⁷ | 1.0 |

FC \|ΔE\| ≈ 4.9×10⁻⁶ is expected (LAMMPS FC cell built in FP32). uma double stays within ~10⁻¹⁰ of ASE.

### uma/kk double vs devices=1 (graph-parallel gate)

Thresholds: \|ΔE\| ≤ 1×10⁻⁸ · max\|ΔF\| ≤ 1×10⁻⁶ · cosine ≥ 1−ε → **3/3 PASS**

| ngpu | devices | \|ΔE\| vs d1 | max \|ΔF\| vs d1 | cosine vs d1 | gate |
|-----:|--------:|-------------:|-----------------:|-------------:|:----:|
| 1 | 1 | — (self) | 0 | 1.0 | PASS |
| 2 | 2 | 1.27×10⁻¹⁰ | 0 | 1.0 | PASS |
| 4 | 4 | 1.27×10⁻¹⁰ | 0 | 1.0 | PASS |

---

## Jobs

| Config | Job | Outcome | Honest ms |
|--------|-----|---------|----------:|
| ASE @1/@2/@4 | `20910344` / `48` / `52` | COMPLETED | 396.5 / 193.9 / 115.2 |
| FC @1/@2/@4 | `20910345` / `49` / `53` | COMPLETED | 345.5 / 193.2 / 118.0 |
| uma_double @1/@2/@4 | `20910346` / `50` / `54` | COMPLETED | 320.4 / 192.0 / 112.6 |

---

## Findings

1. ASE `workers=N`: **2.04× @2**, **3.44× @4** (η@4 ≈ 86%).
2. FC: **1.79× @2**, **2.93× @4** (η@4 ≈ 73%); E offset ~5×10⁻⁶ vs ASE from FP32 cell.
3. uma/kk double GP: **1.67× @2**, **2.85× @4** (η@4 ≈ 71%); forces vs ASE max\|ΔF\| = 5×10⁻⁷; GP gate vs d1 **PASS** with bitwise-identical forces across devices.
4. OOM / max-N: **N\*=10** ([`../multi_node_nacl6/results/geom_sweep/SWEEP.md`](../multi_node_nacl6/results/geom_sweep/SWEEP.md)).

## File index

| Path | Role |
|------|------|
| `RESULTS.md` | This document (canonical) |
| `ngpu{1,2,4}/parity.json` | Merged energies (wall ms often contaminated) |
| `ngpu{1,2,4}/forces.npz` | ASE / FC / uma_double forces |
| `ngpu{1,2,4}/work/fc/fc_result_early.json` | Honest FC ms |
| `SUMMARY.md` / `MULTIGPU_REPORT.md` | Compact copies of this report |
| `NEXT_ROUND_PLAN.md` | Close-out checklist |
