# Multi-GPU NaCl 6×6×6 — results (canonical)

> **Note (2026-08-07):** Mixed precision (`uma/kk mixed`) is **disabled**. Mixed rows below are commented out (historical only).

**Stamp:** 2026-08-07 ~17:50 CDT (Delta A100-SXM4-40GB, `gpuA100x4`)  
**Suite:** `src/ML-UMA/examples/multi_gpu_nacl6/`  
**Status:** uma GP 1/2/4 **DONE** · ASE FP64 1/2/4 **DONE** (path-isolated today) · FC 1/2 **DONE** · FC @4 **PENDING** (`20910353`) · uma @4 requeue `20910354` (dependency)

Canvas: [`uma-multigpu-nacl6-results`](/u/xyan11/.cursor/projects/work-nvme-bfzx-xyan11-workdir-lammps-uma/canvases/uma-multigpu-nacl6-results.canvas.tsx) · Detail: [`gp_round/RESULTS.md`](gp_round/RESULTS.md)

---

## Ground truth — ASE FairChem FP64 (`workers=1`)

| Field | Value |
|-------|--------|
| Artifact | [`gp_round/oracle_ase_fp64_w1.json`](gp_round/oracle_ase_fp64_w1.json) + [`.npz`](gp_round/oracle_ase_fp64_w1.npz) |
| Energy | **−5830.9237201666 eV** |
| Forces | `(1728, 3)` float64 · RMS 0.159 · max\|F\| 0.493 eV/Å |
| Timing @1 GPU | **396.5 ms/eval** (job `20910344`; oracle E/F unchanged) |
| API | ASE FairChem FP64, `workers=1`, **no** ParallelMLIPPredictUnit |

Reuse E/F oracle for all gates; timing table below uses today’s path-isolated ASE/FC jobs.

---

## Geometry (immutable)

| Field | Value |
|-------|--------|
| File | `structures/nacl6_rattle_fixed.extxyz` |
| Atoms | 1728 · cell 33.84³ Å · Unif[−0.1,0.1] Å seed=0 · never re-rattle |

---

## Timing (ms/eval) — path-isolated ASE/FC + uma GP

| Path | 1 GPU | 2 GPU | 4 GPU | 1→2 | 1→4 |
|------|------:|------:|------:|----:|----:|
| ASE FairChem FP64 | **396.5** | **193.9** | **115.2** | **2.04×** | **3.44×** |
| FairChem FC LAMMPS | **345.5** | **193.2** | *PENDING* | **1.79×** | — |
| uma double (GP) | 322.2 | 192.4 | 112.6 | 1.67× | **2.86×** |

Sources:

| Path | Jobs |
|------|------|
| ASE | `20910344` / `20910348` / `20910352` |
| FC | `20910345` / `20910349` / `20910353` (pending) |
| uma double GP | `gp_round` (`20901312` / `20903160` / `20907648`) |

Note: do **not** use `parity.json` `ms_per_eval` when it equals SLURM wall/`N_TIMING` (contaminated). Prefer the `ASE E=… XXX ms` / `fc_result_early.json` lines.

### vs ASE FP64@1 ground truth

| Path | ngpu | Energy (eV) | \|ΔE\| | max \|ΔF\| | cosine |
|------|------|-------------|---------|------------|--------|
| ASE | 1/2/4 | −5830.9237201666 | ~0–10⁻¹² | 0 (self) | 1.0 |
| double | 1 | −5830.9237201667 | 1.3×10⁻¹⁰ | 5.0×10⁻⁷ | 1.0 |
| double | 2 | −5830.9237201666 | ~10⁻¹² | 5.0×10⁻⁷ | 1.0 |
| double | 4 | −5830.9237201666 | ~10⁻¹² | 5.0×10⁻⁷ | 1.0 |

### Jobs

| Config | Job | Outcome |
|--------|-----|---------|
| ASE @1/@2/@4 | `20910344` / `20910348` / `20910352` | COMPLETED |
| FC @1/@2 | `20910345` / `20910349` | COMPLETED |
| FC @4 | `20910353` | PENDING |
| uma double @4 (requeue) | `20910354` | PENDING (afterok FC) |
| uma GP 1/2/4 | `20901312` / `20903160` / `20907648` | PASS |

---

## Findings

1. Today’s ASE path-isolated timings still show **~2.0× @2** and **~3.4× @4** (FairChem `workers=N`).
2. FC @2 ≈ **1.8×**; FC @4 still queued.
3. uma GraphParallelRuntime double: **~2.9× @4**.
4. OOM / max-N: **N\*=10** ([`../multi_node_nacl6/results/geom_sweep/SWEEP.md`](../multi_node_nacl6/results/geom_sweep/SWEEP.md)).

## File index

| Path | Role |
|------|------|
| `RESULTS.md` | This document |
| `gp_round/oracle_ase_fp64_w1.*` | ASE FP64@1 E/F ground truth |
| `gp_round/ngpu{1,2,4}/` | uma GP parity + forces |
| `ngpu{1,2,4}/` | Path-isolated ASE/FC merges |
| `NEXT_ROUND_PLAN.md` | Close-out checklist |
| `SUMMARY.md` / `MULTIGPU_REPORT.md` | Compact tables |
