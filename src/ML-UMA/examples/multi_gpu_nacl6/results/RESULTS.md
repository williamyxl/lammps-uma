# Multi-GPU NaCl 6×6×6 — results (canonical)

> **Note (2026-08-07):** Mixed precision (`uma/kk mixed`) is **disabled**. Mixed rows below are commented out (historical only).

**Stamp:** 2026-08-07 ~17:42 CDT (Delta A100-SXM4-40GB, `gpuA100x4`)  
**Suite:** `src/ML-UMA/examples/multi_gpu_nacl6/`  
**Status:** uma GP 1/2/4 **DONE** · ASE FP64@1 ground truth **CACHED** · ASE @4 **DONE** (`20909845`, 117.7 ms) · FC @4 **PENDING** (`20910353`; `20909846` cancelled) · requeue ASE/double `20910352`/`20910354` still pending (optional)

Canvas: [`uma-multigpu-nacl6-results`](/u/xyan11/.cursor/projects/work-nvme-bfzx-xyan11-workdir-lammps-uma/canvases/uma-multigpu-nacl6-results.canvas.tsx) · Detail: [`gp_round/RESULTS.md`](gp_round/RESULTS.md)

---

## Ground truth — ASE FairChem FP64 (`workers=1`)

| Field | Value |
|-------|--------|
| Artifact | [`gp_round/oracle_ase_fp64_w1.json`](gp_round/oracle_ase_fp64_w1.json) + [`.npz`](gp_round/oracle_ase_fp64_w1.npz) |
| Energy | **−5830.9237201666 eV** |
| Forces | `(1728, 3)` float64 · RMS 0.159 · max\|F\| 0.493 eV/Å |
| Timing @1 GPU | 396.1 ms/eval |
| API | ASE FairChem FP64, `workers=1`, **no** ParallelMLIPPredictUnit |
| Source | Promoted from `results/ngpu1/` (job `20898588`) |

Reuse for all later gates; recompute only if geometry or checkpoint changes.

---

## Geometry (immutable)

| Field | Value |
|-------|--------|
| File | `structures/nacl6_rattle_fixed.extxyz` |
| Atoms | 1728 · cell 33.84³ Å · Unif[−0.1,0.1] Å seed=0 · never re-rattle |

---

## uma graph-parallel (`gp_round`) — final

| `devices` | Backend |
|-----------|---------|
| 1 | Traced LibTorch (`pair_style uma/kk`) |
| N>1 | FairChem eager GP via `GraphParallelRuntime` (Ray; same-node only) |

### Timing (ms/eval) — all paths

| Path | 1 GPU | 2 GPU | 4 GPU | 1→2 | 1→4 |
|------|------:|------:|------:|----:|----:|
| ASE FairChem FP64 | 396.1 | 191.9 | **117.7** | 2.06× | **3.37×** |
| FairChem FC LAMMPS | 345.0 | 194.8 | *PENDING* | 1.77× | — |
| uma double (GP) | 322.2 | 192.4 | 112.6 | 1.67× | **2.86×** |
<!-- DISABLED mixed: | uma mixed (GP) | 246.4 | 148.7 | 91.2 | 1.66× | **2.70×** | -->

ASE @4: job `20909845` → `results/ngpu4/parity.json` (117.7 ms/eval). FC @4: `20909846` cancelled; requeue `20910353` pending. uma double η@4: 71%.

### vs ASE FP64@1 ground truth

| Path | ngpu | Energy (eV) | \|ΔE\| | max \|ΔF\| | cosine |
|------|------|-------------|---------|------------|--------|
| ASE | 4 | −5830.9237201666 | ~10⁻¹² | 0 (self) | 1.0 |
| double | 1 | −5830.9237201667 | 1.3×10⁻¹⁰ | 5.0×10⁻⁷ | 1.0 |
| double | 2 | −5830.9237201666 | ~10⁻¹² | 5.0×10⁻⁷ | 1.0 |
| double | 4 | −5830.9237201666 | ~10⁻¹² | 5.0×10⁻⁷ | 1.0 |
<!-- DISABLED mixed: | mixed | 1 | −5830.9819335938 | **5.82×10⁻²** | 7.2×10⁻⁶ | 1.0 | -->
<!-- DISABLED mixed: | mixed | 2 | −5830.9234143138 | 3.06×10⁻⁴ | 7.2×10⁻⁶ | 1.0 | -->
<!-- DISABLED mixed: | mixed | 4 | −5830.9235703731 | 1.50×10⁻⁴ | 7.2×10⁻⁶ | 1.0 | --> |

Campaign gates for display / reporting: **always vs ASE FairChem FP64@1**
(`oracle_ase_fp64_w1.*`). Historical campaign also checked double vs traced d1.

### Jobs

| Config | Job | Outcome |
|--------|-----|---------|
| devices=1 | `20901312` | COMPLETED |
| devices=2 double | `20903160` | PASS |
| devices=4 double | `20907648` | PASS |
| ASE @4 | `20909845` | COMPLETED (117.7 ms) |
| FC @4 | `20909846` | CANCELLED |
| ASE/FC/double requeue @4 | `20910352`–`20910354` | PENDING |

---

## Findings

1. Graph-parallel uma double scales **~2.9×** on 4 A100s (NaCl6).
2. ASE FP64@1 is the permanent E/F ground truth (cached); ASE @4 timing **3.37×** vs @1.
3. FC @4 still needed to finish the full prior-art 1/2/4 table.
4. Legacy Kokkos-only uma @2 stayed ~**1.00×** (superseded by GP).

---

## Related

- OOM / max-N sweep: [`../multi_node_nacl6/results/geom_sweep/SWEEP.md`](../multi_node_nacl6/results/geom_sweep/SWEEP.md) (**N\*=10**).

## File index

| Path | Role |
|------|------|
| `RESULTS.md` | This document |
| `gp_round/oracle_ase_fp64_w1.*` | ASE FP64@1 ground truth |
| `gp_round/ngpu{1,2,4}/` | uma GP parity + forces |
| `ngpu{1,2}/` | ASE/FC + legacy |
| `ngpu4/` | ASE @4 done; FC pending |
| `NEXT_ROUND_PLAN.md` | Close-out checklist |
| `SUMMARY.md` / `MULTIGPU_REPORT.md` | Merged tables |
