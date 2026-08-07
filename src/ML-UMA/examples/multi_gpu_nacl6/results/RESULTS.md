# Multi-GPU NaCl 6×6×6 — results (canonical)

**Stamp:** 2026-08-07 (Delta A100-SXM4-40GB, `gpuA100x4`)  
**Suite:** `src/ML-UMA/examples/multi_gpu_nacl6/`  
**Status:** uma GP 1/2/4 **DONE** · ASE FP64@1 ground truth **CACHED** · ASE/FC @4 **PENDING** (`20909845`/`20909846`)

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
| File | `../delta_parity/structures/nacl6_rattle_fixed.extxyz` |
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
| ASE FairChem FP64 | 396.1 | 191.9 | *PENDING* | 2.06× | — |
| FairChem FC LAMMPS | 345.0 | 194.8 | *PENDING* | 1.77× | — |
| uma double (GP) | 322.2 | 192.4 | 112.6 | 1.67× | **2.86×** |
| uma mixed (GP) | 246.4 | 148.7 | 91.2 | 1.66× | **2.70×** |

ASE/FC @4: `20909845` → `20909846`. uma η@4: double 71%, mixed 68%.

### vs ASE FP64@1 ground truth

| Path | ngpu | Energy (eV) | \|ΔE\| | max \|ΔF\| | cosine |
|------|------|-------------|---------|------------|--------|
| double | 1 | −5830.9237201667 | 1.3×10⁻¹⁰ | 5.0×10⁻⁷ | 1.0 |
| double | 2 | −5830.9237201666 | ~10⁻¹² | 5.0×10⁻⁷ | 1.0 |
| double | 4 | −5830.9237201666 | ~10⁻¹² | 5.0×10⁻⁷ | 1.0 |
| mixed | 1 | −5830.9819335938 | **5.82×10⁻²** | 7.2×10⁻⁶ | 1.0 |
| mixed | 2 | −5830.9234143138 | 3.06×10⁻⁴ | 7.2×10⁻⁶ | 1.0 |
| mixed | 4 | −5830.9235703731 | 1.50×10⁻⁴ | 7.2×10⁻⁶ | 1.0 |

Campaign gates for display / reporting: **always vs ASE FairChem FP64@1**
(`oracle_ase_fp64_w1.*`). Historical campaign also checked double vs traced d1
and mixed vs ASE float32@1; those are superseded for E/F comparison. Traced
mixed @1 **FAIL**s vs ASE FP64 (~58 meV); GP mixed @2/4 agrees within ~0.3 meV.

### Jobs

| Config | Job | Outcome |
|--------|-----|---------|
| devices=1 | `20901312` | COMPLETED |
| devices=2 double / mixed | `20903160` / `20903538` | PASS |
| devices=4 hung / PASS | `20904146` / `20907648` | TIMEOUT then **PASS** |

---

## Findings

1. Graph-parallel uma scales **~2.7–2.9×** on 4 A100s (NaCl6).
2. ASE FP64@1 is the permanent E/F ground truth (cached).
3. GP mixed @2/4 agrees with ASE FP64 within ~0.3 meV; traced mixed @1 does not (~58 meV).
4. ASE/FC @4 still queued for the full 1/2/4 prior-art timing table.
5. Legacy Kokkos-only uma @2 stayed ~**1.00×** (superseded by GP).

---

## File index

| Path | Role |
|------|------|
| `RESULTS.md` | This document |
| `gp_round/oracle_ase_fp64_w1.*` | ASE FP64@1 ground truth |
| `gp_round/ngpu{1,2,4}/` | uma GP parity + forces |
| `ngpu{1,2}/` | ASE/FC + legacy Kokkos-only |
| `ngpu4/` | ASE/FC @4 (pending) |
| `NEXT_ROUND_PLAN.md` | Pending close-out checklist |
| `SUMMARY.md` / `MULTIGPU_REPORT.md` | Merged tables |
