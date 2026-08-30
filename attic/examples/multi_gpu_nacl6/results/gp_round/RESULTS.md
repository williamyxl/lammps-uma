# Multi-GPU NaCl 6×6×6 — graph-parallel uma campaign (`gp_round`)

> **Canonical product timings:** [`../RESULTS.md`](../RESULTS.md) — uma/kk Kokkos+LibTorch + **NCCL** **321.04 / 183.30 / 112.04** ms @ devices=1/2/4 (job `20940474`, campaign PASS). Numbers in this folder (Ray-era ~192 / ~113, P1 cuda_ipc ≈320 / ≈265 / ≈193, pre-IPC ~361 / ~473) are historical; keep for ASE oracle artifacts only.

> **Note (2026-08-07):** Mixed precision (`uma/kk mixed`) is **disabled**. Mixed rows below are commented out (historical only).


**Stamp:** 2026-08-07 (Delta A100-SXM4-40GB, `gpuA100x4`)  
**Suite:** `src/ML-UMA/examples/multi_gpu_nacl6/gp_round/`  
**Status:** **DONE** — ngpu1/2/4 complete · E/F reported **vs ASE FP64@1**

**ASE FP64@1 ground truth (sole E/F oracle):** `oracle_ase_fp64_w1.{json,npz}` —  
E = **−5830.9237201666 eV**, forces `(1728, 3)` float64, `workers=1`, no ParallelMLIP  
(source: promoted from `results/ngpu1/`).

---

## Geometry (immutable)

| Field | Value |
|-------|--------|
| File | `structures/nacl6_rattle_fixed.extxyz` (under `multi_gpu_nacl6/`) |
| Atoms | 1728 |
| Cell | 33.84³ Å, PBC |
| Policy | Frozen; never re-rattle |

---

## Jobs (gp_round)

| Config | Job ID | Outcome |
|--------|--------|---------|
| devices=1 baseline | `20901312` | COMPLETED |
| devices=2 double | `20903160` | COMPLETED — PASS |
<!-- DISABLED mixed: | devices=2 mixed (oracle retarget) | `20903538` | COMPLETED — PASS | -->
<!-- DISABLED mixed: | devices=4 (hung attempt) | `20904146` | TIMEOUT mid mixed NVE | -->
| devices=4 | `20907648` | COMPLETED — PASS (~4.4 min) |

---

## Backend

| `devices` | Mechanism |
|-----------|-----------|
| 1 | Traced LibTorch `Predictor` (`pair_style uma/kk`) |
| N>1 | FairChem eager GP via `GraphParallelRuntime` → `uma_gp_worker.py` (`pair_style uma`, no Kokkos; Ray owns GPUs) |

**Oracle (policy):** all |ΔE| / max|ΔF| vs ASE FairChem FP64 `workers=1`
(`oracle_ase_fp64_w1.*`). Historical harness also checked double vs traced
<!-- DISABLED mixed: `devices=1` and mixed vs ASE float32@1 — superseded for reported E/F. -->
---

## Timing (ms / eval) — uma GP

| Path | 1 GPU | 2 GPU | 4 GPU | Speedup 1→4 |
|------|------:|------:|------:|------------:|
| double | 322.2 | 192.4 | 112.6 | **2.86×** |
<!-- DISABLED mixed: | mixed | 246.4 | 148.7 | 91.2 | **2.70×** | -->

### Strong scaling

Fixed problem size (1728 atoms). \(S(N)=t_1/t_N\), \(\eta=S/N\).

<!-- DISABLED mixed: | N | Ideal | double S (η) | mixed S (η) | -->
|--:|------:|-------------:|------------:|
| 1 | 1× | 1.00× (100%) | 1.00× (100%) |
| 2 | 2× | 1.67× (84%) | 1.66× (83%) |
| 4 | 4× | 2.86× (71%) | 2.70× (68%) |

Canvas plot: `canvases/uma-multigpu-nacl6-results.canvas.tsx`.

---

## Parity (vs ASE FP64@1)

| Path | ngpu | \|ΔE\| | max \|ΔF\| | gate |
|------|------|--------|------------|------|
| double | 1 | 1.3×10⁻¹⁰ | 5.0×10⁻⁷ | PASS |
| double | 2 | ~10⁻¹² | 5.0×10⁻⁷ | PASS |
| double | 4 | ~10⁻¹² | 5.0×10⁻⁷ | PASS |
<!-- DISABLED mixed: | mixed | 1 | **5.82×10⁻²** | 7.2×10⁻⁶ | FAIL | -->
<!-- DISABLED mixed: | mixed | 2 | 3.06×10⁻⁴ | 7.0×10⁻⁶ | PASS | -->
<!-- DISABLED mixed: | mixed | 4 | 1.50×10⁻⁴ | 7.0×10⁻⁶ | PASS | -->

---

## Findings

<!-- DISABLED mixed: 1. **Graph-parallel uma scales** on same-node 2/4 A100s: double **2.86×** / mixed **2.70×** (1→4) for this 1728-atom cell — **final answer, not the legacy 1.00×**. -->
2. **Double** matches ASE FP64@1 at ~1e-10–1e-12 energy / ~5e-7 force max.
<!-- DISABLED mixed: 3. **Mixed GP** @2/4 agrees with ASE FP64 within ~0.3 meV; **traced mixed @1 FAIL**s (~58 meV). -->
4. **devices=4 mixed NVE** can hang (job `20904146`); successful rerun `20907648` completed cleanly.
5. Kokkos `-k on g N` alone (pre-GP) stayed ~1.00× — historical contrast only.

---

## File index

| Path | Role |
|------|------|
| `results/RESULTS.md` | Legacy multi-path (ASE/FC) results |
| `results/gp_round/` | **This campaign** artifacts + SUMMARY |
| `results/gp_round/ngpu{1,2,4}/parity.json` | Per-N gates |
| `uma-engine/docs/multi_gpu_graph_parallel.md` | Spec + thresholds |
