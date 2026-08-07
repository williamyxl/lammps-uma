# Multi-GPU NaCl 6×6×6 — graph-parallel uma campaign (`gp_round`)

**Stamp:** 2026-08-07 (Delta A100-SXM4-40GB, `gpuA100x4`)  
**Suite:** `src/ML-UMA/examples/multi_gpu_nacl6/gp_round/`  
**Status:** **DONE** — ngpu1 / ngpu2 / ngpu4 double+mixed gates all **PASS**

---

## Geometry (immutable)

| Field | Value |
|-------|--------|
| File | `../delta_parity/structures/nacl6_rattle_fixed.extxyz` |
| Atoms | 1728 |
| Cell | 33.84³ Å, PBC |
| Policy | Frozen; never re-rattle |

---

## Jobs (gp_round)

| Config | Job ID | Outcome |
|--------|--------|---------|
| devices=1 baseline | `20901312` | COMPLETED |
| devices=2 double | `20903160` | COMPLETED — PASS |
| devices=2 mixed (oracle retarget) | `20903538` | COMPLETED — PASS |
| devices=4 (hung attempt) | `20904146` | TIMEOUT mid mixed NVE |
| devices=4 | `20907648` | COMPLETED — PASS (~4.4 min) |

---

## Backend

| `devices` | Mechanism |
|-----------|-----------|
| 1 | Traced LibTorch `Predictor` (`pair_style uma/kk`) |
| N>1 | FairChem eager GP via `GraphParallelRuntime` → `uma_gp_worker.py` (`pair_style uma`, no Kokkos; Ray owns GPUs) |

**Oracles**

| Mode | Gate reference |
|------|----------------|
| double | uma traced `devices=1` |
| mixed | ASE FairChem `float32` `workers=1` (traced mixed disagrees ~0.058 eV on NaCl6) |

Mixed `|ΔE|` threshold: **5×10⁻⁴** eV (FairChem float32 w1↔w2 ~1.4×10⁻⁴).

---

## Timing (ms / eval) — uma GP

| Path | 1 GPU | 2 GPU | 4 GPU | Speedup 1→4 |
|------|------:|------:|------:|------------:|
| double | 322.2 | 192.4 | 112.6 | **2.86×** |
| mixed | 246.4 | 148.7 | 91.2 | **2.70×** |

---

## Parity gates (vs oracle)

| Path | ngpu | \|ΔE\| | max \|ΔF\| | gate |
|------|------|--------|------------|------|
| double | 2 | 1.3×10⁻¹⁰ | 0 | PASS |
| double | 4 | 1.3×10⁻¹⁰ | 0 | PASS |
| mixed | 2 | 1.9×10⁻⁴ | 7.0×10⁻⁷ | PASS |
| mixed | 4 | 7.7×10⁻⁵ | 7.3×10⁻⁷ | PASS |

---

## Findings

1. **Graph-parallel uma scales** on same-node 2/4 A100s for this 1728-atom cell (unlike prior Kokkos-only `g N` SP).
2. **Double** matches traced `devices=1` at ~1e-10 energy / zero force max.
3. **Mixed GP** matches ASE FairChem float32@1 within 5e-4; do not gate vs traced mixed artifact energy.
4. **devices=4 mixed NVE** can hang (job `20904146`); successful rerun `20907648` completed cleanly.

---

## File index

| Path | Role |
|------|------|
| `results/RESULTS.md` | Legacy multi-path (ASE/FC) results |
| `results/gp_round/` | **This campaign** artifacts + SUMMARY |
| `results/gp_round/ngpu{1,2,4}/parity.json` | Per-N gates |
| `uma-engine/docs/multi_gpu_graph_parallel.md` | Spec + thresholds |
