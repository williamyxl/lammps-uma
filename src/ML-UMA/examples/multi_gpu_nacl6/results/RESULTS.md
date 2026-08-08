# Multi-GPU NaCl 6×6×6 — results (canonical)

**Phase 4 report** · Stamp: 2026-08-08 ~01:50 CDT · Branch `uma-kokkos-mlip` @ `5513482e9b`  
**Suite:** `src/ML-UMA/examples/multi_gpu_nacl6/`  
**Status:** same-node graph-parallel **scientifically GREEN** through Phase 3 (engine CLI + LAMMPS)

## Product backend (uma/kk)

| Field | Value |
|-------|--------|
| Backend | **Kokkos + LibTorch** (`gp=kokkos_libtorch_vesin`) |
| Runtime | C++ `LibtorchMpRuntime` — process-per-rank workers + `/dev/shm` `uma_peer` collectives + vesin NL |
| Launch | `lmp -k on g N -sf kk` · `pair_style uma/kk precision double devices N` · **1 MPI rank** |
| Precision | **FP64 only** |
| Artifacts | `model_mp_w{N}_n{NATOMS}_r{R}.pt` (+ legacy `model_mp_w{N}_r*` for n=64) |
| **Not** product | Ray · FairChem `ParallelMLIPPredictUnit` · Python GP worker (env opt-in only) |

ASE FairChem / FC rows below are **reference baselines only** (historical path-isolated batch). They are not the uma/kk product path.

---

## Geometry (immutable)

| Field | Value |
|-------|--------|
| File | `structures/nacl6_rattle_fixed.extxyz` |
| Atoms | 1728 · cell 33.84³ Å · Unif[−0.1,0.1] Å seed=0 · never re-rattle |

---

## Ground truth — ASE FairChem FP64 (`workers=1`)

| Field | Value |
|-------|--------|
| Artifact | [`gp_round/oracle_ase_fp64_w1.json`](gp_round/oracle_ase_fp64_w1.json) |
| Energy | **−5830.9237201666 eV** |
| Timing @1 GPU | **396.5 ms/eval** (job `20910344`) — FairChem ASE, not uma/kk |

---

## Phase 2b — engine / CLI E+F gate (`uma_parity_cli`)

FP64. Forces vs devices=1; energies vs d1 (and ASE where noted).

| Structure | devices | Job | dE_d1 | max\|ΔF\| | dE_ase |
|-----------|--------:|-----|------:|----------:|-------:|
| nacl64 | 2 | `20925398` | 0 | ~5.3×10⁻¹⁶ | — |
| NaCl6 1728 | 2 | `20925457` | 1.8×10⁻¹² | ~5.3×10⁻¹⁶ | ≈1.2×10⁻¹⁰ |
| nacl64 | 4 | `20925504` | **0** | **6.7×10⁻¹⁶** | — |
| NaCl6 1728 | 4 | `20925506` | **1.8×10⁻¹²** | **5.8×10⁻¹⁶** | **1.2×10⁻¹⁰** |

---

## Phase 3 — LAMMPS `uma/kk` E+F gate (product path)

NaCl6 1728 · FP64 · `gp=kokkos_libtorch_vesin` · single MPI rank.  
Gates vs `results/ngpu1` uma_double (devices=1) and ASE oracle.

| devices | Job | Energy (eV) | dE_d1 | max\|ΔF\| vs d1 | dE_ase | pair ms/eval\* |
|--------:|-----|-------------:|------:|-----------------:|-------:|---------------:|
| 1 | (baseline `ngpu1`) | −5830.9237201667 | — | 0 | 1.2×10⁻¹⁰ | **≈320** |
| 2 | `20925747` | −5830.9237201667 | **9.1×10⁻¹³** | **0** | 1.2×10⁻¹⁰ | **≈361** |
| 4 | `20925801` | −5830.9237201667 | **2.7×10⁻¹²** | **0** | 1.2×10⁻¹⁰ | **≈473** |

\*Honest pair-path timer from `run_multigpu` (`uma64 E=… XXX ms`). Do **not** use SLURM `wall/N_TIMING` (inflates with setup).

Thresholds: \|ΔE\| ≤ 1×10⁻⁸ · max\|ΔF\| ≤ 1×10⁻⁶ → **PASS** (devices 2 and 4).

Stamp gates:  
`agent_stamps/cpp_libtorch/lammps_gate_w2_20925747/gate.json` ·  
`agent_stamps/cpp_libtorch/lammps_gate_w4_20925801/gate.json`

### Timing note (honest)

Same-node Kokkos+LibTorch MP is **correctness-first**. Pair ms/eval at devices=2/4 (~361 / ~473) is **not** strong speedup vs devices=1 (~320); process-per-rank + host-staged peer gather adds overhead. Do not quote older Ray/Python GP ms (~192 / ~113 @2/@4) as the product backend.

---

## Reference — ASE / FC path-isolated batch (not product)

Historical FairChem timings (jobs `20910344`–`20910354`). Useful for ASE/FC scaling context only.

| Path | 1 GPU | 2 GPU | 4 GPU | Notes |
|------|------:|------:|------:|-------|
| ASE FairChem FP64 | 396.5 | 193.9 | 115.2 | `workers=N` → Ray ParallelMLIP |
| FairChem FC LAMMPS | 345.5 | 193.2 | 118.0 | FC cell FP32 → \|ΔE\|≈4.9×10⁻⁶ vs ASE |
| uma/kk (pre-C++ MP era) | 320.4 | *(obsolete)* | *(obsolete)* | Superseded by Phase 3 Kokkos+LibTorch numbers above |

---

## Findings

1. **Product path is Kokkos+LibTorch** (`kokkos_libtorch_vesin`), not Ray / FairChem eager GP / Python workers.
2. Engine CLI and LAMMPS agree: devices=2 and devices=4 E+F match devices=1 to numerical noise; forces vs d1 are exact (max\|ΔF\|=0) on the LAMMPS path.
3. Energy vs ASE FP64@1 stays ~10⁻¹⁰ eV on NaCl6.
4. Honest multi-GPU pair timing does **not** claim Ray-like speedups; correctness gate is the Phase 3 deliverable.
5. **Phase 5 (multi-node MPI-GP)** is out of scope for this report.

---

## File index

| Path | Role |
|------|------|
| `RESULTS.md` | This document (canonical Phase 4) |
| `SUMMARY.md` | Compact Phase 4 summary |
| `ngpu{1,2,4}/parity.json` | Merged energies (wall ms often contaminated) |
| `ngpu{1,2,4}/forces.npz` | Forces for parity |
| `gp_round/oracle_ase_fp64_w1.json` | ASE FP64 oracle |
| `../agent_stamps/cpp_libtorch/` | Campaign stamps + LAMMPS gate JSON |
