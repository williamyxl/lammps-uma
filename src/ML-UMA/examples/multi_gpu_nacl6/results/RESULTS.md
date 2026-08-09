# Multi-GPU NaCl 6×6×6 — results (canonical)

**Phase 4 + Perf P3c** · Stamp: 2026-08-08 ~19:00 CDT · Branch `uma-kokkos-mlip` · P3c `20940474`  
**Campaign bar:** uma ≤ ASE **and** ≤ FC · **PASS** · honest **321.04 / 183.30 / 112.04** (NCCL) · E+F green · @4 −3.2 ms vs ASE / −6.0 ms vs FC

## Product backend (uma/kk)

| Field | Value |
|-------|--------|
| Backend | **Kokkos + LibTorch** (`gp=kokkos_libtorch_vesin`) |
| Runtime | C++ `LibtorchMpRuntime` — process-per-rank + **NCCL** peer (`UMA_PEER_TRANSPORT=nccl`) + **payload shm** + P3a pack/sync + vesin NL |
| Launch | `lmp -k on g N -sf kk` · `pair_style uma/kk precision double devices N` · **1 MPI rank** |
| Precision | **FP64 only** |
| Artifacts | `model_mp_w{N}_n{NATOMS}_r{R}.pt` (+ legacy `model_mp_w{N}_r*` for n=64) |
| Current pair ms | **321.04 / 183.30 / 112.04** @ devices=1/2/4 (job `20940474`, NCCL) |
| Prior P3a (cuda_ipc) | 320.6 / 183.57 / 117.63 (`20934280`) |
| **Not** product | Ray · FairChem `ParallelMLIPPredictUnit` · Python GP worker (env opt-in only) |

ASE FairChem / FC rows below are **reference baselines**; uma/kk is the product path. ASE/FC timing: path-isolated batch (`20910344`–`20910354`); uma/kk: Perf P3c (`20940474`).

---

## Three-path comparison (NaCl6 1728, FP64)

Same frozen geometry. Oracle = ASE FairChem FP64 `workers=1` (−5830.9237201666 eV).

### Timing (honest ms/eval)

| Path | 1 GPU | 2 GPU | 4 GPU | 1→2 | 1→4 | Backend |
|------|------:|------:|------:|----:|----:|---------|
| ASE FairChem FP64 | 396.5 | 193.9 | 115.2 | 2.04× | 3.44× | Ray ParallelMLIP (`workers=N`) |
| FairChem FC LAMMPS | 345.5 | 193.2 | 118.0 | 1.79× | 2.93× | Ray ParallelMLIP in FC |
| **uma/kk double (product)** | **321.04** | **183.30** | **112.04** | **1.75×** | **2.87×** | Kokkos+LibTorch + **NCCL** (P3c) |

Jobs: ASE `20910344/48/52` · FC `20910345/49/53` · uma P3c `20940474` (P3a cuda_ipc was 320.60 / 183.57 / 117.63).

### Energy vs ASE FP64@1

| Path | devices | Energy (eV) | \|ΔE\| vs ASE@1 |
|------|--------:|-------------:|----------------:|
| ASE FairChem FP64 | 1 | −5830.9237201666 | — (oracle) |
| ASE FairChem FP64 | 2 / 4 | −5830.9237201666 | ≲ 10⁻¹² |
| FairChem FC LAMMPS | 1 / 2 / 4 | −5830.9237152511 | **≈4.92×10⁻⁶** |
| **uma/kk double** | 1 / 2 / 4 | −5830.9237201667 | **≈1.2×10⁻¹⁰** |

FC energy offset is consistent with FC cell build in FP32 (documented). uma/kk stays ~10⁻¹⁰ eV of the ASE FP64 oracle at all device counts.

### Per-atom forces vs ASE FP64@1

| Path | devices | max\|ΔF\| (eV/Å) | max‖ΔFᵢ‖ (eV/Å) | force cosine |
|------|--------:|-----------------:|----------------:|-------------:|
| ASE FairChem FP64 | 1 / 2 / 4 | ~10⁻¹⁶ | ~10⁻¹⁶ | 1.000000 |
| FairChem FC LAMMPS | 1 / 2 / 4 | **7.12×10⁻⁶** | **7.13×10⁻⁶** | 1.000000 |
| **uma/kk double** | 1 / 2 / 4 | **5.00×10⁻⁷** | **7.67×10⁻⁷** | 1.000000 |

uma/kk forces identical across devices=1/2/4 vs each other (max|ΔF| vs uma d1 = **0** on P3c); residual vs ASE ~5×10⁻⁷ eV/Å.

### Readout

| Metric | Winner @1 GPU | Winner @2 GPU | Winner @4 GPU | Notes |
|--------|---------------|---------------|---------------|-------|
| Timing | **uma/kk** | **uma/kk** | **uma/kk** (−3.2 vs ASE, −6.0 vs FC) | Campaign **PASS** |
| Energy vs ASE | **uma/kk** (~1e-10) | **uma/kk** | **uma/kk** | FC stuck at ~5e-6 |
| Forces vs ASE | **uma/kk** (~5e-7) | **uma/kk** | **uma/kk** | FC ~7e-6 |

P3b: residual vs ASE was in worker collectives. P3c NCCL (`20940474`) closed it: @4 **112.04** ≤ ASE **115.2** and ≤ FC **118**. Hang `20940376` was teardown deadlock (fixed: broadcast shutdown + destroy rendezvous).

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
| 2 | `20925747` | −5830.9237201667 | **9.1×10⁻¹³** | **0** | 1.2×10⁻¹⁰ | ≈361 (pre-P1) |
| 4 | `20925801` | −5830.9237201667 | **2.7×10⁻¹²** | **0** | 1.2×10⁻¹⁰ | ≈473 (pre-P1) |

\*Honest pair-path timer from `run_multigpu` (`uma64 E=… XXX ms`). Do **not** use SLURM `wall/N_TIMING` (inflates with setup).

Thresholds: \|ΔE\| ≤ 1×10⁻⁸ · max\|ΔF\| ≤ 1×10⁻⁶ → **PASS** (devices 2 and 4).

Stamp gates:  
`agent_stamps/cpp_libtorch/lammps_gate_w2_20925747/gate.json` ·  
`agent_stamps/cpp_libtorch/lammps_gate_w4_20925801/gate.json`

### Perf P1 — CUDA IPC self-scale (job `20932975`)

`UMA_PEER_TRANSPORT=cuda_ipc` (default): device payloads via `cudaIpcMemHandle_t`; E+F still green.

| devices | pair ms/eval | vs devices=1 | dE_d1 | max\|ΔF\| |
|--------:|-------------:|-------------:|------:|----------:|
| 1 | **320.34** | — | 1.8×10⁻¹² | **0** |
| 2 | **264.96** | **0.83×** | 0 | **0** |
| 4 | **193.32** | **0.60×** | 1.8×10⁻¹² | **0** |

**Hard gate PASS:** `ms(2)<ms(1)` and `ms(4)<ms(2)`.  
vs P0 host-shm (320.6 / 330.5 / 382.9) and Phase-3 pre-IPC (~361 / ~473). Historical milestone only — **current product = P3c NCCL** below.

### Perf P3a → P3c (campaign close)

| Phase | Job | Transport | pair ms @1/2/4 | Notes |
|-------|-----|-----------|----------------:|-------|
| P3a | `20934280` | cuda_ipc | 320.6 / 183.57 / 117.63 | Beat FC @4; **+2.4 vs ASE** → OPEN |
| P3c hang | `20940376` | nccl | 321.6 / — / — | Teardown deadlock (fixed) |
| **P3c** | **`20940474`** | **nccl** | **321.04 / 183.30 / 112.04** | **PASS** ≤ASE/FC @1/2/4 |

### Timing note (honest)

Quote **`uma64 E=… XXX ms`** from `run_multigpu` / PERF gates — **not** SLURM `wall/N_TIMING`. Canonical product numbers are P3c NCCL job `20940474`.

---

## Reference — ASE / FC path-isolated batch (not product)

Historical FairChem timings (jobs `20910344`–`20910354`). ASE/FC scaling context only — **not** uma/kk product numbers.

| Path | 1 GPU | 2 GPU | 4 GPU | Notes |
|------|------:|------:|------:|-------|
| ASE FairChem FP64 | 396.5 | 193.9 | 115.2 | `workers=N` → Ray ParallelMLIP |
| FairChem FC LAMMPS | 345.5 | 193.2 | 118.0 | FC cell FP32 → \|ΔE\|≈4.9×10⁻⁶ vs ASE |

**Current uma/kk pair ms** (product, Kokkos+LibTorch + **NCCL**): **321.04 / 183.30 / 112.04** at devices=1/2/4 — job `20940474`. Opt-in via `UMA_PEER_TRANSPORT=nccl` (code default remains cuda_ipc fallback). Do not quote Ray/Python GP, pre-IPC host-shm, or P1-only rows as the current product backend.

---

## Findings

1. **Product path is Kokkos+LibTorch** (`kokkos_libtorch_vesin`), not Ray / FairChem eager GP / Python workers.
2. Engine CLI and LAMMPS agree: devices=2 and devices=4 E+F match devices=1 to numerical noise; forces vs d1 are exact (max\|ΔF\|=0) on the LAMMPS path.
3. Energy vs ASE FP64@1 stays ~10⁻¹⁰ eV on NaCl6.
4. **Self-scale green** from P1 onward; P3c NCCL reaches **321 → 183 → 112** ms @1/2/4 with E+F still PASS.
5. **Campaign PASS:** uma beats ASE and FC at every GPU count (hard bar). Margin @4 vs ASE is −3.2 ms.
6. Remaining headroom: default NCCL (still opt-in), parent NL+publish (~8–10 ms @4), scaling efficiency (1→4 = 2.87× vs ASE 3.44×).
7. **Phase 5 (multi-node MPI-GP)** is out of scope for this report.

---

## File index

| Path | Role |
|------|------|
| `RESULTS.md` | Canonical — **three-path** ASE/FC/uma timing + E + F + product gates |
| `MULTIGPU_REPORT.md` | Same three-path tables (report mirror) |
| `SUMMARY.md` / `SUMMARY.json` | Compact + `three_path_compare` JSON |
| `ngpu{1,2,4}/parity.json` | Merged energies (wall ms often contaminated) |
| `ngpu{1,2,4}/forces.npz` | Forces for ASE/FC/uma parity |
| `gp_round/oracle_ase_fp64_w1.json` | ASE FP64 oracle |
| `../agent_stamps/cpp_libtorch/` | Campaign stamps + LAMMPS / perf gate JSON |
