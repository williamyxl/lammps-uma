# Multi-GPU NaCl 6×6×6 — results (canonical)

**Phase 4 + Perf P1** · Stamp: 2026-08-08 ~09:00 CDT · Branch `uma-kokkos-mlip` @ `8e7e6a0d27` / report `d2bb98cf6c`  
**Suite:** `src/ML-UMA/examples/multi_gpu_nacl6/`  
**Status:** same-node graph-parallel **scientifically GREEN** (E+F) · **self-scale GREEN** (CUDA IPC)

## Product backend (uma/kk)

| Field | Value |
|-------|--------|
| Backend | **Kokkos + LibTorch** (`gp=kokkos_libtorch_vesin`) |
| Runtime | C++ `LibtorchMpRuntime` — process-per-rank workers + **CUDA IPC** `uma_peer` collectives (`UMA_PEER_TRANSPORT=cuda_ipc`) + vesin NL |
| Launch | `lmp -k on g N -sf kk` · `pair_style uma/kk precision double devices N` · **1 MPI rank** |
| Precision | **FP64 only** |
| Artifacts | `model_mp_w{N}_n{NATOMS}_r{R}.pt` (+ legacy `model_mp_w{N}_r*` for n=64) |
| Current pair ms | **≈320 / ≈265 / ≈193** @ devices=1/2/4 (job `20932975`) |
| **Not** product | Ray · FairChem `ParallelMLIPPredictUnit` · Python GP worker (env opt-in only) |

ASE FairChem / FC rows below are **reference baselines**; uma/kk is the product path. Timing for ASE/FC is from the path-isolated batch (jobs `20910344`–`20910354`); uma/kk timing is Perf P1 CUDA IPC (job `20932975`).

---

## Three-path comparison (NaCl6 1728, FP64)

Same frozen geometry. Oracle = ASE FairChem FP64 `workers=1` (−5830.9237201666 eV).

### Timing (honest ms/eval)

| Path | 1 GPU | 2 GPU | 4 GPU | 1→2 | 1→4 | Backend |
|------|------:|------:|------:|----:|----:|---------|
| ASE FairChem FP64 | 396.5 | 193.9 | 115.2 | 2.04× | 3.44× | Ray ParallelMLIP (`workers=N`) |
| FairChem FC LAMMPS | 345.5 | 193.2 | 118.0 | 1.79× | 2.93× | Ray ParallelMLIP in FC |
| **uma/kk double (product)** | **320.34** | **264.96** | **193.32** | **1.21×** | **1.66×** | Kokkos+LibTorch + CUDA IPC |

Jobs: ASE `20910344/48/52` · FC `20910345/49/53` · uma `20932975`.

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

uma/kk forces are identical across devices=1/2/4 to numerical noise vs each other (max\|ΔF\| vs uma d1 = **0** on P1); residual vs ASE is ~5×10⁻⁷ eV/Å (well under the 1×10⁻⁶ gate). FC is ~14× larger force error vs ASE than uma/kk.

### Readout

| Metric | Winner @1 GPU | Winner @4 GPU | Notes |
|--------|---------------|---------------|-------|
| Timing | **uma/kk** (320 vs 397/346) | ASE/FC (~115–118) | uma self-scales but trails Ray bandwidth @4 |
| Energy vs ASE | **uma/kk** (~1e-10) | **uma/kk** | FC stuck at ~5e-6 |
| Forces vs ASE | **uma/kk** (~5e-7) | **uma/kk** | FC ~7e-6 |

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
vs P0 host-shm (320.6 / 330.5 / 382.9) and Phase-3 pre-IPC (~361 / ~473). Soft gap vs ASE/FC Ray (~194 / ~115 @2/@4) remains; further wins are P2 pipe tax / optional.

### Timing note (honest)

Product path now **self-scales** with CUDA IPC collectives. Quote P1 job `20932975` numbers above — not Phase-3 host-shm (~361 / ~473) and not Ray/Python GP (~192 / ~113).

---

## Reference — ASE / FC path-isolated batch (not product)

Historical FairChem timings (jobs `20910344`–`20910354`). ASE/FC scaling context only — **not** uma/kk product numbers.

| Path | 1 GPU | 2 GPU | 4 GPU | Notes |
|------|------:|------:|------:|-------|
| ASE FairChem FP64 | 396.5 | 193.9 | 115.2 | `workers=N` → Ray ParallelMLIP |
| FairChem FC LAMMPS | 345.5 | 193.2 | 118.0 | FC cell FP32 → \|ΔE\|≈4.9×10⁻⁶ vs ASE |

**Current uma/kk pair ms** (product, Kokkos+LibTorch + CUDA IPC): **≈320 / ≈265 / ≈193** at devices=1/2/4 — job `20932975`. Do not quote Ray/Python GP or pre-IPC host-shm multi-GPU rows as the product backend.

---

## Findings

1. **Product path is Kokkos+LibTorch** (`kokkos_libtorch_vesin`), not Ray / FairChem eager GP / Python workers.
2. Engine CLI and LAMMPS agree: devices=2 and devices=4 E+F match devices=1 to numerical noise; forces vs d1 are exact (max\|ΔF\|=0) on the LAMMPS path.
3. Energy vs ASE FP64@1 stays ~10⁻¹⁰ eV on NaCl6.
4. **Self-scale green** after CUDA IPC (P1): pair ms falls 320 → 265 → 193 at devices=1/2/4 with E+F still PASS.
5. ASE/FC Ray still faster at 4 GPUs (~115 ms); further uma/kk wins are optional P2 (parent↔worker IPC).
6. **Phase 5 (multi-node MPI-GP)** is out of scope for this report.

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
