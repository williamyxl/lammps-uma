# NaCl 6×6×6 multi-GPU — Phase 4 + Perf P1 summary

Fixed geometry `nacl6_rattle_fixed.extxyz` (1728 atoms).  
**Product:** `lmp -k on g N -sf kk` + `pair_style uma/kk precision double devices N` (`--ntasks=1`), backend **Kokkos+LibTorch** / `gp=kokkos_libtorch_vesin` + **CUDA IPC** peer transport (`UMA_PEER_TRANSPORT=cuda_ipc`). FP64 only.  
**Not product:** Ray · FairChem ParallelMLIP · Python GP.

- **ASE oracle:** −5830.9237201666 eV (`gp_round/oracle_ase_fp64_w1.json`)
- **uma d1 reference:** devices=1 @ `ngpu1`
- **Parity gates (Phase 3 LAMMPS):** **2/2 PASS** (devices=2,4) — max|ΔF|=0
- **Self-scale (Perf P1):** **PASS** — `ms(2)<ms(1)` and `ms(4)<ms(2)`
- **Commits:** Phase 3 `5513482e9b` · P1 IPC `8e7e6a0d27` · report `d2bb98cf6c` on `uma-kokkos-mlip`

### Thresholds (uma vs devices=1)

| Mode | \|ΔE\| max | max\|ΔF\| | cosine min |
|------|----------:|----------:|-----------:|
| double (FP64) | 1×10⁻⁸ | 1×10⁻⁶ | ~1 |

### LAMMPS uma/kk double (product — current timings)

| ngpu | devices | Energy (eV) | pair ms\* | \|dE\| vs ASE | \|dE\| vs d1 | max\|ΔF\| vs d1 | gate | Job |
|-----:|--------:|-------------:|----------:|-------------:|-------------:|----------------:|:----:|-----|
| 1 | 1 | −5830.9237201667 | **320.34** | 1.2×10⁻¹⁰ | — | 0 | PASS | `20932975` |
| 2 | 2 | −5830.9237201667 | **264.96** | 1.2×10⁻¹⁰ | 0 | **0** | **PASS** | `20932975` |
| 4 | 4 | −5830.9237201667 | **193.32** | 1.2×10⁻¹⁰ | 1.8×10⁻¹² | **0** | **PASS** | `20932975` |

\*Honest pair-path `uma64 E=… ms` from `run_multigpu` (Perf P1, CUDA IPC). Ignore SLURM wall/`N_TIMING`.  
Phase-3 host-shm jobs `20925747`/`20925801` remain E+F green historical (~361 / ~473 ms) — **superseded for timing**.

### Engine CLI (Phase 2b)

| Structure | devices | Job | dE_d1 | max\|ΔF\| |
|-----------|--------:|-----|------:|----------:|
| nacl64 | 4 | `20925504` | 0 | 6.7×10⁻¹⁶ |
| NaCl6 1728 | 4 | `20925506` | 1.8×10⁻¹² | 5.8×10⁻¹⁶ |
| nacl64 / NaCl6 | 2 | `20925398` / `20925457` | 0 / 1.8×10⁻¹² | ~5×10⁻¹⁶ |

Canonical detail: [`RESULTS.md`](RESULTS.md). Phase 5 multi-node is later / out of scope.
