# NaCl 6×6×6 multi-GPU — Phase 4 + Perf P3c summary

**Stamp:** 2026-08-08 ~19:00 CDT · Campaign **PASS**

Fixed geometry `nacl6_rattle_fixed.extxyz` (1728 atoms).  
**Product:** `lmp -k on g N -sf kk` + `pair_style uma/kk precision double devices N` (`--ntasks=1`), backend **Kokkos+LibTorch** / `gp=kokkos_libtorch_vesin` + **NCCL** peer + **payload shm** + P3a pack/sync. FP64 only.  
**Not product:** Ray · FairChem ParallelMLIP · Python GP.

- **ASE oracle:** −5830.9237201666 eV
- **Parity / self-scale:** **PASS** (E+F green, max|ΔF|=0 vs d1)
- **Perf P3c job:** `20940474` (NCCL; prior hang `20940376` teardown fix)
- **Campaign bar:** uma ≤ ASE **and** ≤ FC at @1/@2/@4 · **PASS**

### LAMMPS uma/kk double (current = P3c NCCL)

| ngpu | pair ms\* | \|dE\| vs d1 | max\|ΔF\| vs d1 | vs ASE @N | vs FC @N | Job |
|-----:|----------:|-------------:|----------------:|----------:|---------:|-----|
| 1 | **321.04** | — | 0 | beats 396.5 | beats 345.5 | `20940474` |
| 2 | **183.30** | ~0 | **0** | **beats** 193.9 (−10.6) | **beats** 193.2 (−9.9) | `20940474` |
| 4 | **112.04** | 2.7×10⁻¹² | **0** | **beats** 115.2 (−3.2) | **beats** 118.0 (−6.0) | `20940474` |

\*Honest `uma64 E=… ms`. P3a cuda_ipc was 320.6 / 183.57 / 117.63.

### Three-path snapshot

| Path | ms @1/2/4 | \|ΔE\| vs ASE | max\|ΔF\| vs ASE |
|------|-----------|--------------:|-----------------:|
| ASE FairChem FP64 | 396.5 / 193.9 / 115.2 | ~0 | ~0 |
| FairChem FC LAMMPS | 345.5 / 193.2 / 118.0 | ≈4.9×10⁻⁶ | ≈7.1×10⁻⁶ |
| **uma/kk (P3c NCCL)** | **321.0 / 183.3 / 112.0** | ≈1.2×10⁻¹⁰ | ≈5×10⁻⁷ |

Canonical: [`RESULTS.md`](RESULTS.md). Campaign **PASS** on hard ≤ASE/FC @2/@4 with E+F green.
