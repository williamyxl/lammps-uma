# NaCl 6×6×6 multi-GPU — Phase 4 + Perf P2 summary

Fixed geometry `nacl6_rattle_fixed.extxyz` (1728 atoms).  
**Product:** `lmp -k on g N -sf kk` + `pair_style uma/kk precision double devices N` (`--ntasks=1`), backend **Kokkos+LibTorch** / `gp=kokkos_libtorch_vesin` + **CUDA IPC** + **payload shm** fan-out. FP64 only.  
**Not product:** Ray · FairChem ParallelMLIP · Python GP.

- **ASE oracle:** −5830.9237201666 eV
- **Parity / self-scale:** **PASS** (E+F green, `ms(2)<ms(1)<` chain)
- **Perf P2 job:** `20933393` · commit `1a5f8a06c0`

### LAMMPS uma/kk double (current)

| ngpu | pair ms\* | \|dE\| vs d1 | max\|ΔF\| vs d1 | vs ASE @N | Job |
|-----:|----------:|-------------:|----------------:|----------:|-----|
| 1 | **321.54** | — | 0 | faster than ASE 396.5 | `20933393` |
| 2 | **190.80** | 2.7×10⁻¹² | **0** | **beats** ASE 193.9 / FC 193.2 | `20933393` |
| 4 | **140.90** | 2.7×10⁻¹² | **0** | +26 vs ASE 115.2 / +23 vs FC 118 | `20933393` |

\*Honest `uma64 E=… ms`. P1 was 320.34 / 264.96 / 193.32.

### Three-path snapshot

| Path | ms @1/2/4 | \|ΔE\| vs ASE | max\|ΔF\| vs ASE |
|------|-----------|--------------:|-----------------:|
| ASE FairChem FP64 | 396.5 / 193.9 / 115.2 | ~0 | ~0 |
| FairChem FC LAMMPS | 345.5 / 193.2 / 118.0 | ≈4.9×10⁻⁶ | ≈7.1×10⁻⁶ |
| **uma/kk (P2)** | **321.5 / 190.8 / 140.9** | ≈1.2×10⁻¹⁰ | ≈5×10⁻⁷ |

Canonical: [`RESULTS.md`](RESULTS.md). Remaining soft gap: devices=4 vs ASE/FC (~115–120).
