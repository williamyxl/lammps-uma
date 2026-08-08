# NaCl 6×6×6 multi-GPU — Phase 4 summary

Fixed geometry `nacl6_rattle_fixed.extxyz` (1728 atoms).  
**Product:** `lmp -k on g N -sf kk` + `pair_style uma/kk precision double devices N` (`--ntasks=1`), backend **Kokkos+LibTorch** / `gp=kokkos_libtorch_vesin`. FP64 only.  
**Not product:** Ray · FairChem ParallelMLIP · Python GP.

- **ASE oracle:** −5830.9237201666 eV (`gp_round/oracle_ase_fp64_w1.json`)
- **uma d1 reference:** devices=1 @ `ngpu1`
- **Parity gates (Phase 3 LAMMPS):** **2/2 PASS** (devices=2,4)
- **Commit:** `5513482e9b` on `uma-kokkos-mlip`

### Thresholds (uma vs devices=1)

| Mode | \|ΔE\| max | max\|ΔF\| | cosine min |
|------|----------:|----------:|-----------:|
| double (FP64) | 1×10⁻⁸ | 1×10⁻⁶ | ~1 |

### LAMMPS uma/kk double (product)

| ngpu | devices | Energy (eV) | pair ms\* | \|dE\| vs ASE | \|dE\| vs d1 | max\|ΔF\| vs d1 | gate | Job |
|-----:|--------:|-------------:|----------:|-------------:|-------------:|----------------:|:----:|-----|
| 1 | 1 | −5830.9237201667 | ≈320 | 1.2×10⁻¹⁰ | — | 0 | PASS (self) | ngpu1 baseline |
| 2 | 2 | −5830.9237201667 | ≈361 | 1.2×10⁻¹⁰ | 9.1×10⁻¹³ | **0** | **PASS** | `20925747` |
| 4 | 4 | −5830.9237201667 | ≈473 | 1.2×10⁻¹⁰ | 2.7×10⁻¹² | **0** | **PASS** | `20925801` |

\*Honest pair-path `uma64 E=… ms` from `run_multigpu`. Ignore SLURM wall/`N_TIMING`.

### Engine CLI (Phase 2b)

| Structure | devices | Job | dE_d1 | max\|ΔF\| |
|-----------|--------:|-----|------:|----------:|
| nacl64 | 4 | `20925504` | 0 | 6.7×10⁻¹⁶ |
| NaCl6 1728 | 4 | `20925506` | 1.8×10⁻¹² | 5.8×10⁻¹⁶ |
| nacl64 / NaCl6 | 2 | `20925398` / `20925457` | 0 / 1.8×10⁻¹² | ~5×10⁻¹⁶ |

Canonical detail: [`RESULTS.md`](RESULTS.md). Phase 5 multi-node is later / out of scope.
