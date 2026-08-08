# UMA multi-GPU NaCl 6×6×6 parity report

> **Canonical:** [`RESULTS.md`](RESULTS.md) · [`SUMMARY.md`](SUMMARY.md) · canvas `nacl6-multigpu-results`  
> **Stamp:** 2026-08-08 ~09:59 CDT · Perf P2 job `20933393`

## Setup

- **System:** NaCl 6×6×6 rocksalt, 1728 atoms (`structures/nacl6_rattle_fixed.extxyz`)
- **ASE oracle:** −5830.9237201666 eV (`gp_round/oracle_ase_fp64_w1.json`)
- **Product launch:** `lmp -k on g N -sf kk` · `pair_style uma/kk precision double devices N` · `--ntasks=1`
- **Backend:** Kokkos+LibTorch (`gp=kokkos_libtorch_vesin`) · **CUDA IPC** + **payload shm** fan-out
- **Precision:** FP64 only · mixed disabled

---

## Three-path comparison — ASE FairChem FP64 vs FC LAMMPS vs uma/kk FP64

Same frozen geometry. Oracle = ASE FairChem FP64 `workers=1`.

### Timing (honest ms/eval)

| Path | 1 GPU | 2 GPU | 4 GPU | 1→2 | 1→4 | Backend |
|------|------:|------:|------:|----:|----:|---------|
| ASE FairChem FP64 | 396.5 | 193.9 | 115.2 | 2.04× | 3.44× | Ray ParallelMLIP (`workers=N`) |
| FairChem FC LAMMPS | 345.5 | 193.2 | 118.0 | 1.79× | 2.93× | Ray ParallelMLIP in FC |
| **uma/kk double (product)** | **321.54** | **190.80** | **140.90** | **1.69×** | **2.28×** | Kokkos+LibTorch + CUDA IPC + payload shm |

Jobs: ASE `20910344/48/52` · FC `20910345/49/53` · uma **P2 `20933393`** (P1 was 320.34 / 264.96 / 193.32).

**@2:** uma **beats** ASE/FC. **@4:** soft ≲150 PASS; still ~26 ms behind ASE (~115).

### Energy vs ASE FP64@1

| Path | devices | Energy (eV) | \|ΔE\| vs ASE@1 |
|------|--------:|-------------:|----------------:|
| ASE FairChem FP64 | 1 | −5830.9237201666 | — (oracle) |
| ASE FairChem FP64 | 2 / 4 | −5830.9237201666 | ≲ 10⁻¹² |
| FairChem FC LAMMPS | 1 / 2 / 4 | −5830.9237152511 | **≈4.92×10⁻⁶** |
| **uma/kk double** | 1 / 2 / 4 | −5830.9237201667 | **≈1.2×10⁻¹⁰** |

### Per-atom forces vs ASE FP64@1

| Path | devices | max\|ΔF\| (eV/Å) | max‖ΔFᵢ‖ (eV/Å) | force cosine |
|------|--------:|-----------------:|----------------:|-------------:|
| ASE FairChem FP64 | 1 / 2 / 4 | ~10⁻¹⁶ | ~10⁻¹⁶ | 1.000000 |
| FairChem FC LAMMPS | 1 / 2 / 4 | **7.12×10⁻⁶** | **7.13×10⁻⁶** | 1.000000 |
| **uma/kk double** | 1 / 2 / 4 | **5.00×10⁻⁷** | **7.67×10⁻⁷** | 1.000000 |

uma devices=2/4 vs uma devices=1: max\|ΔF\| = **0**.

### Readout

| Metric | Winner @1 | Winner @2 | Winner @4 |
|--------|-----------|-----------|-----------|
| Timing | **uma/kk** | **uma/kk** | ASE/FC (~115–118) |
| Energy vs ASE | **uma/kk** | **uma/kk** | **uma/kk** |
| Forces vs ASE | **uma/kk** | **uma/kk** | **uma/kk** |

---

## Product gate (uma/kk vs devices=1) — P2

| devices | Energy (eV) | pair ms | \|dE\| vs d1 | max\|ΔF\| vs d1 | gate | Job |
|--------:|-------------:|--------:|-------------:|----------------:|:----:|-----|
| 1 | −5830.9237201667 | **321.54** | — | 0 | PASS | `20933393` |
| 2 | −5830.9237201667 | **190.80** | 2.7×10⁻¹² | **0** | **PASS** | `20933393` |
| 4 | −5830.9237201667 | **140.90** | 2.7×10⁻¹² | **0** | **PASS** | `20933393` |

**Self-scale:** PASS. Soft targets @2 ≲200 / @4 ≲150: **PASS**.

## Notes

- Tick policy: each parent 2‑min tick refreshes this report + canvas from latest `perf/summary_*.json` / honest `uma64` lines.
- Machine-readable: `SUMMARY.json` · `../agent_stamps/cpp_libtorch/perf/summary_20933393.json`.
