# UMA multi-GPU NaCl 6×6×6 parity report

> **Canonical (current):** [`RESULTS.md`](RESULTS.md) · [`SUMMARY.md`](SUMMARY.md).  
> This file is a short product-facing mirror. Do **not** use the obsolete FAIL/wall-time tables that previously lived here.

## Setup

- **System:** NaCl 6×6×6 rocksalt, 1728 atoms (`structures/nacl6_rattle_fixed.extxyz`)
- **ASE oracle:** −5830.9237201666 eV (`gp_round/oracle_ase_fp64_w1.json`)
- **Product launch:** `lmp -k on g N -sf kk` · `pair_style uma/kk precision double devices N` · `--ntasks=1`
- **Backend:** Kokkos+LibTorch (`gp=kokkos_libtorch_vesin`) · peer **CUDA IPC** (`UMA_PEER_TRANSPORT=cuda_ipc`)
- **Precision:** FP64 only · mixed disabled
- **Primary gate:** uma/kk `devices=N` vs `devices=1` (E + per-atom F)

### Parity thresholds (vs devices=1)

| Mode | \|ΔE\| max | max\|ΔF\| |
|------|----------:|----------:|
| double | 1×10⁻⁸ | 1×10⁻⁶ |

## Product results (Perf P1, job `20932975`)

| Path | devices | Energy (eV) | pair ms\* | \|dE\| vs d1 | max\|ΔF\| vs d1 | gate |
|------|--------:|-------------:|----------:|-------------:|----------------:|:----:|
| uma/kk double | 1 | −5830.9237201667 | **320.34** | — | 0 | PASS |
| uma/kk double | 2 | −5830.9237201667 | **264.96** | 0 | **0** | **PASS** |
| uma/kk double | 4 | −5830.9237201667 | **193.32** | 1.8×10⁻¹² | **0** | **PASS** |

\*Honest `uma64 E=… ms` from `run_multigpu` — not SLURM wall.

**Self-scale:** PASS (`ms(2)<ms(1)`, `ms(4)<ms(2)`).

## Reference timings (not product)

| Path | 1 GPU | 2 GPU | 4 GPU | Notes |
|------|------:|------:|------:|-------|
| ASE FairChem FP64 | 396.5 | 193.9 | 115.2 | Ray ParallelMLIP |
| FairChem FC LAMMPS | 345.5 | 193.2 | 118.0 | FC cell may be FP32 |
| uma/kk (pre-IPC Phase 3) | ≈320 | ≈361 | ≈473 | host `/dev/shm` — superseded |

## Notes

- Same-node only (`gpuA100x4`); Phase 5 multi-node out of scope.
- Machine-readable: `SUMMARY.json` · stamps `../agent_stamps/cpp_libtorch/perf/summary_20932975.json`.
