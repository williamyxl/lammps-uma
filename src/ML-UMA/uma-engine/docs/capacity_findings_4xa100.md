> See [CAMPAIGN_SUMMARY.md](CAMPAIGN_SUMMARY.md) for the authoritative overview and final conclusions.

# UMA capacity on 4x A100-40GB (Polaris) — NaCl NxNxN, FP64

System: NaCl rocksalt, N conventional cells/side = **8·N^3 atoms**, box = N·5.64 A,
per-atom random displacement |d| in [0.05,0.10] A (seed 0). All graph-parallel
(FairChem GP over torch.distributed, Ray-free). "fits" = single-point E + per-atom
force (autograd) completes without CUDA OOM.

## Headline

| recipe | max N | max atoms | box | where verified |
|--------|:-----:|----------:|----:|----------------|
| **baseline** (no checkpointing, no offload) | **10** | **8,000** | 56 A | ASE-FC 4-GPU sweep |
| **+ activation checkpointing** | **22** | **85,184** | 124 A | ASE-FC FP64 4-GPU sweep |

Activation checkpointing raises the 4-GPU ceiling **8,000 -> 85,184 atoms (10.6x)**,
bit-exact (recompute in backward), for ~1.33x step time.

## Measured sweeps (4 GPU, per-rank peak)

Baseline (checkpointing OFF), N=8..14:
| N | atoms | box | peak/GPU | fit |
|--:|------:|----:|---------:|:---:|
| 8 | 4,096 | 45.1 A | 19.1 GiB | OK |
| **10** | **8,000** | **56.4 A** | 35.1 GiB | **OK (max)** |
| 11 | 10,648 | 62.0 A | 36.8 GiB | OOM |
| 12 | 13,824 | 67.7 A | 35.7 GiB | OOM |

Checkpointing ON, N=8..24:
| N | atoms | box | peak/GPU | fit |
|--:|------:|----:|---------:|:---:|
| 20 | 64,000 | 112.8 A | 31.0 GiB | OK |
| **22** | **85,184** | **124.1 A** | 35.5 GiB | **OK (max)** |
| 24 | 110,592 | 135.4 A | — | OOM |

## N=11 baseline check (explicit, without checkpointing/offload)

NaCl 11^3 = 10,648 atoms **does NOT fit** on 4 GPUs without checkpointing:
ASE-FC 4-GPU baseline OOMs at 36.8 GiB/GPU. Consistent with the baseline ceiling
of 8,000 (N=10). The LAMMPS-UMA libtorch traced 4-GPU path uses the same
non-checkpointed model / same ~37 GiB/GPU, so it also does not fit (its run
additionally tripped a same-node C++ MP transport config bug — see below).

## What was tested in LAMMPS vs model-level (ASE)

| N | atoms | ASE-FC (model) | LAMMPS-UMA libtorch |
|--:|------:|----------------|---------------------|
| 8  | 4,096  | fits (baseline & ckpt) | **checkpointing: SP+NVT OK** (1 GPU); traced 1-GPU OOMs |
| 14 | 21,952 | fits (ckpt) | **checkpointing: SP+NVT OK** (1 GPU, ~79 A box) |
| 22 | 85,184 | **fits (ckpt, 4 GPU)** | **NOT TESTED** — needs 4-GPU eager ckpt |

**LAMMPS checkpointing is single-GPU only today** (`UMA_EAGER_CKPT=1`, devices 1;
ceiling N=14 = 21,952). Reaching N=22 (85,184) in LAMMPS requires **4-GPU eager
checkpointing**: drive the eager worker across 4 GPUs via torch.distributed GP
(the mechanism proven by gp_mpi_sweep.py at the model level), then wire that into
pair_uma. This integration is the outstanding step; N=22 in LAMMPS is not yet run.

## Known bug (separate from capacity)

The same-node traced 4-GPU C++ MP path (`build-uma-mn/lmp`, devices 4) defaults to
NCCL transport (`select_transport()` prefers NCCL now that the MN engine links it)
and fails with "nccl requested without NCCL" from the fork worker; forcing
`UMA_PEER_TRANSPORT=cuda_ipc` gets past that but then MPI_Aborts. Same-node fork
workers should not use NCCL. Fix: default same-node fork MP to cuda_ipc/shm (NCCL
only for the cross-node MPI-peer path). Does not affect the capacity conclusions.

## Reproduce
- 4-GPU sweeps (Ray-free GP): `polaris/pbs/cap_sweep_4gpu_gp_baseline.pbs`
  (ckpt off), `cap_sweep_4gpu_gp.pbs` / `_hi.pbs` (ckpt on) -> `gp_mpi_sweep.py`.
- N=11 baseline parity: `polaris/pbs/n11_baseline_parity.pbs`.
- LAMMPS single-GPU checkpointing: `polaris/pbs/lammps_eager_ckpt.pbs` (M3_NREP=N).
