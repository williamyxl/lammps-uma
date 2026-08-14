# UMA on Polaris — campaign summary (authoritative)

Single source of truth for the LAMMPS + LibTorch-UMA scaling campaign on ALCF
Polaris (4x A100-40GB/node, PBS + Cray `mpiexec`, Slingshot/CXI). Individual docs
are point-in-time; this file states the final conclusions and links the details.

Model: UMA-S (`uma-s-1p2`), FP64, task=omat. 4 message-passing blocks x 6 A cutoff
=> ~24 A receptive field. Test system: NaCl NxNxN (8*N^3 atoms), per-atom random
displacement |d| in [0.05,0.10] A.

---

## 1. Single-node validation (P0) — PASS

LAMMPS-UMA vs ASE-FairChem FP64, 1/2/4 GPUs, NaCl666 + water888: energy/force at
the FP64 machine floor; multi-GPU (graph-parallel) bit-identical; ~1.9x/3.0x
step scaling on 2/4 GPUs. See `../../alcf_polaris_single_node.md`,
`multinode_mpi_plan.md` (Phase P0). Geometry precision matters: always feed the
full-precision LAMMPS `.data` (a reduced 8-digit .extxyz inflated water |dE| to
1e-7).

## 2. Multi-node MPI edge-parallel (M3) — CORRECT but SLOW

Built `MpiPeerPredictor` (one MPI rank per GPU across nodes; NCCL-over-MPI;
edge-parallel; memory-sharded shards). 8 GPU/2 node vs 4 GPU/1 node, NaCl 8x8x8
(4096): **energy/force bit-exact** (dE 2e-11, dF 9e-16) but **0.50x speed**
(407 vs 184 ms/step). Cause: FairChem graph parallel does a full-system
`all_gather` every block -> O(N) communication per layer and O(N) per-GPU
memory regardless of GPU count. Cross-node that latency dominates.
See `multinode_impl_polaris.md`.

## 3. Capacity levers (memory-bound) — the real multi-GPU win

Measured on 1 A100-40GB (NaCl, FP64, all bit-exact):

| lever | max atoms | vs base | cost | notes |
|---|---:|---:|---|---|
| baseline (traced) | 1,728 | 1.0x | - | current product |
| **activation checkpointing** | **21,952** | **12.7x** | 1.33x step | eager only (can't trace) |
| + save_on_cpu (offload) | 32,768 | 19x | 1.73x step | eager only |
| CUDA managed memory (A1) | unbounded | - | 10-18x | REJECTED (thrash) |
| edge chunking (C1) | - | - | - | no benefit (activations bind) |

4-GPU (graph-parallel):
| recipe | max atoms | box |
|---|---:|---:|
| baseline (no ckpt) | 8,000 (N=10) | 56 A |
| + checkpointing | 85,184 (N=22) | 124 A |

Details: `activation_checkpointing.md`, `cpu_offload_plan.md`,
`capacity_findings_4xa100.md`.

## 4. Checkpointing IN LAMMPS

- **Single-GPU** (`UMA_EAGER_CKPT=1`, devices 1 -> eager `uma_gp_worker.py`):
  works; ran NaCl 14^3 = **21,952 atoms** SP+NVT in LAMMPS (12.7x the traced
  ceiling).
- **Multi-GPU 1-node** (`UMA_EAGER_CKPT=1`, devices N -> `uma_dist_gp_worker.py`,
  real torch.distributed GP): works and is **bit-exact vs ASE** at N=20 (64,000)
  and N=21 (74,088): dE ~1e-10, max|dF| ~7e-14, ~5.5 s/step. (A force bug in the
  host-staged kgp emulation was root-caused and fixed by switching to real
  torch.distributed GP.)
- **Multi-node** (`MpiPeerPredictor`): added C++ gradient checkpointing
  (`CheckpointModuleFn`, recompute in backward; `UMA_MN_CKPT`, default ON).
  Runtime memory reduced, BUT N=21 end-to-end is **blocked upstream**: the traced
  w8 shards must be exported first, and `export_mp_artifact.py` traces on the
  full N-atom geometry with checkpointing OFF (tracing can't capture
  torch.utils.checkpoint) -> the **export OOMs at 74k atoms**.
See `lammps_checkpointing_Nge20.md`.

## 5. N=21 LAMMPS vs ASE (74,088 atoms, 4 GPU/1 node, checkpointed)

| metric | LAMMPS-UMA | ASE-FC | diff |
|---|---:|---:|---|
| Energy | -250375.82218906094 | -250375.82218906110 | \|dE\| 8.7e-11 eV |
| Force | \|F\|max 0.4576 | 0.4576 | max\|dF\| 7.1e-14, mean 2.0e-14 |
| Timing | 5,536 ms/step | 5,759 ms/SP | ~1.04x (noise) |

Machine-floor agreement at 9x the 4-GPU baseline ceiling, same per-step cost.

## 6. 2-node investigation — where it breaks

Layered isolation (see `two_node_investigation.md`):
- **Fabric is fine**: raw cross-node NCCL reduce_scatter of 720 MB = 194 ms.
- UMA GP works at small N on 8 GPU/2 node; **fails at large N** on (a) memory
  (GP doesn't shard node memory: 8 GPU peak == 4 GPU peak ~35 GiB) and (b) a
  rank-divergent collective count (ranks on the 2 nodes issue different numbers
  of collectives at large N -> NCCL desync/hang), and (c) the traced-shard
  export OOM.

---

## Bottom line: is UMA scalable for multi-node?

**Not for strong-scaling one system across nodes (speed).** UMA graph parallel is
O(N) communication per layer and O(N) memory per GPU regardless of world size, so
adding nodes does not speed up or enlarge a single system; cross-node it is
slower. True single-system multi-node speedup needs **spatial decomposition
(Scheme C: subdomains + ~6 A per-layer halo exchange)** — a model-parallelism
rewrite, not a config. The 24 A receptive field + two global couplings
(balance_channels, MOLE) are the architectural reasons.

**Yes for capacity and throughput.**
- **Capacity/GPU** scales via activation checkpointing (1,728 -> 21,952 on 1 GPU,
  exact; stacks with offload to 32,768). Most "bigger box" needs are met on 1-4
  GPUs without cross-node pain.
- **Throughput** scales near-linearly via **data parallelism** (many independent
  structures across all GPUs, no cross-node collectives) — the right multi-node
  mode for ensembles/sampling/screening.

The walls we hit at large-N multi-node are **tooling** (torch.jit.trace can't
checkpoint -> blocks large-N shard export) and **cross-node GP** (O(N) collective
desync/latency), NOT model correctness — the model runs multi-node and is
bit-exact where it fits.

## Open work (ordered)
1. Unblock large-N shards: dynamic-axis export, or a scriptable/checkpointable
   export, or drive the eager dist-GP path multi-node (fix the cross-node
   collective desync in `two_node_investigation.md`).
2. Scheme C (spatial halo) for real single-system multi-node speedup (large).
3. Productionize data-parallel ensemble runs across nodes (throughput).

## Doc index
- `multinode_mpi_plan.md` — overall plan incl. Phase P0.
- `../../alcf_polaris_single_node.md` — P0 1/2/4-GPU parity + timing report.
- `multinode_impl_polaris.md` — MPI edge-parallel (M3) design + result.
- `activation_checkpointing.md` — the checkpointing method.
- `cpu_offload_plan.md` — offload ladder (A1/A3/C1 measured).
- `capacity_findings_4xa100.md` — 1/4-GPU capacity ceilings (baseline vs ckpt).
- `lammps_checkpointing_Nge20.md` — checkpointing in LAMMPS incl. N=20/21 + MN.
- `two_node_investigation.md` — where 8-GPU/2-node breaks + improvement levers.
- `*_outdated.md` — superseded, kept for history.
