> See [CAMPAIGN_SUMMARY.md](CAMPAIGN_SUMMARY.md) for the authoritative overview and final conclusions.

# Multi-node UMA on Polaris — implementation design (rev 2)

## Rev 2 — measured result of rev 1 (edge-parallel) and the redesign

**Rev 1 (MpiPeerPredictor, edge-parallel via FairChem GP) result on 8 GPU/2 node,
NaCl 8x8x8 = 4096:**
- Energy/force parity vs 4-GPU ground truth: **EXACT** (|dE| 2.0e-11 eV,
  max|dF| 9.2e-16). Cross-node NCCL bootstrap over MPI works; memory sharded.
- Timing: **0.50x** (407 vs 202 ms/step) — FAILS the >1.5x gate.

**Why (profiled, `UMA_MP_PERF`):** per step rank0 `ms_fwd≈130 ms_bwd≈260
ms_force_ar≈5`. The force all-reduce is negligible; the cost is fwd+bwd. The
FairChem GP path routes every message-passing block's node gather through
`torch.ops.uma_peer.all_gather_nodes`, which reconstructs the FULL N-atom node
tensor on every rank (`all_gather_concat`). So per-rank COMPUTE is O(N), not
O(N/W); adding GPUs cannot speed the step, and moving those 4-layer x (fwd+bwd)
gathers across nodes (IB vs NVLink) doubles it. This is the documented GP
limitation (plan sec 3): GP buys capacity, not speed.

**Redesign (rev 2) = Scheme C, at the single substitution point.** Keep the whole
GP machinery (shards, NCCL, autograd custom ops, the global `all_reduce_sum` for
balance_channels/MOLE), but replace the O(N) node gather with a SPATIAL halo
gather so per-rank node tensors are O(nlocal + nghost_halo):

```
all_gather_nodes(x, N)   -> full [N, C]      (rev 1: O(N)/rank, the bottleneck)
halo_gather(x, plan)     -> [nlocal+nghost, C] (rev 2: O(N/W), real compute shrink)
```

Crossover (NaCl density, 6 A halo, cube split): even at N=4096 the 8-GPU halo
set is ~1601 atoms/rank vs 4096 for the 4-GPU whole-system path — a 2.56x
compute reduction; it grows with N (3.0x @ 8000, 4.1x @ 32768). So Scheme C can
clear >1.5x once comm is overlapped/amortized.

**Substitution point:** `uma_peer_op_all_gather_nodes` (peer_context.cpp:57) and
its autograd (`all_gather_nodes_autograd`, backward = scatter-add of the halo
region). The global `all_reduce_sum` op (balance_channels, MOLE composition)
stays a true global reduce — 3 doubles, latency-only, mandatory for correctness.

**Halo plan:** built from the full tag-ordered graph (already assembled each
step): for this rank's owned node set (FairChem `node_partition`), the halo is
the set of source nodes appearing in edges whose center is owned, iterated to
cover the receptive field the model actually reaches per block. v1: exchange the
1-hop (6 A) neighbor node features each block (bit-comparable to serial since the
owner computed them), backward scatter-adds ghost grads to owners.

Everything below is rev 1 (still valid: transport, NCCL bootstrap, pair_uma
wiring). Only the gather op semantics change.

---

# Multi-node UMA on Polaris — implementation design (rev 1)

**Goal:** run NaCl 8x8x8 (4096) and larger across **8 GPUs / 2 nodes** with the
**edge-parallel, memory-sharded** model (same math as the 4-GPU/1-node ground
truth), reproducing its energy + per-atom forces and showing >1.5x NVT scaling
vs 1 node.

## Why the previous multi-node code was wrong

`pair_uma.cpp`'s `mn_active` branch is **Scheme A replicated**: every MPI rank
calls `predictor->predict_host(natoms_global, ...)` with a **`devices 1` traced
full-system model**. Memory is O(N) per GPU, so 4096 OOMs on any node count.
That path is deleted/replaced here.

## Architecture: MPI ranks ARE the graph-parallel peers

One `mpiexec` job, **W ranks = W GPUs** across nodes, LAMMPS MPI-parallel as
usual. For the UMA force evaluation every rank is one edge-parallel peer:

```
mpiexec -n W --ppn 4  (Polaris: 4 GPUs/node)
  rank r  ->  GPU (PMI_LOCAL_RANK % 4)
             loads model_mp_w{W}_n{N}_r{r}.pt   (its edge shard)
             holds all atoms, 1/W of the edges  -> memory ~O(N/W)
```

Per MD step, inside `PairUMA::compute` (multi-rank branch):

1. **Gather** the tag-ordered global atom set (positions, Z) with
   `MPI_Allgatherv` — already implemented and unit-tested; keep it. Every rank
   builds the identical full-system input.
2. **Predict on the shard:** each rank runs its `model_mp_w{W}_n{N}_r{r}`
   module, which internally partitions edges by center atom (baked at export)
   and calls `uma_peer` collectives for the per-layer reductions.
3. **Force all-reduce over NCCL** (`SharedPeerGatherSlot::all_reduce`) — sums
   the per-shard force contributions across all W GPUs. Same op as the
   single-node path.
4. **Scatter** owned forces back to each rank's local atoms; rank 0 contributes
   `eng_vdwl`.

Transport differences vs single-node GP (the only things that change):

| concern | single-node (fork) | multi-node (this) |
|---|---|---|
| peer processes | rank 0 forks W workers | mpiexec launches W ranks (= LAMMPS ranks) |
| geometry to peers | /dev/shm payload | already local: MPI_Allgatherv result |
| NCCL bootstrap | ncclUniqueId via /dev/shm | rank 0 `make_unique_id` -> `MPI_Bcast` -> `init_nccl_external` |
| GPU binding | CUDA_VISIBLE_DEVICES per worker | `PMI_LOCAL_RANK % 4`, pass device_index to init_nccl_external |

`init_nccl_external(rank, world, id, device_index)`, `make_unique_id`,
`unique_id_bytes` already exist in `shared_peer.h` for exactly this.

## New engine entry point

Add a **`ShardedPeerPredictor`** (or extend `LibtorchMpRuntime`) usable by an
in-process peer (no fork). Minimal surface:

```cpp
// one per MPI rank; wraps the rank's shard module + NCCL peer
class MpiPeerPredictor {
  static std::unique_ptr<MpiPeerPredictor> create(
      const std::string& artifact_dir, const ArtifactMetadata& md,
      int world, int rank, int device_index, const void* nccl_unique_id,
      torch::ScalarType dtype);
  // full global system in; this rank's shard evaluated; forces all-reduced
  Prediction predict_host(int n, const double* pos, const int* z,
                          const double* cell3x3, const int* pbc3,
                          double* forces_out /*global [n,3], reduced*/);
};
```

Internally this is the body of `uma_libtorch_mp_worker.cpp`'s loop, minus the
pipe/shm plumbing: H2D from the caller's arrays, `graph_shard::shard_edges`,
fwd+bwd, `slot.all_reduce(forces)`, return reduced forces to host. It reuses
`PeerContext`, `graph_shard`, `postprocess`, `vesin_nl`, `shared_peer` verbatim.

## pair_uma.cpp changes

- `settings()`: allow `devices > 1` with `nprocs > 1` **iff** `nprocs == devices`
  (one rank per GPU across nodes). Remove the blanket refusal for that case;
  keep refusing `devices>1 && nprocs>1 && nprocs!=devices` (mixed fork+MPI).
- `load_predictor()`: when `mn_world > 1`, bootstrap NCCL:
  - rank 0 `make_unique_id`, `MPI_Bcast` the `unique_id_bytes()` buffer;
  - `device_index = local_rank % gpus_per_node`;
  - construct `MpiPeerPredictor` with (world=mn_world, rank=mn_rank, id, dev).
- `compute()` multi-rank branch: keep the tag-ordered gather; replace the
  `predictor->predict_host(natoms_global,...)` (Scheme A) with the
  `MpiPeerPredictor::predict_host` (sharded). Scatter owned forces; rank 0 adds
  energy. Drop the `!use_f64` restriction message only after FP64 verified.

## Shards

Reuse `model_mp_w{W}_n{N}_r{R}.pt` from `export_mp_artifact.py` (already used by
the fork path). w4_n4096 exists (ground truth). Need **w8_n4096** for the
2-node run — export once (8 single-GPU exports, 2 waves on a 4-GPU node).

## Correctness gate

8-GPU/2-node SP energy and per-atom forces must match the 4-GPU/1-node ground
truth (E = -13841.956154912199 eV) to the FP64 reduction-order floor
(|dE| <= 1e-6, max|dF| <= 1e-5). Both use the same edge-parallel math and the
same NCCL all-reduce, so agreement is expected to be tight.

## Timing gate

NVT-300K ms/step on 8 GPUs/2 nodes vs the 1-node baseline (4 GPUs): target
> 1.5x. (Edge-parallel DOES shard compute, unlike Scheme A, so a speedup is
expected here — bounded by cross-node NCCL latency for the per-layer reductions.)

## Risks

- Cross-node NCCL latency on the per-layer collectives may dominate at 4096;
  mitigate by measuring, and if needed raise atoms or reduce layers-per-comm.
- Shard export bakes W and N: changing either needs re-export.
- `MPI_Bcast` of ncclUniqueId must complete before any rank calls
  `ncclCommInitRank` (all ranks call it collectively) — order carefully in
  `load_predictor`.
