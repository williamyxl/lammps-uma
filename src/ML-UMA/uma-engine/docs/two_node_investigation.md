# 2-node (8 GPU) investigation — where it breaks and where to improve

Question: can N=21 (NaCl 21^3 = 74,088 atoms) single point be sped up on 8 GPUs /
2 nodes vs 4 GPUs / 1 node (5,118 ms/SP, checkpointed)?

## Layered isolation (what actually fails)

| test | atoms | GPUs/nodes | ckpt | result |
|------|------:|:---------:|:----:|--------|
| raw NCCL all_reduce/reduce_scatter, up to 720 MB | - | 8 / 2 | - | **OK** (reduce_scatter 720 MB = 194 ms) |
| UMA GP small | 1,728 | 8 / 2 | ON | **OK** (200.8 ms/SP) |
| UMA GP small | 1,728 | 8 / 2 | OFF | **OK** (133.4 ms/SP) |
| UMA GP big | 74,088 | 8 / 2 | OFF | **OOM** (35.3 GiB/GPU) |
| UMA GP big | 74,088 | 8 / 2 | ON | **hang** (REDUCE_SCATTER SeqNum=11, 600 s timeout) |

## Root cause

**The fabric is NOT the problem** — raw cross-node NCCL moves the exact 720 MB
reduce_scatter that "hung" in UMA in 194 ms. The 8-GPU/2-node GP path works fine
at small N. It breaks at large N because of **per-GPU MEMORY**, not comm:

1. **FairChem GP does not shard memory with more GPUs.** Each message-passing
   block does `all_gather_nodes` -> the FULL [N, C] node tensor is materialized
   on EVERY rank every layer. Going 4->8 GPUs halves the owned node partition but
   the gathered full tensor stays O(N). So 8-GPU peak ~= 4-GPU peak (~35 GiB at
   N=21). This is the documented GP property: capacity per GPU does NOT grow with
   GPU count; only compute is (partially) divided.
2. Therefore at N=21: 4-GPU **checkpointed** fits (~35 GiB); 8-GPU **un-checkpointed**
   OOMs (~35 GiB + larger NCCL buffers for 8 ranks). 8-GPU **checkpointed** should
   fit on memory but **hung** — one rank stalls (near-OOM / different code path)
   so the ranks desync at the 11th collective and the others wait forever.

## Consequence for "speedup"

- GP is **O(N) communication per layer** and **O(N) memory per GPU regardless of
  world size**. So 2-node GP was never going to *speed up* the step (earlier M3
  at 4096: 4-GPU/1-node 184 ms vs 8-GPU/2-node 407 ms — 2x SLOWER, cross-node
  all_gather latency added, no compute win). It buys neither speed NOR extra
  capacity per GPU.
- 2-node with this architecture is only useful for **data parallel** (many
  independent systems), not for one larger/faster system.

## Refined root cause (rev 2): rank-divergent collective COUNT, node-correlated

With `expandable_segments` + async error handling the 8-GPU-ckpt N=21 run no
longer OOMs but still fails — and the NCCL error is explicit:

  Rank 3 (node 1): last completed work 20, last enqueued 27
  Ranks 4-7 (node 2): last completed work 2
  "most likely caused by ... the order of collectives is not the same for all
   ranks" (ProcessGroupNCCL)

So the two nodes' ranks execute a DIFFERENT NUMBER of collectives at large N —
node 1 raced to work #20+, node 2 stalled at #2. This is a **rank-divergent
collective schedule**, not memory and not the fabric. Small N (1,728) does not
diverge (works on 8 GPU); large N (74,088) does. Likely a data/shape-dependent
branch in the GP model (node_partition via torch.tensor_split gives uneven
shards; a per-rank shape/expert/mask conditional then changes the collective
count) that only manifests when the partition is large/uneven across 8 ranks.

This is the concrete thing to fix for 2-node correctness: make the GP collective
sequence rank-invariant (no data-dependent collective count), or pad/align the
node partition so every rank issues identical collectives. It is an
upstream-FairChem-GP interaction, exposed at 8-way GP + large N.

## Where 2-node CAN improve (actionable)

1. **Reduce per-GPU memory so 8-GPU large-N fits, then measure honestly**
   (still won't beat 4-GPU on speed, but completes):
   - always checkpoint on the multi-node path;
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the OOM msg suggests it);
   - lower `max_neighbors` (300 -> ~50) to cut edge memory;
   - cap NCCL buffers (`NCCL_BUFFSIZE`) for 8 ranks.
2. **Fix the desync-hang robustness**: the 8-GPU-ckpt hang is a rank diverging
   (near-OOM) then collectives mismatch. With headroom (item 1) it should not
   diverge; also set `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` so a diverged rank
   aborts the group instead of a 600 s hang.
3. **The only path to real 2-node SPEEDUP is spatial decomposition (Scheme C)**:
   halo exchange moves O(surface) ~ O((N/W)^2/3), not O(N), and shrinks per-GPU
   memory as W grows. This is the model-surgery item (replace the per-block
   all_gather with a 6 A halo exchange). Everything else is capacity/robustness,
   not speed.

## Data-parallel alternative (works today, real throughput)
For sampling/throughput (many structures), run W independent single-GPU
(checkpointed) LAMMPS-UMA instances across the 8 GPUs — near-linear throughput,
no cross-node collectives. This is the pragmatic 2-node win for ensembles.

## Reproduce
- Fabric probe: `polaris/pbs/nccl_probe.pbs` (`nccl_probe.py`).
- UMA isolation: `polaris/pbs/uma8_isolate.pbs` (`ase_ckpt_ref.py --ckptflag`).
- N=21 speedup attempt: `polaris/pbs/n21_8gpu_speedup.pbs`.
