# CPU-memory offload plan for UMA (capacity / memory-bound)

**Goal:** run larger boxes on fixed GPU memory (A100-40GB) by spilling model
state to host RAM (Polaris: 512 GB/node). Speed is secondary; correctness (E/F
parity) is mandatory.

## Memory profile (measured, single A100, NaCl)

| N atoms | ckpt OFF peak | ckpt ON peak | notes |
|--------:|--------------:|-------------:|-------|
| 1728 | 30.6 GiB | 10.1 GiB | activations ≈ 20 GiB (66%) |
| 4096 | OOM (37.5) | 21.0 GiB | 4096 fits 1 GPU w/ eager ckpt |
| 21952 | — | 33.1 GiB | eager-ckpt ceiling (1 GPU) |

- **Activations dominate** (~66%, scale with N) → primary offload target.
- Edges O(50·N) second; weights fixed 2.33 GB (traced .pt).
- **Eager activation checkpointing** = 12.7× atoms (1728→21952), bit-identical
  E/F, 1.33× step cost — but `torch.jit.trace` CANNOT serialize
  `torch.utils.checkpoint` (`_NoopSaveInputs`), so the traced LAMMPS artifact is
  stuck at 1728. Offload must therefore either (a) be allocator-level and
  transparent to the traced model, or (b) run the eager model.

## Ladder (capacity per unit effort) — execute in order

### A1 — Managed-memory oversubscription (allocator-level)  [effort XS–S]
Let allocations spill to host via CUDA Unified/Managed memory, transparent to the
traced model. Implemented via `torch.cuda.CUDAPluggableAllocator` + a
`cudaMallocManaged` shared lib (`managed_alloc.cpp`).
- Exact ✅ | traced-model OK ✅ | speed: **catastrophic thrash**.
- **Status:** DONE — REJECTED for MD throughput.
- **Result (N=1728, traced autograd):**
  - native allocator: dev 37.8 GiB, **772 ms/step**
  - managed off-prefer: dev **2.8 GiB**, **14,223 ms** (18.4×)
  - managed prefer-device: dev 2.8 GiB, **7,578 ms** (9.8×)
- **Verdict:** managed memory DOES spill (2.8 GiB device for a job that needs
  37.8) so capacity is essentially unbounded, but it makes ALL tensors pageable
  → 10–18× step penalty on A100/PCIe-gen4 (no Grace C2C). Usable only as a
  last-resort "run at any cost"; not viable for production MD. `MANAGED_PREFER_
  DEVICE=1` (cudaMemAdvise) roughly halves the penalty but not enough. A
  smarter variant (advise only cold tensors to host, keep hot on GPU) is the D1
  refinement, but A3 is the better lever — deprioritise A1 tuning.

### A3 — Eager model + `torch.autograd.graph.save_on_cpu(pin_memory=True)`  [effort S–M]
Run eager FairChem model with automatic activation offload; STACKS with eager
activation checkpointing.
- Exact ✅ | traced-model OK ❌ (eager) | speed: moderate (pinned + overlap).
- **Status:** DONE — BEST single-GPU capacity lever.
- **Result (single A100-40GB, NaCl):**
  - A3(i) save_on_cpu only: max **8,000** atoms (4.6× vs 1728); peak@1728
    30.6→**5.67 GiB** (5.4× less).
  - A3(ii) save_on_cpu + checkpointing (stacked): max **32,768** atoms
    (**19×** vs 1728); 32768 fits at 26.9 GiB, OOM at 64000 (34.5 GiB).
- **Verdict:** the winning capacity path. Exact eager execution, ~19x atoms on
  one GPU when stacked with checkpointing, and (unlike A1) keeps activations on
  the GPU except the offloaded saved tensors → far better throughput than
  managed memory. Next: measure ms/step and wire the eager+offload model into
  the engine's Python-worker path (UMA_PYTHON_GP_WORKER) so LAMMPS can use it.
  Stacks further with graph-parallel across GPUs/nodes for aggregate capacity.

### C1 — Edge chunking (`edge_chunk_size`) + host-resident edges  [effort S–M]
Stream O(50·N) edge blocks from host instead of materializing all on GPU.
Complements A1/A3. `edge_chunk_size` is a valid InferenceSettings field
(currently None).
- Test: eager sweep with edge_chunk=65536, alone and + checkpointing.
- Exact ✅ | traced-model: partial | **Status:** DONE — no benefit here.
- **Result:** ckpt_off+ec: max **1728** (= baseline, no gain); ckpt_on+ec: max
  **13824** (WORSE than ckpt-alone 21952). edge_chunk=65536 exceeded the edge
  count at these sizes (nothing to chunk) and added overhead. Edge memory is not
  the binding constraint — activations are. **Deprioritise C1.**

### A2 — Per-block C++ activation offload (pinned host, double-buffered)  [effort L]
Trace 4 blocks separately; engine stages each block's activation to pinned host
in forward, prefetches in backward. Pure C++/no-Python, controlled, highest
capacity. Only if A1/A3/C1 insufficient.
- **Status:** NOT STARTED (gated on A1–C1 results)

### D2 — Pinned host pool  [effort XS, multiplier]
Pre-pin a large host buffer as the offload target to maximize H2D/D2H BW.
Enabler for A2/A3.
- **Status:** NOT STARTED

## Related docs
- `activation_checkpointing.md` — the checkpointing method in detail (12.7× on
  its own; the base lever that offload stacks on).

## Enablers already landed
- `--activation-checkpointing` flag in export_artifact.py (eager only; does not
  trace).
- `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` in env (no compute-node stalls).

## Results log
(updated as tests complete)

Capacity (max atoms, 1 A100-40GB):
| option | max atoms | vs base | verdict |
|--------|----------:|--------:|---------|
| baseline (eager) | 1728 | 1.0× | reference |
| checkpointing | 21952 | 12.7× | exact |
| A1 managed mem | unbounded | — | REJECTED (10–18× slow) |
| A3(i) save_on_cpu | 8000 | 4.6× | exact |
| **A3(ii) soc+ckpt** | **32768** | **19×** | **BEST 1-GPU** |
| C1 edge_chunk (+ckpt) | 13824 | 8× | no benefit (activations bind, not edges) |

Throughput + parity (N=1728, ms/step, all bit-exact ΔE=0):
| config | ms/step | vs base | mem saving |
|--------|--------:|--------:|-----------:|
| baseline | 418.6 | 1.00× | 1.0× |
| checkpointing | 557.1 | 1.33× | 3.0× |
| save_on_cpu alone | 2881.5 | 6.88× | 3.9× |
| **soc + ckpt** | 725.8 | **1.73×** | 2.2× |

**Key synergy:** save_on_cpu ALONE is costly (6.88×) because it offloads every
saved activation. Stacked with checkpointing it is only 1.73× — checkpointing
drops most saved tensors (recompute), so there is far less to move over PCIe.
=> Always pair offload with checkpointing.

## Conclusion so far
- **A1 rejected**: managed memory = unlimited capacity but 10–18× step penalty
  (all tensors pageable, PCIe-gen4, no C2C). Last resort only.
- **A3(ii) soc+ckpt is the winner on one GPU**: **19× atoms (1728→32768)** for
  only **1.73× step time**, bit-exact. Always pair offload WITH checkpointing
  (alone save_on_cpu is 6.88×).
- **Stacking**: A3(ii) × graph-parallel(8 GPU/2 node) ≈ 32768 × ~8 ≈ 260k+ atoms
  aggregate — memory-bound goal well served.
- **Blocker for LAMMPS product path**: A3 is eager-only (checkpointing +
  save_on_cpu are eager autograd features; neither survives torch.jit.trace).
  The C++ engine loads a traced .pt, so to use A3 in LAMMPS we must run the eager
  model via the engine's Python-worker path (UMA_PYTHON_GP_WORKER) OR implement
  per-block C++ recompute/offload (A2).
- Next: (1) C1 edge chunking (job 7434451, queued — update when it lands);
  (2) engine wiring of the eager+offload worker (the path to make A3 usable by
  LAMMPS, since A3 is eager-only).

## Recommendation
For the memory-bound goal, the production recipe is:
1. **Eager execution** (not traced) via the engine Python-worker path, with
2. **activation_checkpointing=True** + **save_on_cpu(pin_memory=True)** →
   19× atoms/GPU at 1.73× step cost, bit-exact.
3. **Graph-parallel across GPUs/nodes** on top for aggregate capacity
   (~260k+ atoms on 8 GPU/2 node), accepting the inter-node chattiness as a
   capacity (not speed) cost.
4. Optionally **edge chunking** (C1) as a further trim if it helps.
Avoid A1 managed memory except as a last-resort "run at any cost".
