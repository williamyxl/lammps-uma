# Agent 2 Review and Suggestions: LibTorch UMA on Aurora XPU

## Executive assessment

The native LibTorch/XPU/XCCL implementation is a credible proof of concept. Existing code and archived results support FP64 energy and autograd-force inference on XPU, native oneCCL graph parallelism, and numerically accurate 12-tile execution at N=16.

The current review document overstates how close the implementation is to the N=32 graph-parallel target. The immediate memory diagnosis is inconsistent with the current source, block/chunk activation checkpointing is not yet integrated into the graph-parallel runtime, and replicated quadratic CPU neighbor construction may prevent target-scale execution even after device-memory issues are resolved.

## Corrections to the current review

### Confirmed traced capacity is at least N=13

The stated N=8 single-tile ceiling is stale. A later per-block/per-chunk run completed N=13 successfully:

- `scripts/p6h_j1N.o8777884:5`
- `scripts/out/phase6_h_jsweep/run_n13.log:1`

The exact ceiling above N=13 remains unestablished in the inspected outputs.

### Block-level checkpointing is already implemented

The review lists making `uma_ckpt::block` a real checkpoint as a remaining fix. Current code already dispatches the operator through `BlockCheckpointFn`, executes its forward under `NoGradGuard`, and recomputes during backward:

- `src/block_context.cpp:112`
- `include/uma/block_context.h:131`

Adding another block-level wrapper will not address the current N=18 failure.

### Full-edge work remains outside checkpointed blocks

The chunk modules recompute chunk-local Wigner and edge features, but the top-level exported module still computes full-edge Wigner tensors, envelopes, `x_edge`, and edge-degree embeddings before entering the block loop:

- `python/export_blocks_xpu.py:445`
- `python/export_blocks_xpu.py:479`

The top graph runs with autograd enabled, so these operations can retain large backward state. Therefore, the claim that the full-edge Wigner is never retained is true only inside the rewritten block/chunk path, not for the complete model.

### Numerical agreement should not be called bit-exact when differences are nonzero

Results such as `dE=4.6e-8 eV` and nonzero force differences are excellent FP64 agreement, but they are not bitwise equality. Suggested terminology is:

- “numerically equivalent within FP64 tolerances” for nonzero differences;
- “bit-exact” only when all compared values have identical representations or exactly zero differences.

### Force comparisons are sampled in some large tests

Gate 1 and the N=16 report compare 100 sampled atoms rather than every atom:

- `scripts/phase6_gate1_compare.py:54`
- `scripts/p6_mpar.o8777376:5`

The review should distinguish exhaustive force parity from sampled force parity.

## Feasibility assessment

| Objective | Assessment |
|---|---|
| Pure C++ FP64 energy and autograd forces on XPU | Demonstrated and feasible |
| Native XCCL graph parallelism | Demonstrated through 12-tile N=16 |
| N=18 single-tile traced execution | Likely feasible after correcting the full-edge prologue memory path |
| Nested block/chunk recomputation | Feasible and already operational on the single-tile path |
| Multi-chunk graph-parallel checkpointing | Technically feasible but not currently integrated or validated |
| N=32 on 12 tiles | Plausible research target, but requires substantial additional implementation |
| General production LAMMPS backend | Not yet: no virial/stress, N-specific artifacts, fixed execution assumptions, and environment-specific build integration |

## Primary technical risks

### 1. The N=18 allocation likely originates outside the chunk path

The N=18 failure requests a 12.82 GiB allocation from `IndexAddBackward0`:

- `scripts/out/phase6_h_j3/n18.log:4`

This is consistent with an uncheckpointed full-edge scatter or index backward, especially the top-level edge-degree embedding. It is less consistent with the review's claim that the missing block checkpoint is the primary cause, because block checkpointing is already active.

Weight deduplication may recover several GiB of resident memory, but it will not necessarily eliminate a single 12.82 GiB transient allocation.

### 2. Block/chunk activation checkpointing is not connected to XCCL graph parallelism

The graph-parallel runtime loads rank-specific `model_mp_w...pt` shards and uses whole-module checkpointing:

- `src/mpi_peer_predictor.cpp:214`
- `src/mpi_peer_predictor.cpp:308`

It does not load the block or chunk modules. Conversely, the block/chunk exporter assumes non-graph-parallel execution in several places, including full-node state and fixed node offsets:

- `python/export_blocks_xpu.py:250`
- `python/export_blocks_xpu.py:636`

A combined graph-parallel plus activation-checkpoint artifact and runtime path remains to be designed and implemented.

### 3. CPU neighbor construction may prevent N=32 execution

Every graph-parallel rank gathers the complete system and independently builds the full neighbor graph:

- `src/mpi_peer_predictor.cpp:262`
- `src/neighbor_list.cpp:154`

The current neighbor construction is effectively quadratic in atom count. For N=32 NaCl, the system contains 262,144 atoms, making replicated all-pairs candidate generation potentially prohibitive. This should be treated as a target-capacity blocker rather than a later performance optimization.

### 4. Validation scripts can fail open

The AG=FD script ignores subprocess failures and can report success when no finite-difference samples complete because the maximum error remains initialized to zero:

- `scripts/phase6_agfd.py:81`
- `scripts/phase6_agfd.py:87`

The Gate 1 comparator can also retain a successful ASE status if the oracle comparison throws:

- `scripts/phase6_gate1_compare.py:73`

These paths should fail immediately on command failure, missing output, NaN/Inf values, malformed force dumps, oracle errors, or zero completed samples.

### 5. N-specific artifacts lack a strict runtime contract

N-specific exports are acceptable for controlled capacity studies, but the runtime should reject mismatches before inference. Required metadata should include:

- artifact-format version;
- model and checkpoint hash;
- Torch and FairChem versions;
- backend and precision;
- task, charge, and spin policy;
- world size and rank;
- traced atom count;
- partition convention;
- chunk size and expected chunk count;
- supported shape constraints.

### 6. XCCL synchronization and communicator assumptions need hardening

The oneCCL barrier call does not wait for completion, while all-reduce and all-gather do:

- `src/xccl_peer.cpp:82`
- `src/xccl_peer.cpp:112`

KVS bootstrap also uses `MPI_COMM_WORLD` rather than the communicator supplied by LAMMPS:

- `src/xccl_peer.cpp:60`

The implementation should wait on the barrier event, use the active LAMMPS communicator, and resolve the observed MPI thread-level warning before relying on more complex nested recomputation and collective sequences.

## Recommended implementation plan

### Priority 0: establish the real memory source

Instrument synchronized XPU memory measurements at these boundaries:

1. before top-level forward;
2. after graph-distance generation;
3. after full-edge Wigner preparation;
4. after initial edge-degree embedding;
5. before and after every block;
6. before `torch::autograd::grad`;
7. during block recomputation;
8. during chunk recomputation.

Record allocated, reserved, and peak memory. Run each configuration in a fresh process and vary the edge chunk size. If the 12.82 GiB transient remains constant across chunk sizes, that strongly confirms that it lies outside chunk-local work.

### Priority 0: checkpoint or chunk the top-level edge prologue

Recommended approaches, in order:

1. Implement a checkpointed and chunked custom operation for the initial edge-degree embedding.
2. Recompute Wigner and edge embeddings per chunk and accumulate the initial node embedding.
3. Place the complete pre-block edge operation behind a dedicated C++ checkpoint function.
4. Move remaining full-edge radial preparation into block/chunk modules where practical.

The intended memory invariant should be explicit: no edge-sized tensor larger than one configured chunk may be retained for backward.

### Priority 0: define a combined GP+AC architecture

Treat this as a separate milestone. It requires:

- rank-specific top-level artifacts containing `uma_peer` operations;
- block modules that preserve full-node gather semantics;
- chunk modules with correct rank-local edge partitions and node offsets;
- block/chunk loading in `MpiPeerPredictor`;
- deterministic collective ordering during backward recomputation;
- metadata keyed by model, world size, rank, atom count, and chunk configuration.

The N=32 target should not be described as close until this combined path runs a multi-chunk correctness gate.

### Priority 0: replace quadratic replicated neighbor construction

Preferred solution: consume LAMMPS neighbor data and convert it into FairChem edge orientation and periodic offsets. A secondary option is an O(N·neighbors) cell-list implementation. For graph-parallel runs, build once and distribute edge shards where practical rather than rebuilding the complete graph on all ranks.

### Priority 1: make correctness tests fail closed

Every test must fail on:

- nonzero subprocess return code;
- missing output or force rows;
- duplicate or missing atom IDs;
- NaN or infinite energy/force values;
- oracle exceptions;
- zero completed finite-difference samples;
- mismatched expected atom counts.

Emit machine-readable JSON containing the number of attempted and completed comparisons and all tolerance decisions.

### Priority 1: measure and deduplicate weights

Before redesigning serialization, enumerate parameter names, shapes, storage pointers, and unique storage bytes across the top, block, and chunk modules. Then evaluate either:

1. post-load storage aliasing with strict name and shape validation; or
2. a single owning TorchScript module exposing shared block/chunk methods.

The second approach is architecturally cleaner. Any aliasing prototype must prove that device transfer, repeated backward calls, destruction, and reload do not break shared storage.

### Priority 1: add a multi-chunk graph-parallel correctness gate

Use a size at which every rank executes multiple chunks. Validate:

- graph-parallel versus single-tile/eager energy and forces;
- autograd versus finite differences;
- all-rank collective call counts and ordering;
- all atoms for at least one moderate system;
- several consecutive MD steps to detect allocator or synchronization instability.

### Priority 2: define production scope accurately

Until additional functionality is implemented, document the supported scope as:

- one atomic system;
- OMAT task;
- fixed charge 0 and spin 0 for block artifacts;
- FP64;
- energy and forces only;
- no virial or stress;
- N-specific GP/AC artifacts;
- one MPI rank per XPU tile;
- Aurora-specific software environment.

## Numerical validation recommendations

Use combined absolute and relative criteria rather than only total-energy absolute tolerance:

- total energy absolute error;
- energy error per atom;
- maximum, RMS, and relative force error;
- force cosine similarity;
- force-sum or momentum-conservation check;
- NaN/Inf checks;
- AG=FD over multiple displacement sizes.

The current `1e-6 eV` energy and `1e-5 eV/Å` force gates are reasonable regression bounds for the tested systems, but total-energy tolerance should scale with atom count. Observed errors are substantially smaller, so tighter gates may be adopted after testing more structures, compositions, cells, and near-cutoff configurations.

## Suggested milestone gates

### Gate A: single-tile memory closure

- N=13 baseline reproduced.
- N=18 completes in a fresh process.
- Full-system energy and force parity passes.
- Memory attribution confirms no unexpected full-edge backward allocation.

### Gate B: combined GP+AC correctness

- Two-tile, multi-chunk graph-parallel execution passes.
- Graph-parallel results agree with single-tile/eager results.
- AG=FD completes with no skipped samples.
- Multiple consecutive LAMMPS steps complete.

### Gate C: 12-tile capacity progression

- Reproduce N=16.
- Validate N=18 and N=20 before larger sweeps.
- Record host neighbor time, graph transfer, forward, backward, and collective time separately.
- Attempt N=32 only after neighbor construction and memory scaling are demonstrated to be viable.

### Gate D: performance justification

Compare traced LibTorch, eager worker, and FairChem/ASE paths at matched precision and system sizes. Report:

- end-to-end LAMMPS pair time;
- CPU neighbor-list time;
- host-to-device transfer time;
- forward and backward time;
- collective time;
- checkpoint recomputation overhead;
- peak allocated and reserved memory.

## Final recommendation

The N=16 native-XCCL result should be retained as the main validated achievement. N=18 single-tile is likely achievable after correcting the uncheckpointed full-edge prologue and reducing resident weight duplication. N=32 on 12 tiles remains feasible in principle, but should be presented as a substantial follow-on milestone requiring combined graph-parallel checkpoint integration, scalable neighbor construction, strict artifact contracts, and fail-closed validation.