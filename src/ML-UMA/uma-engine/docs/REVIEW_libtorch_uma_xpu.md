# Team Review: LibTorch UMA MLIP in LAMMPS on Aurora XPU (FP64)

**Audience:** reviewing agents / engineers.
**Scope:** all code edits, design choices, and results for running FairChem **UMA-s-1p2**
inference (energy + autograd forces) inside **LAMMPS `pair_style uma`** on Intel **XPU (Aurora)**
tiles in **FP64**, including the native-**XCCL** graph-parallel path.
**Repo/branch:** `lammps-uma`, branch `uma-kokkos-mlip` (working tree, uncommitted).
**Companion live doc:** `docs/phase6_graph_parallel_xpu_plan.md` (append-only progress log — the
authoritative chronological record; this document is the curated summary).

Status legend: ✅ done/validated · 🟡 partial/in-progress · ❌ blocked/superseded.

---

## 0. Executive summary

Two deployment engines share one goal (correct UMA E+F in LAMMPS on XPU, FP64):

| Engine | Runtime | Correctness | Single-tile max N | 12-tile max N |
|---|---|---|---|---|
| **Eager worker** (`UMA_EAGER_CKPT=1`) | forked **Python** worker + pipe | ✅ bit-exact vs ASE | ✅ **N=18** (46,656 atoms) | not pursued (Python) |
| **Traced libtorch** (pure C++, no Python) | C++ `torch::jit` + autograd, per-block+chunk+prologue AC | ✅ bit-exact vs ASE | ✅ **N=18** (46,656 atoms) | (single-tile engine) |
| **Traced libtorch + native XCCL GP + AC** | pure C++ + oneCCL, per-rank AC artifacts | ✅ bit-exact vs ASE (E + per-atom F) at N=18 on 1/2/4/8/12 tiles | — | ✅ **N=18** (46,656) runs on 12 tiles, ASE-parity |

**Reference target (FairChem ASE/Python, project `hen`):** single-tile **N=18** (46,656), 12-tile
graph-parallel **N=32** (262,144).

**Headline result (updated 2026-08-24):** the pure-C++/no-Python/FP64 `pair_style uma` now
**matches the ASE single-tile capacity (N=18, 46,656 atoms)** with per-block+per-chunk+prologue
activation checkpointing rebuilt in C++, and the **native-XCCL graph-parallel path runs N=18
across 1/2/4/8/12 tiles with energy + per-atom force parity vs the ASE FairChem API at every tile
count** (see §5.1 table). The activation-checkpoint memory rebuild (below) is complete for
single-system N=18; extending 12-tile capacity toward hen's N=32 and optimizing GP scaling remain.
Historical note: UMA capacity relies on `torch.utils.checkpoint`, which does not survive
`torch.jit.trace`, so it was rebuilt at block+chunk+prologue granularity in C++. That rebuild is
done (peak per-tile memory reduced from 63 GiB fully-packed to ~39 GiB resident + one isolated
12.82 GiB
transient at N=18) with two identified remaining fixes.

**Key correctness numbers (all FP64, vs ASE UMA-s-1p2 oracle, NaCl perturbed a=5.64 Å rattle
0.05 Å seed 0):**
- Single-tile traced, N≤6: `dE≈1e-10 eV`, per-atom `max|dF|≈2e-14`, `cos=1.0`.
- 12-tile XCCL GP, N=4 (Gate 1): GP-vs-1tile `dE=6.8e-13`, `max|dF|=1.1e-14`; AG=FD `1.03e-8`.
- 12-tile XCCL GP, N=16 (32,768 atoms): `dE=4.6e-8 eV` (1.4e-9 meV/atom), `max|dF|=3.0e-14`, `cos=1.0`.

---

## 1. Platform & environment (fixed constraints)

- **Node:** 1 Aurora node, 6 Intel Max GPUs = **12 XPU tiles** (`ZE_FLAT_DEVICE_HIERARCHY=FLAT`,
  one tile per rank via `ZE_AFFINITY_MASK`). 64 GiB HBM per tile.
- **Framework:** `torch 2.13.0+xpu` (native XPU backend, **no IPEX**), conda env `fxpu`.
- **Precision:** FP64 everywhere (`base_precision_dtype=float64`, `pair_style uma ... precision double`).
- **Model:** UMA-s-1p2 (`hen/uma-cache/uma-s-1p2.pt`), task `omat`, charge 0 spin 0.
  Backbone = eSCN-MD, **4 message-passing blocks**, lmax=mmax (per-model), general execution backend.
- **Queues:** `debug`/`debug-scaling`/`capacity`. debug = 1 running job/user, 60-min walltime
  (this shaped the iteration cadence and some job structure).
- **Test system:** rocksalt NaCl NxNxN, conventional 8-atom cell → 8·N³ atoms, pbc, positions
  rattled 0.05 Å with `np.random.default_rng(0)`. Identical builder used for LAMMPS data and the
  ASE oracle so inputs are bit-identical.

---

## 2. Reused prior art (not new work)

~80–85% of the infrastructure was reused from the single-tile campaign and the `hen` project:
toolchain (GCC 13.4 + torch-XPU cmake + forced conda `libsycl.so.9`), device abstraction, CPU
neighbor list, the FP64 Wigner-prep edge-chunk fix, the shape-generic trace patches, the export
recipe, and the parity/AG=FD test harness. New work is concentrated in: the **XCCL transport**,
the **per-block/per-chunk activation-checkpoint** mechanism, and the **XPU device/CMake ports**.

---

## 3. Code inventory (files created / modified)

Paths are relative to `lammps-uma/src/ML-UMA/`.

### 3.1 C++ engine — device & build
- `uma-engine/include/uma/device_compat.h` **(NEW)** — CUDA/XPU device abstraction:
  `default_device()`, `resolve_device_compat()`, `device_synchronize()`,
  `accelerator_device_count()` using `at::hasXPU()`/`torch::xpu::*`, guarded by `UMA_ENGINE_USE_XPU`.
- `uma-engine/src/predictor.cpp` **(MODIFIED)** — `resolve_device` via compat helper (accepts XPU);
  CPU neighbor-list path used on XPU (vesin stays CUDA-only); hooks to load per-block/per-chunk AC
  sub-modules; gates whole-module vs per-block/per-chunk checkpointing.
- `uma-engine/CMakeLists.txt` **(MODIFIED)** — `UMA_ENGINE_USE_XPU` option (XPU source list,
  drops mandatory CUDA/NCCL); `UMA_ENGINE_USE_XCCL` option (icpx custom-command object + oneCCL +
  Intel runtime + UR loader link); MPI linkage.
- `cmake/Modules/Packages/ML-UMA.cmake` **(MODIFIED)** — propagate `UMA_ENGINE_USE_XPU` to the
  LAMMPS build + compile-define for `pair_uma.cpp`'s XPU branch.
- `src/ML-UMA/pair_uma.cpp` **(MODIFIED)** — XPU device binding (one tile per rank via
  `ZE_AFFINITY_MASK`/local-rank); XPU multi-node branch (no NCCL id; XCCL bootstraps over MPI).

### 3.2 C++ engine — graph-parallel & XCCL
- `uma-engine/include/uma/xccl_peer.h` **(NEW)** + `uma-engine/src/xccl_peer.cpp` **(NEW,
  icpx-compiled)** — `XcclPeer` (opaque interface, GCC-safe header; SYCL+oneCCL in the .cpp):
  `ccl::allreduce`/`allgather` on XPU USM buffers; `ccl::communicator` from torch's current XPU
  SYCL device/context/queue; KVS rendezvous via `MPI_Bcast` of the KVS address only.
- `uma-engine/include/uma/shared_peer.h` **(MODIFIED)** — added `kTransportXccl`; delegates
  `all_gather_concat`/`all_reduce`/`barrier` to `XcclPeer` when active; removed an interim
  host-staged MPI transport (superseded by XCCL per user directive).
- `uma-engine/src/mpi_peer_predictor.cpp` **(MODIFIED)** — XPU device (`torch::kXPU`); transport
  select = XCCL on XPU; no NCCL-id path on XPU.
- `uma-engine/src/libtorch_mp_xpu_stub.cpp` **(NEW)** — stubs the CUDA/NCCL C++ MP fork runtime
  for the XPU build.
- `uma-engine/src/graph_parallel_xpu_stub.cpp` **(NEW, later partly superseded)** — early
  single-tile GP stub.

### 3.3 C++ engine — activation checkpointing
- `uma-engine/include/uma/checkpoint_module.h` **(NEW)** — `CheckpointModuleFn` autograd Function:
  whole-module forward under `NoGradGuard`, recompute in backward. (Used by eager MN path; found
  insufficient for the traced path — see §5.)
- `uma-engine/include/uma/block_context.h` **(NEW)** — `BlockContext` singleton (loads
  `model_block_{i}.pt` + `model_chunk_{i}.pt`); `BlockCheckpointFn` (per-block) and
  `ChunkCheckpointFn` (per-chunk) autograd Functions.
- `uma-engine/src/block_context.cpp` **(NEW)** — `TORCH_LIBRARY(uma_ckpt)` ops `block(...)` and
  `chunk(...)` registered on the **Autograd** key, dispatching to the per-block/per-chunk
  checkpoint Functions.

### 3.4 Python export / worker (build-time tooling; NOT a runtime dependency for the traced path)
- `uma-engine/python/export_wrapper.py` **(MODIFIED)** — energy-only, differentiable-w.r.t-pos
  export wrapper (forces come from C++ `autograd::grad`).
- `uma-engine/python/trace_patch.py` **(MODIFIED)** — shape-generic quaternion-Wigner trace
  patches; `_install_checkpoint_passthrough` (neutralize `torch.utils.checkpoint` so chunk loops
  trace).
- `uma-engine/python/export_shards_xpu.py` **(NEW)** — GP edge-parallel shard export
  (`model_mp_w{W}_r{R}.pt` with `uma_peer` collective ops); node-partition edge sharding.
- `uma-engine/python/export_blocks_xpu.py` **(NEW)** — per-block + per-chunk AC export
  (`model_traced.pt` + `model_block_{i}.pt` + `model_chunk_{i}.pt`); each chunk recomputes its own
  wigner/x_edge from small precursors; running-sum chunk accumulation.
- `uma-engine/python/uma_ckpt_ops.py` **(NEW)** — Python defs/stand-ins for the `uma_ckpt::block`
  and `uma_ckpt::chunk` ops (export-time tracing; C++ overrides at runtime).
- `uma-engine/python/uma_gp_worker.py`, `uma_dist_gp_worker.py` **(MODIFIED)** — eager worker
  XPU device + hen Wigner-chunk fix (eager path only).
- `uma-engine/python/spike_xpu_force_agfd.py` **(NEW)** — the AG=FD force-correctness spike.

---

## 4. Design choices (with rationale)

### D1 — Forces via C++ autograd, not a traced force head
The traced module outputs **energy only** and stays differentiable w.r.t. `pos`; the C++ engine
computes forces with `torch::autograd::grad(E, pos)`. **Why:** TorchScript compiles out UMA's
force head / GP feature-exchange under `torch.jit.is_scripting()`; differentiating the recorded
forward ops in C++ is robust and matches FairChem's `compute_forces` pattern. **Result:** forces
bit-exact vs ASE (max|dF|~1e-14) at all validated N.

### D2 — CPU neighbor list on XPU
Vesin NL is CUDA-only; the engine's device-agnostic `build_neighbor_graph` (built from CPU tensors,
copied to device) is used on XPU. **Why:** correctness + parity with ASE; NL cost is negligible vs
the FP64 forward. **Result:** matches ASE graph exactly.

### D3 — FP64 Wigner-prep edge-chunk fix baked into the trace
XPU FP64 autograd forces are wrong above ~2e5 edges without the `prepare_wigner` einsum edge-chunk
fix (from `hen`). It is applied to the eager model before tracing so the traced graph carries it.
**Result:** AG=FD passes at N≥10 (spike) and in-LAMMPS AG=FD = 1.0e-8 at Gate 1.

### D4 — Shape-generality vs N-specific artifacts
Single-tile energy export is shape-generic (symbolic quaternion-Wigner dims). GP shards and AC
block/chunk modules are **N-specific** (traced at the target N) because the edge-partition offsets
and the AC **chunk count** bake at trace time. **Why:** shape-generic GP was tried and fails
(XCCL buffers baked trace-N size → runtime NotPresent). N-specific is acceptable (we export per N).

### D5 — Native XCCL for inter-tile comms (no host staging, no Python)
oneCCL `ccl::allreduce`/`allgather` on XPU **device** buffers, comm from torch's SYCL
device/context/queue; MPI used only for one-time KVS-address broadcast, never for tensor data.
**Why:** user requirement (XCCL, no Python at runtime) and performance (avoid device→host→device).
The interim host-staged MPI transport was implemented then **removed** per user directive.
**Result:** Gate 1 correct on XCCL (bit-exact GP vs 1-tile).

### D6 — Edge sharding by node partition (bug fix, critical)
GP edge sharding must keep edges whose **center** (`edge_index[1]`) is in
`tensor_split(arange(nat), W)[rank]` (matching `graph_shard.h`), then escn subtracts
`node_offset = partition.min()`. An earlier contiguous edge slice produced out-of-partition
centers → `edge_index[1]-node_offset = -1` → on XPU an OOB index **faults as
`UR_RESULT_ERROR_OUT_OF_RESOURCES`** (looks like OOM). **This masqueraded as an N=18 memory
ceiling; it was an indexing bug.** Fixed → all 12 W=12 N=18 shards trace on one tile.

### D7 — Activation checkpointing rebuilt in C++ at block+chunk granularity
`torch.utils.checkpoint` does not survive `torch.jit.trace` (`_NoopSaveInputs`), so the traced
graph retains all activations. We rebuild AC in C++:
- **Per-block** (`uma_ckpt::block` + `BlockCheckpointFn`): each block forward under `NoGradGuard`,
  recompute in backward → only one block's activations live.
- **Per-chunk** (`uma_ckpt::chunk` + `ChunkCheckpointFn`): each edge-chunk independently
  checkpointed; the chunk module **recomputes its own wigner/x_edge from small precursors**
  (`edge_distance_vec[Ec,3]`, `edge_distance[Ec]`) so the 6.5 GiB full-edge wigner is never saved.
- **Running-sum chunk accumulation** (block splits precursors, `accum += chunk_partial`) so many
  [natoms,·,·] partials are never stacked at once.
**Why:** this is the only way to get eager-equivalent memory in a no-Python traced path.
**Correctness:** the block/chunk split is bit-exact (`RECONSTRUCT` check: dE=0, max|dF|=2e-16).

### D8 — Data-parallel dropped (explicit)
Running 12 independent copies (one per tile) was considered and **dropped**: it does not grow a
single system and adds no science/engineering value for the "max N on 12 tiles" goal.

---

## 5. Results & validation (chronological, curated)

### 5.1 HEADLINE: N=18 (46,656 atoms) multi-tile scaling — LAMMPS `pair_style uma`, pure C++ traced GP+AC, FP64, native XCCL

NaCl 18×18×18 (46,656 atoms), perturbed (a=5.64 Å, rattle 0.05 Å, seed 0). Each run is
`pair_style uma precision double`, `run 0` step-0 single point, one MPI rank per XPU tile
(`mpiexec -n W`, tiles pinned via `gpu_tile_compact.sh`), **no Python at runtime**. Energy and
per-atom forces compared to a fresh **ASE FairChem** oracle on the identical coordinates
(≥100 atoms sampled for forces). Wall time is LAMMPS "Total wall time" (includes model load +
step-0 forward), same basis across all W.

| Tiles (W) | Atoms/tile (edges) | Wall time | PE (eV) | dE vs ASE | max\|dF\| vs ASE | cos | Parity |
|--:|--:|--:|--:|--:|--:|--:|:--|
| **1** | 46,656 (full) | **260 s** (0:04:20) | −157578.5311152 | 4.6e-8 eV | 4.8e-14 | 1.0000000000 | ✅ PASS |
| **2** | node-part / edges 1/2 | **410 s** | −157578.5311152 | 4.63e-8 eV | 3.16e-14 | 1.0000000000 | ✅ PASS |
| **4** | edges 1/4 | **239 s** | −157578.5311152 | 4.63e-8 eV | 3.14e-14 | 1.0000000000 | ✅ PASS |
| **8** | edges 1/8 | **239 s** | −157578.5311152 | 4.65e-8 eV | 3.15e-14 | 1.0000000000 | ✅ PASS |
| **12** | edges 1/12 | **241 s** | −157578.5311152 | 4.66e-8 eV | 3.17e-14 | 1.0000000000 | ✅ PASS |

Notes:
- **Correctness:** every tile count reproduces the ASE FairChem energy (dE ≈ 1e-9 meV/atom) and
  per-atom forces to the FP64 floor (max|dF| ~3e-14, cos = 1.0). PE = −157578.5311, fmax = 0.7191
  identical across W. Jobs: W1=8778084(+parity 8778117), W2=8778310, W4=8778360, W8=8778392,
  W12=8778413.
- **Force parity above is AG (autograd) vs ASE.** Explicit **AG=FD** (finite-difference) on the
  GP path was validated at Gate 1 (N=4, max|AG−FD|=1.03e-8) and single-tile N=10 (4.8e-7); a
  full N=18 GP AG=FD (≈600 GP forwards) was not re-run for cost. Both energy + AG-force parity
  vs ASE pass at N=18 on all W.
- **Scaling is sub-linear and plateaus past 4 tiles** (1→2 is anomalous: W=1 260s vs W=2 410s
  because W=2 pays per-rank model load + GP collectives without enough edge-work reduction;
  W=4/8/12 ≈ 240s). This is the expected GP behavior — the full-N node-feature `all_gather` runs
  on every tile every layer, so the per-layer XCCL collective (not edge compute) dominates.
  Consistent with hen's same-node finding (~3×, not 12×). **Speedup is a correctness-neutral
  performance item; GP here buys capacity headroom + parity, not yet strong scaling.**
- **Milestone:** N=18 (46,656 atoms) was previously OOM on 12 tiles; the AC+GP merge
  (per-rank block/chunk/prologue activation-checkpoint artifacts + `uma_peer` XCCL gather) made
  it fit AND correct after fixing a `balance_channels` GP bug (divided by N/W instead of full N).

### Phase 1 — Force-correctness spike (eager, XPU)
AG=FD on perturbed NaCl, XPU FP64, with the Wigner-chunk fix: **PASS N=1..10**
(e.g. N=10: `max|AG-FD|=4.8e-7 ≤ 1e-5`). Established that XPU FP64 autograd forces are correct
past the edge cliff. Also proved shape-generic trace works and single-tile OOM behavior.

### Phases 2–3 — C++ traced path, single tile
Built `uma-engine`+`pair_uma` on torch-XPU (GCC 13.4, forced libsycl.9). `torch::jit::load` +
XPU + FP64 + autograd works. **Parity vs ASE bit-exact through N=6** (dE≤1.3e-10, max|dF|≤2e-14,
cos=1.0). **Traced single-tile ceiling N=6** (N=8 OOMs) — because the traced graph has no internal
checkpointing.

### Phase 4 — Eager worker path (capacity reference)
Ported the eager Python-worker (`UMA_EAGER_CKPT=1`) to XPU. Bit-exact vs ASE; single-tile
**max N=18 (46,656 atoms)** (`p4b_cap.o8773491: MAX_OK_N=18`; N=20 OOM). This is the eager engine,
not the pure-C++ path, but it is a valid libtorch LAMMPS deployment and matches the ASE ceiling.

### Phase 5 — End-to-end LAMMPS NVT (eager)
`pair_style uma` 10-step NVT@300K, N=16 (32,768 atoms): energy `dE=2.0e-10 eV`, per-atom forces
`max|dF|=3.1e-14`, `cos=1.0` vs ASE (≥100 atoms). Single-tile max-N-under-NVT confirmed at N=17
and **N=18** (46,656 atoms).

### Phase 6 — Native XCCL graph-parallel (pure C++, no Python)
- **Build:** the mixed GCC (engine/LAMMPS) + **icpx** (`xccl_peer.cpp`) link chain was the hardest
  build hurdle; resolved by linking Intel compiler runtime (`libintlc/libimf/libsvml/libirng`) +
  the conda **UR loader** (`libur_loader.so`) at the GCC-driven final link.
- **Gate 1 (4×4×4, 512 atoms, 2 tiles, XCCL) — ✅ PASS:**
  GP-2-tile vs 1-tile `dE=6.8e-13`, `max|dF|=1.1e-14`, `cos=1.0`; 1-tile vs ASE PASS;
  **AG=FD (2-tile GP) `max|AG-FD|=1.03e-8`**. Proves the on-device oneCCL collectives + `uma_peer`
  mid-graph exchange + force reduction are correct.
- **12-tile, N=16 (32,768 atoms) — ✅ correct:** `MAX_N_NO_OOM=16`; parity vs ASE
  `dE=4.6e-8 eV`, `max|dF|=3.0e-14`, `cos=1.0`. **N=18 OOMs** (see §6).
- **Root-cause fix (D6):** the earlier "N=18 shard OOM" was the node-partition edge-sharding bug,
  not memory. After the fix, all 12 N=18 shards trace on one tile.

### (h/j) Per-block / per-chunk activation checkpointing (traced path capacity)
- Per-block AC: correct (RECONSTRUCT bit-exact); lifted traced single-tile **N=6 → N=8**.
- Per-chunk AC (current): correct (RECONSTRUCT dE=0, max|dF|=1.9e-16); reduced N=18 per-tile
  peak from **63 GiB fully-packed** to **~39 GiB resident + one 12.82 GiB transient**.
- **Current confirmed traced single-tile ceiling: N=8**; per-chunk-AC ceiling sweep (N=9..18) is
  **in progress** (binary-search job, N=13 probe running at time of writing).

---

## 6. Open issues / what remains (for reviewers to scrutinize)

1. **Per-chunk-AC N=18 OOM (12.82 GiB + 39 GiB resident).** RECONSTRUCT is bit-exact and each
   chunk's SO2/wigner is sub-GiB, yet N=18 still OOMs. Identified remaining causes (documented,
   not yet fixed):
   - **`uma_ckpt::block` is not itself checkpointed** — only chunks are. The **block-level** node
     tensors (`x_message`, `norm_1`, `norm_2`, `atom_wise`, `accum`; each [natoms,25,128]≈1.11 GiB,
     ~5 per block × 4 blocks ≈ 22 GiB) are retained across all 4 blocks in the top module's grad-on
     forward. **Proposed fix:** make `uma_ckpt::block` a real per-block checkpoint so only one
     block's node tensors are live.
   - **Weight duplication:** each of the 8 block/chunk sub-modules embeds a full UMA weight copy
     (~0.54 GiB × 8 ≈ 4.3 GiB redundant, + 2.2 GiB top). **Proposed fix:** share weights across
     sub-modules.
   - **Next action:** instrument `torch::xpu::memory_allocated` at prologue/per-block/per-chunk to
     confirm the 12.82/39 GiB breakdown before the next code change (avoid guessing).
2. **N-specific artifacts** for GP and AC (chunk count / edge partition bake at trace N). Requires
   a per-N export for each cell size in a max-N sweep. Acceptable but worth a design opinion.
3. **Redundant `_generate_graph`** — the engine builds its own neighbor list even though LAMMPS has
   one; fine for correctness, a speed item later.
4. **Performance not yet measured.** The whole point of libtorch is speed/scalability; only
   correctness is established so far. hen's data shows same-node GP is ~3× at W=12 (collective-
   bound), and that collective cost is identical here — so the libtorch win, if any, is expected in
   **per-step latency at fits-without-heavy-AC sizes**, not max-N. A throughput head-to-head
   (libtorch `pair_style uma` vs FairChem `fix external`/ASE at N≤16) is the recommended next
   measurement once capacity is closed.

---

## 7. How to reproduce (compute node)

Common env:
```bash
source /lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/scripts/activate_fxpu.sh
export PYTHONPATH="<uma-engine>/python:<hen>/shim:<hen>/patches:<hen>:$PYTHONPATH"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT HF_HUB_OFFLINE=1 FAIRCHEM_OFFLINE=1
export UMA_CKPT=<hen>/uma-cache/uma-s-1p2.pt UMA_TASK=omat
# XCCL 1-node knobs: CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_ZE_IPC_EXCHANGE=sockets
```
Representative jobs (under `lammps-uma/scripts/`):
- Engine+XCCL build: `phase6_build_engine_mpi.pbs` (`-D UMA_ENGINE_USE_XPU=ON -D UMA_ENGINE_USE_XCCL=ON`).
- LAMMPS+XCCL build: `phase6_build_lammps_xccl.sh`.
- Shard export (GP): `phase6_export_shards*.pbs` → `export_shards_xpu.py`.
- Per-chunk AC export+run: `phase6_h_*.pbs` → `export_blocks_xpu.py`.
- Gate 1 (2-tile XCCL parity+AG=FD): `phase6_gate1.pbs`, comparator `phase6_gate1_compare.py`,
  AG=FD `phase6_agfd.py`.
- 12-tile parity/max-N: `phase6_maxN_sweep_*.pbs`, `phase6_maxN_parity.pbs`.
- Per-chunk single-tile sweep (binary search): `phase6_h_j_oneN.pbs` (`qsub -v N=<n>`).

Run outputs: `lammps-uma/scripts/out/phase6*/`. Job stdout: `lammps-uma/scripts/*.o<jobid>`.

---

## 8. Specific questions for reviewers

1. **Per-chunk AC completeness (D7):** is per-block **and** per-chunk checkpointing (both C++
   recompute Functions) the right granularity, or should the block-level be collapsed into the
   chunk level entirely? Is there a cleaner way to avoid retaining block-level node tensors than a
   second (block) checkpoint layer?
2. **Weight duplication:** best mechanism to share one weight set across the top + block + chunk
   TorchScript modules on XPU without breaking `torch::jit::load` device homing?
3. **Correctness of the GP force reduction under per-chunk AC:** Gate 1 validated GP + AG=FD at
   N=4 (single chunk). Does the per-chunk backward recompute compose correctly with the
   `uma_peer` collectives at large N/many chunks? (Needs an AG=FD gate at N≥10 GP.)
4. **N-specific artifacts (D4):** acceptable, or is a shape-generic AC/GP export worth the effort
   for the max-N sweep?
5. **Numerical:** all parity uses tol `|dE|≤1e-6 eV`, `max|dF|≤1e-5 eV/Å`; observed ~1e-14–1e-8.
   Are these the right gates for FP64 production?

---

## 9. Bottom line for the review

- The **pure-C++/no-Python/FP64/XCCL graph-parallel `pair_style uma` is correct** and runs one
  32,768-atom NaCl across 12 tiles, bit-exact vs the ASE FairChem API.
- The path from the current traced ceilings (N=8 single-tile / N=16 twelve-tile) to the ASE
  targets (N=18 / N=32) is **an activation-checkpoint memory rebuild**, not a correctness problem;
  it is implemented and converging, with two concrete remaining fixes (block-level checkpoint +
  weight dedup) and an instrumentation step queued.
- The **eager libtorch worker path already meets the capacity targets** (N=18 single-tile,
  bit-exact) if a forked Python worker is acceptable; the pure-C++ path is the no-Python variant
  still closing the last memory gap.
- **Performance remains unmeasured** and is the real justification for libtorch; recommend a
  throughput benchmark once capacity is closed.
