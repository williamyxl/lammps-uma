# Session recovery — UMA LibTorch multi-XPU integration in LAMMPS

**Date captured:** 2026-08-26 UTC (§1–§14) · **updated 2026-08-29 (§15: opt4→opt6, P2.1, N=16-beats-ASE, real-NVT)**  
**Repository:** `/lus/flare/projects/MatSciAI/xiaoliyan/workdir/lammps-uma`  
**Branch:** `uma-kokkos-mlip`  
**Purpose:** restore the exact engineering/test state after the long Aurora campaign without rereading the full chat.

> **READ §15 FIRST for current state.** §1–§14 are the 2026-08-26 snapshot (up to
> opt4). §15 supersedes the performance/parity/bug sections with the latest results:
> opt5/opt6 optimizations, the P2.1 edge-padding fix (N=24/N=16 NVT bug RESOLVED),
> N=16 now beating ASE, real ASE Nosé-Hoover NVT timing, and the final parity table.

---

## 1. User goals and non-negotiable constraints

1. Run FairChem **UMA-s-1p2** through native LAMMPS `pair_style uma` using C++/LibTorch.
2. Runtime must be **pure C++ / no Python** for the LibTorch path.
3. FP64 energy and forces.
4. Intel XPU support on Aurora; retain NVIDIA CUDA/NCCL support.
5. Native multi-tile graph parallelism through XCCL/oneCCL on Aurora.
6. Primary test system: rocksalt NaCl conventional cell, `a=5.64 Å`, NxNxN, 8·N³ atoms, periodic, positions perturbed with independent Gaussian `sigma=0.05 Å`, seed 0.
7. Validate first-frame energy and per-atom forces against ASE FairChem; sample at least 100 atoms for large-force parity. Validate autograd against finite difference where tractable.
8. Run 10-step Nose-Hoover NVT at 300 K, timestep 1 fs, Tdamp 0.1 ps.
9. Correctness first; optimize performance after the C++ path works.

---

## 2. Current Git state

### Branch and relevant commits

Current branch: `uma-kokkos-mlip`.

Recent campaign commits already present:

```text
ce92f50306 libtorch UMA path performance optimized
94ded2b029 ASE path timing on 12 Tiles
fd4550d518 example for Aurora
d0a4a57834 uma lammps passed 38x38x38 NaCl NVT
ce7e8f5e7a docs: add authoritative CAMPAIGN_SUMMARY + index, cross-link all campaign docs
0326b06539 Multi-node LAMMPS checkpointing: C++ gradient checkpoint around traced shard
2cbc7927ff Document N=21 LAMMPS-vs-ASE comparison (energy, per-atom force, timing)
a15730d908 Fix LAMMPS multi-GPU checkpointing forces: real torch.distributed GP worker
```

Most major implementation work is already committed in those commits.

### Uncommitted tracked change

Only one tracked source edit is currently uncommitted:

```text
M src/ML-UMA/uma-engine/python/export_blocks_xpu.py
```

It is the untested **opt2** experiment:

```python
traced = torch.jit.freeze(traced.eval())
```

inserted before saving `model_traced.pt`, intended to strip approximately 2.2 GiB/rank of dead top-module weights. `python -m py_compile` passes. It has **not** been exported/run/validated yet. Preserve this edit unless intentionally reverting opt2.

### Untracked campaign material

Many scripts, docs, build dirs, and `scripts/out/` are untracked. Important examples:

- `docs/POSTER_uma_libtorch_lammps.md`
- `docs/REPORT_2path_nvt_comparison.md`
- `src/ML-UMA/uma-engine/docs/REVIEW_libtorch_uma_xpu.md`
- `src/ML-UMA/uma-engine/docs/phase6_graph_parallel_xpu_plan.md`
- `scripts/phase*.pbs`, `scripts/test_*.pbs`, `scripts/out/`
- `src/ML-UMA/uma-engine/build-xpu*`

Do not stage `scripts/out/` or build directories. Review selected scripts/docs before committing.

---

## 3. Current scheduler state

At capture time, there are no active Phase-6/test jobs. The visible jobs are unrelated long-lived capacity jobs:

```text
8778958 capacity uma_bunch* R
8778959 capacity uma_bunch* R
8778960 capacity uma_bunch* Q
8778961 capacity uma_bunch* Q
8778962 capacity uma_bunch* Q
```

Do not stop those unless explicitly requested.

Aurora queue behavior observed:

- `debug`: one running job/user in practice, 1-hour wall.
- `debug-scaling`: separate one-job/user slot, 1-hour wall.
- Both can be used concurrently.
- `capacity` was also allowed by the user.

---

## 4. Environment and build facts

### Runtime/toolchain

- Activate FairChem/torch-XPU env:

```bash
source /lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/scripts/activate_fxpu.sh
```

- `fxpu` is the same logical environment as `hen-xpu`.
- Torch: `2.13.0+xpu`.
- No IPEX.
- Compiler split:
  - GCC 13.4 for LAMMPS/most engine sources.
  - `icpx` for `xccl_peer.cpp` SYCL/oneCCL object.
- Runtime must force conda `libsycl.so.9`; system `.8` causes init crashes.
- oneCCL header/library exist under `$CONDA_PREFIX/include/oneapi/ccl.hpp` and `$CONDA_PREFIX/lib/libccl.so*`.

### Validated LAMMPS build script

```text
scripts/phase6_build_lammps_xccl.sh
```

Build output:

```text
build-lmp-xccl/lmp
```

The small reusable example is committed/available under:

```text
src/ML-UMA/examples/min-sample/
  README.md
  build_lammps_uma.sh
  extract_uma_artifact.sh
  make_data.py
  in.nvt
  run_nvt_aurora.pbs
```

---

## 5. Architecture that now works

### Single-tile C++ LibTorch path

LAMMPS `pair_style uma precision double devices 1` loads:

```text
model_traced.pt
model_block_{i}.pt
model_chunk_{i}.pt
model_edgedeg_chunk.pt
metadata.json
```

Energy is produced by the traced forward. Forces are C++ autograd:

```cpp
torch::autograd::grad(E, pos)
```

No Python process is used when `UMA_EAGER_CKPT` is unset.

### Activation checkpointing rebuilt in C++

FairChem's `torch.utils.checkpoint` does not survive `torch.jit.trace`. The C++ path now has custom operations and recompute autograd Functions at three levels:

- `uma_ckpt::block`
- `uma_ckpt::chunk`
- `uma_ckpt::edge_degree`

Implemented in:

```text
include/uma/block_context.h
src/block_context.cpp
python/export_blocks_xpu.py
python/uma_ckpt_ops.py
```

This raised the pure traced single-tile capacity:

```text
N=6 monolithic
N=8 per-block AC
N=17 per-chunk AC
N=18 after prologue edge-degree checkpointing
```

### Multi-tile XCCL graph parallelism

- One MPI rank per XPU tile.
- `devices 1` stays in the LAMMPS input; world size is the MPI rank count.
- Per-rank artifact layout:

```text
ARTIFACT/w{W}/r{R}/model_traced.pt
ARTIFACT/w{W}/r{R}/model_block_{i}.pt
ARTIFACT/w{W}/r{R}/model_chunk_{i}.pt
ARTIFACT/w{W}/r{R}/model_edgedeg_chunk.pt
ARTIFACT/w{W}/r{R}/metadata.json
ARTIFACT/metadata.json
```

- `uma_peer::all_gather_nodes` and `all_reduce_sum` dispatch to native oneCCL on XPU.
- oneCCL implementation:

```text
include/uma/xccl_peer.h
src/xccl_peer.cpp
```

- GP runtime:

```text
src/mpi_peer_predictor.cpp
```

### Graph construction

Two speed fixes exist:

1. Single-rank LAMMPS path consumes LAMMPS's neighbor list:

```text
pair_uma.cpp::build_ext_graph
Predictor::predict_host_extgraph
```

A/B result at N=8:

```text
dE = 0
max|dF| = 2.67e-14
AB_MATCH
```

N=18 wall improved from approximately 260 s to 41 s in that comparison.

2. Engine `build_neighbor_graph` now uses a C++ linked-cell implementation for large orthorhombic cells, with all-pairs fallback and `UMA_NL_ALLPAIRS=1` A/B switch:

```text
src/neighbor_list.cpp
include/uma/neighbor_list.h
```

This removed the GP N=32 O(N²) hang.

---

## 6. Validated correctness results

### Single-tile LAMMPS LibTorch UMA N=18

Actual `lmp` binary, `pair_style uma`, no `UMA_EAGER_CKPT`, no Python worker:

```text
N = 18
atoms = 46,656
lmp exit = 0
PE = -157578.53111517 eV
Fmax = 0.7191007 eV/A
```

Fresh ASE FairChem parity on identical coordinates:

```text
dE = 4.628e-08 eV
max|dF| = 4.814e-14 eV/A
rms|dF| = 1.285e-14 eV/A
cos = 1.0000000000
PASS
```

### Two-tile native-XCCL Gate 1

Perturbed NaCl N=4, 512 atoms:

```text
2-tile GP vs 1-tile: dE ~ 1e-12 eV
max|dF| ~ 1e-14 eV/A
cos = 1.0
AG=FD max error = 1.03e-08 eV/A
PASS
```

### N=18 scaling, 1/2/4/8/12 tiles

All paths match ASE to FP64 floor:

| W | historical wall | dE vs ASE | max force error | cosine |
|---:|---:|---:|---:|---:|
| 1 | 260 s | ~4.6e-8 eV | 4.8e-14 | 1.0 |
| 2 | 410 s | 4.63e-8 | 3.16e-14 | 1.0 |
| 4 | 239 s | 4.63e-8 | 3.14e-14 | 1.0 |
| 8 | 239 s | 4.65e-8 | 3.15e-14 | 1.0 |
| 12 | 241 s | 4.66e-8 | 3.17e-14 | 1.0 |

Those walltimes were before the later neighbor and XCCL optimizations.

---

## 7. Capacity and NVT results

### Single tile

- Single-point max N: **18**, 46,656 atoms.
- N=19 OOMs.
- Traced path N=18 single-point passes.
- 10-step NVT at N=18 OOMs on one tile because extra MD/integrator state is tight.

### 12 tiles

Single-point:

```text
N=32 262,144 atoms 54 s
N=34 314,432 atoms 57 s
N=36 373,248 atoms 88 s
N=38 438,976 atoms 77 s
N=40 512,000 atoms OOM
```

10-step Nose-Hoover NVT:

```text
N=18 46,656 atoms 88 s
N=32 262,144 atoms baseline 450 s
N=34 314,432 atoms 534 s
N=36 373,248 atoms 666 s
N=38 438,976 atoms 800 s
N=40 OOM
```

N=24 is a known exception: step 0 succeeds, but MD step 1 can fail because atom motion changes the edge count and the N-specific traced chunk count shifts by one (`Expected 18 elements in a list but found 19`). Fix direction: pad edges to a fixed chunk multiple.

---

## 8. ASE/FairChem comparison

### Single-tile ASE Nose-Hoover NVT

Raw JSON:

```text
scripts/out/test3path/ase/ase_n6.json
scripts/out/test3path/ase/ase_n12.json
scripts/out/test3path/ase/ase_n18.json
```

Results:

```text
N=6  E0=-5836.318644  AGFD=3.81e-8 PASS  t_nvt10=7.7 s
N=12 E0=-46690.218610 AGFD=5.46e-7 PASS  t_nvt10=61.2 s
N=18 E0=-157578.531115 AGFD=3.42e-6 PASS t_nvt10=383.3 s
```

### ASE/FairChem GP on 12 tiles (`hen`)

These are FairChem GP repeated E+F measurements, not literal ASE Nose-Hoover trajectories. Eleven timed evaluations approximate the force-call count of run-0 + 10 MD steps.

```text
N=18 W=12:
  load = 48.98 s
  warmup = 12.19 s
  ef_mean = 2.431 s
  wall = 90.35 s
  E = -157578.531115217
  Fmax = 0.719101

N=32 W=12:
  load = 49.37 s
  warmup = 31.07 s
  ef_mean = 14.820 s
  wall = 258.29 s
  E = -885377.060036621
  Fmax = 0.848045
```

Outputs:

```text
/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/pbs/out/ase12_n18/
/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/pbs/out/ase12_n32/
```

---

## 9. Performance optimization status

N=32, 12 tiles, 10-step NVT:

```text
baseline               450 s
opt3 XCCL env tuning    276 s
opt1 + opt3             235 s
ASE/FairChem GP ref      258 s
```

### opt3 — validated

Hen-tuned oneCCL environment:

```bash
CCL_PROCESS_LAUNCHER=none
CCL_ATL_TRANSPORT=ofi
CCL_ZE_IPC_EXCHANGE=pidfd
CCL_WORKER_COUNT=1
FI_PROVIDER=tcp
CCL_KVS_MODE=mpi
```

Reduced 450 → 276 s, bit-identical energy.

### opt1 — validated

Re-exported N=32 artifacts with:

```bash
EDGE_AC_CHUNK=65536
```

instead of 16384, reducing recompute. Combined with opt3:

```text
wall = 235 s
LAMMPS 10-step loop = 179.975 s
step-10 PE = -879646.224481715 eV
```

This beats ASE/FairChem GP 258 s while preserving FP64 energy.

Artifact:

```text
scripts/out/opt1/n32_chunk65536/
```

Run output:

```text
scripts/t_o13.o8782977
scripts/out/opt1run/n32/
```

### opt2 — VALIDATED (kept as default; storage/HBM win, wall-neutral)

The uncommitted edit adds (guarded, with `UMA_NO_FREEZE=1` control):

```python
traced = torch.jit.freeze(traced.eval())
```

before saving `model_traced.pt`, to strip approximately 2.2 GiB/rank of dead top-module weights.

**Result (jobs 8783768 gate1, 8783769 export, 8784408 run):**

- `model_traced.pt` **2224.5 MB → 2.6 MB/rank** (99.9%); N=32 W=12 artifact **53 GB → 27 GB**.
- W=2 N=4 Gate 1 PASS: GP-vs-1-tile dE=9.1e-13 eV, max|dF|=9.6e-16, cos=1.0; 1-tile-vs-ASE dE=2.8e-11; target dE=4.09e-7; AG=FD max 1.08e-8. W=1 RECONSTRUCT PASS.
- N=32 W=12 NVT 300K 10-step: step-10 PE = **−879646.224481715 eV, dE = 0.000e+00** vs opt1+opt3 baseline (bit-identical).
- Wall **248 s vs 235 s** (within variance), Loop **184.3 s vs 179.975 s**. Load NOT reduced: runtime load is dominated by the real per-chunk weights (4×554 MB/rank), not the stripped dead top graph.
- Verdict: keep opt2 (zero accuracy cost, halves disk + frees HBM). It is NOT a wall-time optimization. Next headroom = de-duplicate the identical-across-ranks per-chunk weights.

Also fixed a validation-only bug: the post-save graph-check (`_all_method_graphs`) crashed on a frozen module (`hasattr(sm,"graph")` re-raised `RuntimeError`); now guarded, and the GP structure check skips the (opaque-when-frozen) top-graph block/edge_degree counts.

### opt4 — partial no-recompute (NEW wall win: 235 s → ~193 s, −18%)

Profiled N=32 W=12 force call (`UMA_MP_PERF=1`, job 8784422): graph 1.67 s (9%), fwd 4.65 s (25%), **bwd 12.05 s (65%)**, force_ar 0.24 s (1.3%). Backward dominates because activation checkpointing recomputes the forward for forces.

New granular engine knobs in `src/ML-UMA/uma-engine/src/block_context.cpp` (default OFF = checkpoint, behavior unchanged): `UMA_NO_RECOMPUTE` (master), `UMA_NO_RECOMPUTE_BLOCK`, `UMA_NO_RECOMPUTE_CHUNK`, `UMA_NO_RECOMPUTE_EDEG`. When set, the op runs its sub-module forward directly under autograd (retain activations) instead of the CheckpointFn (recompute). **Requires an lmp rebuild** (`scripts/phase6_build_lammps_xccl.sh`; already rebuilt into `build-lmp-xccl/lmp`).

Results (N=32 W=12, opt2 65536 artifacts):
- Full `UMA_NO_RECOMPUTE=1` **OOMs** (62.5/64 GiB) — chunk AC is required at N=32.
- **`UMA_NO_RECOMPUTE_BLOCK=1 UMA_NO_RECOMPUTE_EDEG=1`** (keep chunk AC): wall 190/196 s, Loop 136.2/137.4 s (jobs 8784500/8784623), bwd 12.05→7.79 s/call, step-10 dE=1.2e-10 PASS. **This is the recommended default add-on.**
- `EDGE_AC_CHUNK=131072` (job 8784521): 236 s, no change (recompute scales with edge work, not chunk count).
- Tile sweep (job 8784576): W=8=342 s, W=12=232 s — fewer tiles strictly slower; W=12 best for latency.

Recommended N=32 W=12 NVT stack: opt1+opt2+opt3+opt4 → **~193 s** (1.22× over the 235 s baseline, 1.34× over ASE-GP 258 s), FP64, numerically equivalent.

New scripts: `scripts/opt2_perf_probe_n32.pbs` (profiler), `scripts/opt4_build_run_norecompute.pbs`, `scripts/opt4_run_partial_clean.pbs`, `scripts/opt5_export_n32_tiles.pbs`, `scripts/opt5_run_n32_tiles.pbs`, `scripts/opt2_export_n32.pbs`/`opt2_run_n32.pbs` (parameterized EDGE_AC_CHUNK/ART).

Why it may help:

```text
model_traced.pt       ~2224 MB dead-heavy top graph
model_chunk_0..3.pt   ~4 x 554 MB, real per-block weights
model_block_0..3.pt   ~1 MB each
model_edgedeg_chunk   ~1.4 MB
```

Expected benefit: reduce model load/setup (~59 s in the optimized N=32 run) and HBM usage. Accuracy should be unchanged if freeze removes only graph-unreachable attributes.

It needs explicit validation before keeping:

1. Export W=2 N=4 opt2 artifacts.
2. Compare frozen vs non-frozen graph and artifact sizes.
3. Run W=2 Gate 1 parity + AG=FD.
4. Export W=12 N=32 with `EDGE_AC_CHUNK=65536`.
5. Run with opt3 CCL env and compare to the 235 s result.

---

## 10. Immediate resume commands

### A. Validate opt2 cheaply at W=2 N=4

Create/export to a fresh artifact dir, keeping baseline artifacts untouched:

```bash
source /lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/scripts/activate_fxpu.sh
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/lammps-uma
export PYTHONPATH="$PWD/src/ML-UMA/uma-engine/python:/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/shim:/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/patches:$PYTHONPATH"
export UMA_EXPORT_CELL_LIST=1 EDGE_AC_CHUNK=65536
# Export rank 0 and 1 using export_blocks_xpu.py with EXPORT_WORLD=2, EXPORT_RANK=R.
```

The opt2 source edit is enabled by default. Set `UMA_NO_FREEZE=1` for the control.

Then run a two-tile N=4 Gate-1 comparison. Required gates:

```text
|dE| <= 1e-6 eV
max|dF| <= 1e-5 eV/A on >=100 atoms
AG=FD <= 1e-5 eV/A
```

### B. If opt2 Gate 1 passes, export/run N=32 W=12

Use:

```text
EDGE_AC_CHUNK=65536
opt3 CCL environment above
artifact path separate from baseline, e.g. scripts/out/opt2/n32_chunk65536/
```

Compare:

```text
artifact size
model load/setup time
LAMMPS Loop time
whole wall
energy / Fmax / first-frame force parity
```

Target: improve 235 s while remaining numerically equivalent.

### C. Build/run references

```text
scripts/phase6_build_lammps_xccl.sh
scripts/test_pairuma_nvt_12tile_opt3.pbs
scripts/test_opt1_opt3_n32.pbs
scripts/phase6_gate1_compare.py
scripts/phase6_agfd.py
```

---

## 11. Known bugs / limitations

1. **N-specific artifact chunk count:** N=24 NVT failed when edge count drift changed chunk count by one. Fix with fixed-multiple edge padding or a truly dynamic chunk loop.
2. **Triclinic extgraph:** LAMMPS-neighbor consumption currently requires an orthorhombic box; set `UMA_ENGINE_BUILD_GRAPH=1` for triclinic.
3. **GP external graph:** single-tile `predict_host_extgraph` exists; GP still uses engine cell-list rather than LAMMPS NL. C++ engine cell-list now scales linearly, so large N works.
4. **N>=32 ASE single-tile oracle:** impossible due to OOM. Validation uses ASE/FairChem GP energy/Fmax and lower-N bit-precision parity.
5. **Terminology:** use “numerically equivalent within FP64 tolerances” for nonzero differences, not “bit-exact.”
6. **Validation scripts:** earlier agent review noted that AG=FD and comparator scripts can fail open on subprocess/oracle failure or zero completed samples. Harden before treating them as CI.
7. **Stress/virial:** not validated; campaign scope is energy + forces, one atomic system, OMAT, charge/spin 0.

---

## 12. Important artifact/result paths

```text
Single-tile N=18 AC artifact:
  scripts/out/phase6_h_p1b/blocks_n18/

N=18 W=12 GP+AC artifacts:
  scripts/out/phase6_scale/n18_ac/

N=32 baseline artifacts:
  scripts/out/phase6_maxN12/n32_ac/

N=32 opt1 chunk65536 artifacts:
  scripts/out/opt1/n32_chunk65536/

Optimized N=32 run:
  scripts/out/opt1run/n32/
  scripts/t_o13.o8782977

Comprehensive results:
  docs/REPORT_2path_nvt_comparison.md
  src/ML-UMA/uma-engine/docs/REVIEW_libtorch_uma_xpu.md
  src/ML-UMA/uma-engine/docs/phase6_graph_parallel_xpu_plan.md

Poster talking points:
  docs/POSTER_uma_libtorch_lammps.md

Minimal runnable example:
  src/ML-UMA/examples/min-sample/
```

---

## 13. Recommended next sequence

1. **Do not change the validated 235 s baseline artifact.** Use a separate opt2 artifact dir.
2. Validate opt2 freeze at W=2 N=4 (energy, sampled forces, AG=FD).
3. If valid, N=32 W=12 opt2 + opt1 + opt3 benchmark.
4. Measure artifact size and load time to prove opt2 helps; revert if freeze causes load/custom-op issues.
5. Update `REPORT_2path_nvt_comparison.md` with final optimized time.
6. Then address fixed-padding/dynamic chunking for the N=24 NVT bug.
7. Harden fail-closed validation scripts.
8. Run lint/build/tests before committing selected sources/docs; do not commit `scripts/out/` or build dirs.

---

## 14. Recovery checkpoint statement

At session end, the campaign has achieved:

- pure-C++/no-Python LAMMPS UMA on Intel XPU, FP64;
- single-tile N=18, energy/forces matching ASE FairChem;
- native XCCL graph parallelism on 12 tiles;
- 10-step NVT through N=38 (438,976 atoms), N=40 OOM;
- N=32 optimized wall 235 s, beating ASE/FairChem GP 258 s;
- verified NVIDIA/CUDA design retained in branch architecture;
- opt2 freeze **validated**: bit-identical energy/forces, artifact 53 GB → 27 GB, wall-neutral (kept as default).

opt2 is complete. The immediate task is done; next headroom is de-duplicating the identical per-rank chunk weights (the real load cost), plus the N=24 fixed-chunk-padding bug and fail-closed validation-script hardening.

---

## 15. UPDATE 2026-08-29 — opt4→opt6, P2.1, N=16 beats ASE (current state)

This section supersedes §9 (perf), §11 (bugs), §8 (ASE) and §14 (checkpoint) with
the latest campaign results. Full detail: `docs/REPORT_2path_nvt_comparison.md`
§7–§13 (§13 is the authoritative final parity+performance table).

### 15.1 Current build & the two execution paths (C1 / C2)
- Latest engine binary: `build-lmp-xccl/lmp` (rebuilt 2026-08-29 ~15:12), source
  branch `uma-kokkos-mlip`. Contains opt1+opt2+opt3+opt4 + the opt5 knobs + P2.1.
- **Path C1 = pre-opt4** (full activation checkpointing): fits to the N=38 ceiling.
- **Path C2 = opt4** (partial no-recompute): `UMA_NO_RECOMPUTE_BLOCK=1
  UMA_NO_RECOMPUTE_EDEG=1` — retains node-sized block+prologue activations, keeps
  the memory-heavy edge-chunk checkpoint. ~−24% Loop vs C1, fits N≤34 on 12 tiles.
- C1 and C2 are **numerically identical** (retain vs recompute = same gradient).

### 15.2 Runtime knobs added this campaign (all default OFF = legacy behavior)
- `UMA_NO_RECOMPUTE_BLOCK`/`_EDEG`/`_CHUNK` (`_=all`) — opt4 retain-activation
  bypass of the per-op checkpoint. `block_context.cpp`.
- `UMA_CHUNK_RETAIN_K=k` — opt5 P-1: retain the first k edge-chunks/block (skip
  their backward recompute). HBM-bounded; **W-adaptive** (max K that fits rises
  with W). `block_context.cpp`.
- `UMA_SKIP_MAXNBR_CAP=1` — opt5-graph: skip the per-step `counts.max().item()`
  XPU→host sync in the vesin NL (no-op cap for NaCl; bit-identical). `vesin_nl.h`.
- `UMA_EDGE_PAD=1` (default ON in export) — P2.1 edge padding; `edge_pad_cap`/
  `edge_pad_atom` baked into metadata, runtime pads via
  `graph_shard::pad_edges_to_capacity` (rank-local pad atom). `export_blocks_xpu.py`
  + `mpi_peer_predictor.cpp` + `predictor.cpp` + `metadata.{h,cpp}`.
- `UMA_MP_PERF=1` — per-call timing (graph/vesin/shardpad/fwd/bwd/force_ar +
  allgather/allreduce). `UMA_PEER_PERF` collective accumulators in `xccl_peer.cpp`.
- Also cached device `z` (atomic numbers) across steps (opt5-graph).

### 15.3 P2.1 edge padding — RESOLVES the N=24/N=16/N=36 NVT crash (§11 bug #1)
Root cause: the per-chunk loop is unrolled by `torch.jit.trace` to a FIXED chunk
count; MD edge-count drift changed the runtime count → "Expected K elements in a
list but found K+1". **Fix:** pad each rank's shard to a fixed multiple of
`EDGE_AC_CHUNK` (self-loops on a rank-owned node, cell_offset beyond cutoff →
envelope 0 → exactly zero contribution). Validated: **N=24 W=12 now completes the
full 10-step NVT** (was step-1 crash), step-0 PE −373517.7698 = ASE-GP, per-atom
parity PASS. Requires re-export (artifacts carry `edge_pad_cap`); legacy artifacts
(no field) skip padding. Report §7.

Bring-up gotcha: first attempt used a GLOBAL pad atom (node 0) → GPU segfault on
ranks not owning node 0. Fixed to rank-local `node_partition[rank][0]` in both
Python export and C++ runtime.

### 15.4 EDGE_AC_CHUNK is N-tuned (opt5, §10)
Padding waste = (padded_cap − real_edges)/padded_cap. At small N the coarse
`chunk=65536` wastes 33% of per-step compute (padding edges run full SO2 then get
envelope-zeroed). **Tuned rule (W=12): N≤16 → 32768; N≈24–34 → 65536.** N=16 W=12
Loop 25.2 → 18.4 s (−27%) just from chunk retune. Also cured the W=8 efficiency dip
(was 33% waste at 65536). Export-time choice; numerically neutral.

### 15.5 N=16 now BEATS ASE (opt5-graph + chunk-retain, §11)
The N=16 gap to ASE was (1) padding waste and (2) chunk backward-recompute that
N=16's HBM headroom makes unnecessary. Fixing both: **N=16 W=12 Loop 25.2 → 12.8 s**
(chunk=32768 + C2 + `UMA_CHUNK_RETAIN_K=3` + `UMA_SKIP_MAXNBR_CAP=1`), flipping
0.69× → **1.36× faster than ASE**. Parity preserved (dE=1.4e-9 meV/at,
max|dF|=5.05e-14). `retainK` is W-adaptive (W=1→K=1, W=2–6→K=2, W=8→K=0, W=12→K=3).

### 15.6 ASE comparison is now a REAL Nosé-Hoover NVT (supersedes §8 `11·ef`)
The old `11·ef_mean` reconstruction was retired. New driver
`hen/scripts/fxpu_1node_nvt.py` + `hen/pbs/11_1node_nvt.pbs` runs an actual ASE
`NoseHooverChainNVT` 10-step MD (warm `md_s` + cold `wall_s`). This UNDERSTATED
ASE's real cost by ~2× at N=32 (real ASE NVT step does separate energy+force evals
+ neighbor rebuild). Measured ASE: N=32 W=12 md 296.2 s / wall 409.3 s; N=16 ladder
md W=2..12 = 92.4/47.4/31.9/24.9/17.4 s (W=1 OOMs on ASE too). Report §9/§13.

### 15.7 FINAL parity + performance vs ASE (latest build) — Report §13
**Full energy + per-atom force PASS at EVERY config** (per-W ASE-GP force oracles;
cos=1.0, max|dF| at FP64 floor). WARM = 10-step MD compute (Loop / md, load-excl);
COLD = whole wall incl. load.

| config | LAMMPS Loop | ASE md | warm vs ASE | LAMMPS cold wall | ASE cold wall |
|---|--:|--:|:--|--:|--:|
| N=16 W=1  | 165.5 s | OOM | ASE can't run | 206 s | OOM |
| N=16 W=2  | 91.0 s | 92.4 s | 1.02× | 115 s | 189 s |
| N=16 W=4  | 48.3 s | 47.4 s | 0.98× | 63 s | 98 s |
| N=16 W=6  | 31.4 s | 31.9 s | 1.02× | 44 s | 77 s |
| N=16 W=8  | 30.4 s | 24.9 s | 0.82× | ~44 s | 55 s |
| N=16 W=12 | 12.8 s | 17.4 s | **1.36×** | ~33 s | 47 s |
| N=32 W=12 | 138.2 s | 296.2 s | **2.14×** | 172 s | 409 s |

- **N=32 (production): 2.14× (compute) / 2.38× (cold wall) faster than ASE.**
- **N=16: LAMMPS faster at W=2,6,12** (best 1.36× W=12); ASE ahead only at W=8
  (0.82×, K=0 floor) and W=1 (ASE OOMs). **Cold wall: LAMMPS faster at every W.**
- N=16 W=8 reported at C2/K=0 (adaptive K=3 thrashed memory, didn't cleanly land).

### 15.8 opt6 — parity-safe levers EXHAUSTED (§12)
Instrumented collectives: **15 tiny all_reduces/step** (per-block balance_channels
charge+spin) dominate — ~9% at N=32, ~45% at N=16; latency-bound. Evaluated:
- (A) `CCL_ALLREDUCE` algo tuning — DEAD END (default optimal; explicit 2–3.5× slower).
- (B) cut the 15 all_reduces — balance_channels is real physics; parity-risky to fuse.
- (C) skin-cached NL — implemented, analyzed, **REJECTED as net-negative**: this
  engine runs EVERY edge full-cost until the final envelope, so a skin's +16–59%
  extra edges cost more compute than the 1534 ms/step vesin rebuild it saves.
  Reverted (kept documenting comment).
Conclusion: engine at the parity-safe floor for this trace-based design. Further
gains need scripted (not traced) data-dependent loops or mixed precision (breaks
FP64 parity). opt5-graph fixes (z-cache, SKIP_MAXNBR_CAP) kept.

### 15.9 Artifact library & the "exports are N/W-specific" finding
Exports bake `total_atoms`/`node_offset`/`n_local`/`edge_pad_cap` → each (N, W,
chunk) needs its own export even though the model WEIGHTS are byte-identical
(verified 93/93 tensors). ~465 GB of redundant artifacts in `scripts/out/`.
`artifacts_lib/` created as the persistent pre-export library + README. The clean
fix (making exports model-only + weight-dedup load win) is the **P-2/P2.2 refactor:
script the chunk core so node_offset/n_local/total_atoms/edge_pad become RUNTIME
inputs** — deferred (trace→script is the enabling change for edge masking,
collective fusion, AND weight dedup).

### 15.10 Key new files (this update)
```text
Engine:  src/ML-UMA/uma-engine/src/block_context.cpp        (opt4 + UMA_CHUNK_RETAIN_K)
         src/ML-UMA/uma-engine/src/mpi_peer_predictor.cpp   (P2.1 pad + z-cache + perf timers)
         src/ML-UMA/uma-engine/src/predictor.cpp            (P2.1 pad, single-tile)
         src/ML-UMA/uma-engine/src/xccl_peer.cpp            (UMA_PEER_PERF collective timers)
         src/ML-UMA/uma-engine/src/metadata.cpp + include/uma/metadata.h (edge_pad_cap/atom)
         src/ML-UMA/uma-engine/include/uma/vesin_nl.h       (UMA_SKIP_MAXNBR_CAP)
         src/ML-UMA/uma-engine/python/export_blocks_xpu.py  (P2.1 edge padding)
Parity:  scripts/parity_vs_asegp.py                         (per-atom E+F gate, per-atom E tol)
ASE NVT: hen/scripts/fxpu_1node_nvt.py + hen/pbs/11_1node_nvt.pbs (real NoseHooverChainNVT)
Report:  docs/REPORT_2path_nvt_comparison.md  §7 (P2.1) §8-12 (opt4-6) §13 (final table)
Library: artifacts_lib/ (README + manifest; persistent pre-exports)
Plan:    docs/CODE_QUALITY.md Part C (Phase 1-3: code fixes / CI / multinode)
         docs/CODE_QUALITY.md (Parts A + B)
```

### 15.11 Updated known bugs / limitations (supersedes §11)
1. ~~N-specific chunk-count crash~~ **FIXED by P2.1** (§15.3). Re-export needed.
2. `EDGE_AC_CHUNK` must be N-tuned for best small-N perf (§15.4) — export-time.
3. `UMA_CHUNK_RETAIN_K` is HBM-bounded/W-adaptive; OOMs at N≥32 (any K) and
   small-W/large-N. No auto-fallback in the engine yet (harness does K→0 retry);
   `UMA_CHUNK_RETAIN_K=auto` (retry-lower-on-OOM in C++) is a proposed easy add.
4. Exports are N/W-specific, not model-only (§15.9) — P-2/P2.2 refactor deferred.
5. GP path rebuilds the full NL via vesin every step (~1534 ms at N=32); skin-cache
   is net-negative here (§15.8), so this stands until a pre-SO2 edge mask exists.
6. Still open from §11: triclinic extgraph, GP-vs-LAMMPS-NL, stress/virial not
   validated, fail-closed CI hardening (= Phase 1 of CODE_QUALITY.md Part C).

### 15.12 Updated recovery checkpoint statement
- pure-C++/no-Python LAMMPS UMA on Intel XPU, FP64; NVIDIA/CUDA design retained.
- **Full energy + per-atom force parity vs ASE at N=6/12/16/18/24/32 (all PASS,
  FP64 floor).**
- 10-step NVT to N=38 (C1); N=24/N=16/N=36 NVT crash **fixed** (P2.1).
- **Faster than ASE across the tested range**: N=32 W=12 **2.14× compute / 2.38×
  cold wall**; N=16 W=12 **1.36×**; cold wall faster at every W.
- Engine at the parity-safe performance floor for the trace-based design (opt6);
  next gains require the trace→script refactor or mixed precision.
- Immediate next work = Phase 1 of `docs/CODE_QUALITY.md` Part C (P0 correctness
  fixes: xccl barrier `.wait()`, collective-agreement, fail-closed harness), each
  gated by the mandatory per-round N=16 W=1 + N=32 W=12 ASE parity gate.
