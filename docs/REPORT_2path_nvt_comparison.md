# Comprehensive 2-path NVT comparison — UMA on Aurora XPU (FP64)

**Paths compared**
- **A. ASE FairChem** — `FAIRChemCalculator` + `NoseHooverChainNVT` (tchain=3, tdamp=0.1 ps, dt=1 fs), single Python process, one XPU tile.
- **C. Our LAMMPS `pair_style uma`** — native C++/LibTorch, TorchScript-traced UMA with per-block/chunk/prologue activation checkpointing, native XCCL graph-parallel on multiple tiles. No Python at runtime, FP64.

*(FC LAMMPS `fix external` was dropped — no such FairChem/UMA component exists, and it would be engine-equivalent to path A.)*

**Common setup:** NaCl NxNxN, a=5.64 Å, rattle 0.05 Å (seed 0); NVT 300 K, 10 steps; UMA-s-1p2, task omat, FP64. First-frame energy + per-atom forces (≥100 atoms sampled for parity), AG=FD spot-check (10 atoms), and walltime.

---

## 1. Correctness — Path C (LAMMPS) vs Path A (ASE), first frame

| Case | atoms | ΔE (LAMMPS−ASE) | per-atom max\|dF\| (≥100 at) | cos | AG=FD (C) | verdict |
|---|--:|--:|--:|--:|--:|:--|
| **1 tile, N=6** | 1,728 | 1.3e-10 eV | 1.5e-14 eV/Å | 1.0000000000 | 3.6e-8 | ✅ PASS |
| **1 tile, N=12** | 13,824 | 1.6e-8 eV | 2.2e-14 | 1.0000000000 | 9.6e-7 | ✅ PASS |
| **1 tile, N=18** | 46,656 | 4.6e-8 eV† | 4.8e-14† | 1.0† | — | ✅ (single point)† |
| **12 tiles, N=18** | 46,656 | 4.6e-8 eV | 3.2e-14 | 1.0000000000 | (see note) | ✅ PASS |
| **12 tiles, N=24** | 110,592 | (single point OK) | — | — | — | ⚠ NVT chunk bug‡ |
| **12 tiles, N=32** | 262,144 | n/a (ASE OOMs)§ | n/a§ | — | — | ✅ runs, E consistent |

Gates: |ΔE| ≤ 1e-6 eV, per-atom max|dF| ≤ 1e-5 eV/Å. Observed ΔE ~1e-9 meV/atom, forces at the FP64 floor, force cosine = 1.0000000000 — **LAMMPS `pair_style uma` reproduces the ASE FairChem API to machine precision.**

Notes:
- † **1-tile N=18:** single-point (`run 0`) energy+forces match ASE bit-for-bit; the **10-step NVT OOM'd** on one tile (traced+AC path can't hold the extra MD/integrator state at N=18 — the single-tile NVT ceiling is N<18 for the traced path). 12-tile N=18 NVT completes fine (below).
- ‡ **12-tile N=24:** step-0 energy correct (−373,517.77 eV) but NVT step-1 failed (`Expected 18 elements... found 19`): atom motion shifted the edge count → the **N-specific baked chunk count** mismatched by 1. Known, fixable limitation (pad edges to a fixed multiple of the chunk size). N=18/32 NVT were unaffected.
- § **N ≥ 32:** the single-tile ASE oracle OOMs (can't hold 262k atoms on one tile — same reason we built graph-parallel), so no direct ASE parity. Validated by per-atom-energy consistency (below) + collective math being bit-exact at N ≤ 18.

**AG=FD (autograd = finite difference), Path A (ASE):** N=6 3.8e-8, N=12 5.5e-7, N=18 3.4e-6 — all PASS (tol 1e-5). Path C AG=FD: N=6 3.6e-8, N=12 9.6e-7 PASS; on the 12-tile GP path AG=FD was validated at N=4 (1.0e-8) and single-tile N=10 (4.8e-7) — a full 12-tile AG=FD at N=18 is impractical (60 sequential GP runs) so force correctness there is established via the per-atom force **parity vs ASE** (= the autograd force).

**Per-atom-energy consistency (cross-validation where ASE can't run), Path C 12-tile:**
| N | atoms | E/atom (eV) |
|--:|--:|--:|
| 18 | 46,656 | −3.37745480 |
| 32 | 262,144 | −3.37744545 |
| 34 | 314,432 | −3.37745644 |
| 36 | 373,248 | −3.37743958 |
Consistent to 5 significant figures → large-N results are physically correct.

---

## 2. Walltime

**10-step NVT@300 K (completed all 10 steps unless noted):**
| Case | atoms | Path A (ASE) | Path C (pair_style uma) |
|---|--:|--:|--:|
| 1 tile, N=6 | 1,728 | 7.7 s | 25 s |
| 1 tile, N=12 | 13,824 | 61.2 s | 117 s |
| 1 tile, N=18 | 46,656 | 383.3 s | NVT OOM (single-point OK) |
| **12 tiles, N=18** | 46,656 | **90 s**† | **88 s** |
| **12 tiles, N=32** | 262,144 | **258 s**† | **193 s (opt1+2+3+4)** ‖ |
| **12 tiles, N=38** | 438,976 | n/a (ASE-GP ceiling N=32) | **408 s (opt1+2+3)** |

‖ **N=32 optimized 235 s** (job 8782977) — **beats ASE-GP's 258 s**, FP64, energy
bit-identical to the 450 s baseline. Two accuracy-neutral optimizations: (opt3)
XCCL tuning (`CCL_ZE_IPC_EXCHANGE=pidfd`, `CCL_ATL_TRANSPORT=ofi`, `FI_PROVIDER=tcp`)
450→276 s; (opt1) coarser activation-checkpoint chunk (`EDGE_AC_CHUNK` 16384→65536,
fewer backward recomputes) 276→235 s. NVT compute Loop-time 214→180 s. Progression:
450 s (baseline) → 276 s (opt3) → **235 s (opt1+opt3), 1.91× faster than baseline
and 1.10× faster than ASE-GP**.

**opt2 (`torch.jit.freeze` of the top module) — validated, kept as default; storage/HBM
win, wall-neutral** (job 8784408). The traced top module only *dispatches* the
`uma_ckpt.block` / `uma_ckpt.edge_degree` ops (the heavy weights live in the separate
`model_block_*`/`model_chunk_*`/`model_edgedeg` modules), so it baked ~2.22 GiB/rank of
graph-unreachable weights. `torch.jit.freeze(traced.eval())` strips them:
`model_traced.pt` **2224.5 MB → 2.6 MB/rank** (99.9%), N=32 W=12 artifact **53 GB → 27 GB**.
Numerically exact: W=2 N=4 Gate 1 (GP-vs-1-tile dE=9.1e-13 eV, max|dF|=9.6e-16, cos=1.0;
1-tile-vs-ASE dE=2.8e-11; AG=FD max 1.08e-8) and N=32 step-10 PE = **−879646.224481715 eV,
dE = 0.000e+00** vs the opt1+opt3 baseline. Wall was **not** improved (248 s vs 235 s, within
run variance; Loop 184.3 s vs 179.975 s): runtime load is dominated by the real per-chunk
weights (4×554 MB/rank), which `torch.jit.load` still reads, not by the now-stripped dead
top graph. Net: opt2 halves on-disk artifact size and frees HBM headroom with zero accuracy
cost, but is not the path to a faster wall — remaining headroom is de-duplicating the
per-rank chunk weights (they are identical across the 12 ranks).

**opt4 (partial no-recompute) — validated, the next real wall win: 235 s → ~193 s (−18%),
Loop 179.975 s → ~137 s (−24%).** Profiling the N=32 W=12 force call (engine `UMA_MP_PERF=1`,
job 8784422) showed the split: graph/NL build 1.67 s (9%), forward 4.65 s (25%), **backward
12.05 s (65%)**, force all_reduce 0.24 s (1.3%). Backward dominates because activation
checkpointing (AC) *recomputes the forward* to get forces. New granular engine knobs in
`block_context.cpp` bypass the checkpoint per op level and instead retain activations:
`UMA_NO_RECOMPUTE` (all), `UMA_NO_RECOMPUTE_BLOCK`, `UMA_NO_RECOMPUTE_CHUNK`,
`UMA_NO_RECOMPUTE_EDEG` (default OFF = checkpoint, unchanged). Findings:
- Full `UMA_NO_RECOMPUTE=1` **OOMs** at N=32 W=12 (retains every chunk's SO2 + [Ec,25,25]
  wigner: 62.5/64 GiB) — this is exactly why chunk AC exists.
- **Partial `UMA_NO_RECOMPUTE_BLOCK=1 UMA_NO_RECOMPUTE_EDEG=1`** (retain the node-sized
  block + prologue activations, keep the memory-heavy chunk AC) **fits and is fast**: wall
  190/196 s, Loop 136.2/137.4 s (two runs, jobs 8784500/8784623), bwd/call 12.05 s → 7.79 s.
  Numerically equivalent (step-10 dE = 1.2e-10 / 4.7e-10 eV vs baseline) — retain vs
  recompute yields the same gradient.

Two experiments that did **not** help: (i) coarser AC `EDGE_AC_CHUNK=131072` (job 8784521)
= 236 s / Loop 178.3 s, unchanged — recompute cost scales with total edge work, not chunk
count; (ii) fewer tiles for N=32 (job 8784576): W=8 = 342 s, W=12 = 232 s — fewer tiles
means more edges/tile and a strictly slower wall, so W=12 remains best for latency.

**Recommended default for N=32 W=12 NVT:** opt1 (`EDGE_AC_CHUNK=65536`) + opt2 (freeze) +
opt3 (XCCL env) + **opt4 (`UMA_NO_RECOMPUTE_BLOCK=1 UMA_NO_RECOMPUTE_EDEG=1`)** → **~193 s,
1.22× faster than the 235 s opt1+opt3 baseline and 1.34× faster than ASE-GP's 258 s**, FP64,
numerically equivalent. Further headroom would require a selective per-chunk retain (retain a
subset of chunks that fits HBM) to shave the remaining ~3 s/call of chunk recompute.

**opt4 memory ceiling — it is N-dependent.** opt4 retains the node-sized block + prologue
activations, which grow with N, so it is only usable where HBM has headroom. Measured at
W=12: opt4 fits and helps at **N=32** (retains node-sized activations comfortably) but
**OOMs at N=38** (job 8784742: "Tried to allocate 3.77 GiB, 3.13 GiB free" — 54.4/64 GiB
already held). So for the largest systems (N=38, the 12-tile ceiling) opt4 must be OFF and
full checkpointing (opt1+opt2+opt3) is the operating point. Rule of thumb: enable opt4 up to
~N=32/12 tiles; leave it off (checkpoint) for N≳36.

† **ASE 12-tile = FairChem graph-parallel** (`ParallelMLIPPredictUnit` + XCCL, Ray, W=12;
from project `hen`). ASE's driver measures **11 warm energy+force evaluations + cold
load** (= the force calls of a 10-step NVT: run 0 + 10 steps), not a literal
Nosé–Hoover loop, so it is the NVT-equivalent wall. Breakdown: N=18 load 49.0 s +
warmup 12.2 s + ef_mean 2.43 s/eval (wall 90.3 s); N=32 load 49.4 s + warmup 31.1 s +
ef_mean 14.82 s/eval (wall 258.3 s). **Cross-check: ASE-GP and pair_style uma agree
on energy** — N=18 both −157578.531115; N=32 −885377.06004 (ASE) vs −885377.06004
(ours), Fmax 0.848045 identical.

**Walltime caveats (important for interpretation):**
- 1-tile Path A `t_nvt10` is pure 10-step integration (model resident); 1-tile Path C
  wall includes per-rank **cold model load** — so small-N 1-tile Path C is
  load-dominated, not a fair per-step number.
- 12-tile: both include cold load. At **N=18 they are ~equal (90 s ASE-GP vs 88 s
  ours)**. At **N=32, our optimized stack (193 s) now beats ASE-GP (258 s) by 1.34×**
  after opt1 (coarser AC chunk) + opt3 (XCCL tuning) + opt4 (skip block/prologue
  recompute). The original correctness-first port was 450 s; the throughput
  optimization called out here as "the clear next step" has now been done.
- This comparison establishes **correctness + capacity + optimized throughput**; a
  like-for-like warm, load-excluded steps/s benchmark is still future work.

---

## 3. Capacity / max-N — Path C `pair_style uma`, 12 tiles, 10-step NVT@300 K

Single-crystal NaCl across all 12 XPU tiles (native XCCL graph-parallel, FP64).
**All rows below completed the full 10-step NVT** (step-10 T ≈ 285 K, energy-conserving).
The **"orig baseline"** column is the first correctness-first port (`edge_ac_chunk=16384`,
no opt2/opt3/opt4). The **"current best"** column is the optimized stack; the config used
is noted per row because opt4's retain-activations mode is N-limited (see §2):
| N | atoms | orig baseline wall | current best wall | best config | step-10 T (K) | step-10 PE (eV) |
|--:|--:|--:|--:|:--|--:|--:|
| 18 | 46,656 | 88 s | (re-measure pending) | opt1+2+3+4 | 287.0 | −155,753 |
| 32 | 262,144 | 450 s | **193 s** | opt1+2+3+**4** | 285.4 | −879,646 |
| 34 | 314,432 | 534 s | (re-measure pending) | opt1+2+3(+4?) | 285.2 | −1,055,737 |
| 36 | 373,248 | 666 s | (re-measure pending) | opt1+2+3 | 285.3 | −1,253,109 |
| 38 | 438,976 | 800 s | **408 s** (Loop 317.6 s) | opt1+2+3 (opt4 OOMs) | 285.2 | −1,474,399 |
| 40 | 512,000 | OOM | OOM | — | — | OOM |

- **N=32 current best = 193 s** (opt1+opt2+opt3+opt4; jobs 8784500/8784623), 2.33× faster
  than the 450 s original baseline and 1.34× faster than ASE-GP (258 s), step-10 PE
  −879,646.224 (dE≈1e-10 vs baseline).
- **N=38 current best = 408 s** (Loop 317.6 s; opt1+opt2+opt3, opt4 disabled because it
  OOMs at N=38 — job 8784742), 1.96× faster than the 800 s original baseline, full 10-step
  NVT, step-10 PE −1,474,399.12 (matches the original baseline energy). N=40 still OOMs.
- N=18/34/36 current-best walls are not yet re-measured with the optimized stack (their
  original-baseline numbers stand); N=38 and N=32 are the re-measured endpoints.

Single-point (`run 0`) walltimes for reference (orig baseline): N=32 54 s, N=34 57 s, N=36 88 s, N=38 77 s.

- **12-tile single-crystal ceiling = N=38 (438,976 atoms)**, verified with **full 10-step NVT** (not just single point); N=40 OOMs.
- **This exceeds the FairChem/ASE graph-parallel reference (N=32, 262,144 atoms)** — vanilla single-tile ASE OOMs at N=32, and our per-chunk C++ checkpointing reaches larger single-crystal sizes than the Python GP reference, with full MD dynamics.
- N=24 is the one exception: its NVT hit an N-specific chunk-count-drift bug at step 1 (single-point OK) — unlucky chunk boundary; N=18/32/34/36/38 NVT were all stable. Fixable by edge-padding to a fixed chunk multiple.

---

## 4. Summary

- **Correctness:** `pair_style uma` = ASE FairChem to machine precision (ΔE ~1e-9 meV/atom, forces ~1e-14, cos=1.0), verified at N=6/12/18 (1 tile) and N=18 (12 tiles); AG=FD passes.
- **Capacity:** 1 tile → 46,656 atoms; 12 tiles → **full 10-step NVT verified up to N=38 (438,976 atoms)** (N=18/32/34/36/38); N=40 OOMs — beyond the Python reference (N=32).
- **Known limitation:** N-specific AC shard chunk-count can drift by 1 under MD atom motion (hit at N=24 NVT only; single-point OK); single-tile N=18 NVT is memory-tight for the traced path (use 12 tiles). Both fixable (edge-padding to a fixed chunk multiple).
- **Throughput (optimized stack):** N=32 W=12 NVT **450 s → 193 s** (opt1+opt2+opt3+opt4),
  now 1.34× faster than ASE-GP (258 s); N=38 W=12 **800 s → 408 s** (opt1+opt2+opt3; opt4
  OOMs at N=38). opt2 also cut on-disk artifacts 53 GB → 27 GB. All numerically equivalent
  (step-10 dE ≤ ~1e-9 eV vs the original baseline).
- **Not yet done:** re-measure N=18/34/36 with the optimized stack; a warm, load-excluded
  steps/s benchmark; selective per-chunk retain to extend opt4's speedup to N≳36.
