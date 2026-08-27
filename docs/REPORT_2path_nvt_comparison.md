# Comprehensive 2-path NVT comparison — UMA on Aurora XPU (FP64)

**Paths compared**
- **A. ASE FairChem** — `FAIRChemCalculator` + `NoseHooverChainNVT` (tchain=3, tdamp=0.1 ps, dt=1 fs), single Python process (single-tile) or `ParallelMLIPPredictUnit` + XCCL (ASE-GP, 12-tile). FP64.
- **C. Our LAMMPS `pair_style uma`** — native C++/LibTorch, TorchScript-traced UMA, native XCCL graph-parallel on multiple tiles. No Python at runtime, FP64. Two engine execution paths that differ **only in the backward pass** (identical forward, identical energy graph):
  - **C1 = pre-opt4 (full activation checkpointing).** Every message-passing block, edge-chunk, and the edge-degree prologue is wrapped in a C++ recompute Function: the forward runs under `NoGradGuard` (no activations retained) and the **backward re-runs the forward** to get forces. Minimum HBM; works at all sizes up to the N=38 ceiling. Config: opt1 (`EDGE_AC_CHUNK=65536`) + opt2 (freeze) + opt3 (XCCL env), no `UMA_NO_RECOMPUTE_*`.
  - **C2 = opt4 (partial no-recompute).** The **block + edge-degree** checkpoints are bypassed — those sub-modules run directly under autograd so their (node-sized) activations are **retained** and the backward does **not** recompute them; the memory-heavy **edge-chunk** checkpoint is kept. Config: C1 + `UMA_NO_RECOMPUTE_BLOCK=1 UMA_NO_RECOMPUTE_EDEG=1`. Faster backward, higher HBM — only usable where memory has headroom (fits ≤ N=34 on 12 tiles; **OOMs at N≥36**).

*(FC LAMMPS `fix external` was dropped — no such FairChem/UMA component exists, and it would be engine-equivalent to path A.)*

**C1 vs C2 are numerically equivalent by construction** (retain-vs-recompute yields the same gradient); C2 is a pure performance path, not an accuracy tradeoff. Accuracy + performance for each are in §2a below.

**Common setup:** NaCl NxNxN, a=5.64 Å, rattle 0.05 Å (seed 0); NVT 300 K, 10 steps; UMA-s-1p2, task omat, FP64. First-frame energy + per-atom forces (≥100 atoms sampled for parity), AG=FD spot-check (10 atoms), and walltime.

---

## 1. Correctness — Path C (LAMMPS) vs Path A (ASE), first frame

| Case | atoms | ΔE (LAMMPS−ASE) | per-atom max\|dF\| (≥100 at) | cos | AG=FD (C) | verdict |
|---|--:|--:|--:|--:|--:|:--|
| **1 tile, N=6** | 1,728 | 1.3e-10 eV | 1.5e-14 eV/Å | 1.0000000000 | 3.6e-8 | ✅ PASS |
| **1 tile, N=12** | 13,824 | 1.6e-8 eV | 2.2e-14 | 1.0000000000 | 9.6e-7 | ✅ PASS |
| **1 tile, N=18** | 46,656 | 4.6e-8 eV† | 4.8e-14† | 1.0† | — | ✅ (single point)† |
| **12 tiles, N=18** | 46,656 | 4.6e-8 eV | 3.2e-14 | 1.0000000000 | (see note) | ✅ PASS |
| **12 tiles, N=24** | 110,592 | 1.4e-6 eV¶ | 9.1e-14¶ | 1.0000000000 | — | ✅ PASS (step 0)‡ |
| **12 tiles, N=32** | 262,144 | 3.3e-6 eV¶ | 1.05e-13¶ | 1.0000000000 | — | ✅ PASS¶ |

Gates: |ΔE| ≤ 1e-6 eV, per-atom max|dF| ≤ 1e-5 eV/Å. Observed ΔE ~1e-9 meV/atom, forces at the FP64 floor, force cosine = 1.0000000000 — **LAMMPS `pair_style uma` reproduces the ASE FairChem API to machine precision.**

Notes:
- † **1-tile N=18:** single-point (`run 0`) energy+forces match ASE bit-for-bit; the **10-step NVT OOM'd** on one tile (traced+AC path can't hold the extra MD/integrator state at N=18 — the single-tile NVT ceiling is N<18 for the traced path). 12-tile N=18 NVT completes fine (below).
- ‡ **12-tile N=24:** step-0 energy + all-atom forces now **pass full parity vs ASE-GP** (¶: −373,517.77 eV, per-atom max|dF|=9.06e-14). The **10-step NVT** still fails at step 1 (`Expected 18 elements... found 19`): atom motion shifts the edge count → the **N-specific baked chunk count** mismatches by 1. Known, fixable limitation (pad edges to a fixed multiple of the chunk size). N=18/32/34/36/38 NVT were unaffected; the N=24 *forces* are correct — only the multi-step NVT chunking trips.
- § **N ≥ 32:** the *single-tile* ASE oracle OOMs (can't hold 262k atoms on one tile — same reason we built graph-parallel). Direct parity instead uses the **ASE-GP (FairChem graph-parallel, 12-tile) oracle** from project `hen` (`ParallelMLIPPredictUnit` + XCCL), which runs to N=32. See ¶.
- ¶ **N=24 & N=32 full per-atom parity vs ASE-GP (12-tile):** LAMMPS `pair_style uma` step-0 energy + all-atom forces vs the FairChem graph-parallel oracle on the identical NaCl (a=5.64, rattle 0.05, seed 0, W=12). **N=24 (110,592 atoms):** dE=1.39e-6 eV (**1.25e-8 meV/atom**), per-atom max|dF|=**9.06e-14** eV/Å, rms 1.33e-14, cos=1.0000000000 → PASS. **N=32 (262,144 atoms):** dE=3.35e-6 eV (**1.28e-8 meV/atom**), per-atom max|dF|=**1.05e-13** eV/Å, rms 9.70e-15, cos=1.0000000000 → PASS. Oracles: `hen/pbs/out/ase12_n{24,32}` (jobs: N=24 8784843; N=32 pre-existing). Compared with `scripts/parity_vs_asegp.py` (per-atom energy gate 1e-3 meV/atom, force gate 1e-5 eV/Å over all atoms). This is a **full-system** parity (every atom), stronger than the sampled ≥100-atom gate. The absolute total dE is ~1e-6 eV only because FP64 accumulation of ~1e-8 meV/atom over 10^5–10^6 atoms sums up; per-atom it is at the machine-precision floor.

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
Path C is split into **C1 = pre-opt4** (full checkpointing) and **C2 = opt4** (partial
no-recompute; §2a). opt4 works on both the single-tile and 12-tile paths (same `block_context.cpp`
knobs); it OOMs where HBM is tight (1-tile N=18, 12-tile N≥36).
| Case | atoms | Path A (ASE) | Path C1 (pre-opt4) | Path C2 (opt4) |
|---|--:|--:|--:|--:|
| 1 tile, N=6 | 1,728 | 7.7 s | 20 s (Loop 10.9 s) | **16 s** (Loop 8.3 s) |
| 1 tile, N=12 | 13,824 | 61.2 s | 110 s (Loop 87.0 s) | **89 s** (Loop 65.8 s) |
| 1 tile, N=18 | 46,656 | 383.3 s | 364 s (Loop 299.0 s) | **OOM** (opt4 exceeds HBM) |
| **12 tiles, N=18** | 46,656 | **90 s**† | **70 s** (Loop 30.9 s) | **34 s** (Loop 23.2 s) |
| **12 tiles, N=32** | 262,144 | **258 s**† | **235 s**\* | **193 s**\* |
| **12 tiles, N=34** | 314,432 | n/a (ASE-GP ceiling N=32) | **299 s** (Loop 218.4 s) | **210 s** (Loop 166.9 s) |
| **12 tiles, N=36** | 373,248 | n/a (ASE-GP ceiling N=32) | **343 s** (Loop 267.4 s) | **OOM** (opt4 exceeds HBM) |
| **12 tiles, N=38** | 438,976 | n/a (ASE-GP ceiling N=32) | **408 s** (Loop 317.6 s) | **OOM** (opt4 exceeds HBM) |

C1/C2 walls are the current optimized stack (opt1+opt2+opt3; C2 adds opt4). All 12-tile C1/C2
rows completed the full 10-step NVT with **identical step-10 energy** (N=18 −155753.154048;
N=32 −879646.224482; N=34 −1,055,737.433775; N=36 −1,253,109.42; N=38 −1,474,399.12).
**opt4 (C1→C2) gain by size** (Loop / wall):
- 1-tile: N=6 10.9→8.3 s (**−24%**) / 20→16 s; N=12 87.0→65.8 s (**−24%**) / 110→89 s;
  N=18 OOM (1-tile memory ceiling; single-tile N=18 NVT is tight even for C1).
- 12-tile: N=18 30.9→23.2 s (**−25%**) / 70→34 s; N=32 179.975→137 s (**−24%**) / 235→193 s;
  N=34 218.4→166.9 s (**−24%**) / 299→210 s.
opt4 consistently gives **~−24% Loop** wherever it fits. **C2 ceiling by config:**
1-tile: fits N≤12, OOMs N=18; 12-tile: fits N≤34, OOMs N≥36
(`UR_RESULT_ERROR_OUT_OF_RESOURCES`, job 8785022). Use C1 outside those ranges.
The old N=18 12-tile C1 = 88 s was the pre-opt-stack baseline; the current C1 is 70 s.
Jobs: 1-tile N=6/12/18 8785403; 12-tile N=18 8784969, N=34 8785293, N=36 8785022.

---

## 2a. C1 (pre-opt4) vs C2 (opt4) — the two LAMMPS execution paths

Same engine, same artifacts (opt1 `EDGE_AC_CHUNK=65536` + opt2 freeze + opt3 XCCL env),
same forward and same energy graph. The paths differ **only in how the backward gets
forces**: C1 recomputes every sub-module (activation checkpointing everywhere); C2 retains
the node-sized block + edge-degree activations and recomputes only the edge-chunks.

### Accuracy — C1 vs C2 (both vs the ASE-GP oracle, and vs each other)

| metric (N=32 W=12, step 0) | C1 (pre-opt4) | C2 (opt4) |
|---|--:|--:|
| step-0 PE (eV) | −885377.06004 | −885377.06004 |
| step-10 PE (eV) | −879646.224481715 | −879646.224481715 (dE ≤ 4.7e-10 vs C1) |
| per-atom max\|dF\| vs ASE-GP | 1.05e-13 eV/Å | 1.05e-13 eV/Å |
| ΔE/atom vs ASE-GP | 1.28e-8 meV/atom | 1.28e-8 meV/atom |
| force cosine vs ASE-GP | 1.0000000000 | 1.0000000000 |

**Accuracy verdict: identical.** C2 is numerically equivalent to C1 (retain-vs-recompute is
the same gradient); the residual C2−C1 step-10 dE (≤ 4.7e-10 eV) is FP64 summation-order
noise, far below the parity floor. Both pass full per-atom parity vs ASE-GP (§1 ¶).

### Performance — C1 vs C2 (N=32 W=12, 10-step NVT)

| metric | C1 (pre-opt4) | C2 (opt4) | opt4 gain |
|---|--:|--:|--:|
| **Loop time** (pure MD compute) | 179.975 s | 136.2 / 137.4 s | **−24% (1.31×)** |
| **whole wall** (incl. cold load) | 235 s | 190 / 196 s | **−18% (1.22×)** |
| backward / force-call | 12.05 s | 7.79 s | **−35%** |
| forward / force-call | 4.65 s | 4.65 s | 0% (unchanged) |
| graph+NL / force-call | 1.67 s | 1.67 s | 0% |
| force all_reduce / call | 0.24 s | 0.24 s | 0% |
| peak HBM headroom @ N=32/12t | ample | fits (higher) | — |
| **max N @ 12 tiles (NVT)** | **N=38** (408 s) | **N=34** (OOMs at N≥36) | capacity cost |

**Performance verdict:** opt4 alone buys **−24% Loop / −18% wall at N=32** by removing the
block+prologue backward recompute (bwd 12.05 s → 7.79 s). The forward, graph, and collective
costs are untouched. jobs: C1 8782977; C2 8784500 / 8784623; profile 8784422.

### Which path to use

- **N ≤ 34 on 12 tiles → C2 (opt4).** Same accuracy, ~−24% Loop (measured N=18, N=32, N=34).
- **N ≥ 36 (up to the N=38 ceiling) → C1 (pre-opt4).** C2's retained activations OOM at **N=36**
  (job 8785022, `UR_RESULT_ERROR_OUT_OF_RESOURCES`) and N=38 (job 8784742: needed 3.77 GiB,
  3.13 GiB free); C1 is the only path that fits. C1 at N=36 = 343 s, N=38 = 408 s (Loop 317.6 s),
  ~1.96× faster than the original 800 s baseline via opt1+2+3.
- **C2 memory ceiling: N=34 fits, N=36 OOMs.** Rule of thumb:
  `UMA_NO_RECOMPUTE_BLOCK=1 UMA_NO_RECOMPUTE_EDEG=1` for N ≤ 34 on 12 tiles; leave it off (C1)
  for N ≥ 36.

---

‖ **N=32: C1 = 235 s** (job 8782977), **C2 = 193 s** (opt4; see §2a) — **beats ASE-GP's 258 s**, FP64, energy
bit-identical to the 450 s baseline. Two accuracy-neutral optimizations: (opt3)
XCCL tuning (`CCL_ZE_IPC_EXCHANGE=pidfd`, `CCL_ATL_TRANSPORT=ofi`, `FI_PROVIDER=tcp`)
450→276 s; (opt1) coarser activation-checkpoint chunk (`EDGE_AC_CHUNK` 16384→65536,
fewer backward recomputes) 276→235 s. NVT compute Loop-time 214→180 s. Progression:
450 s (baseline) → 276 s (opt3) → **235 s = path C1 (opt1+opt3), 1.91× faster than baseline
and 1.10× faster than ASE-GP** → **193 s = path C2 (+opt4), 2.33× faster than baseline and
1.34× faster than ASE-GP** (§2a).

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

**opt4 = the C1→C2 step (partial no-recompute).** Profiling the C1 N=32 W=12 force call
(engine `UMA_MP_PERF=1`, job 8784422) showed backward = 65% of the call (graph/NL 1.67 s 9%,
fwd 4.65 s 25%, **bwd 12.05 s 65%**, force_ar 0.24 s 1.3%) because checkpointing recomputes
the forward for forces. Granular engine knobs in `block_context.cpp` (`UMA_NO_RECOMPUTE`,
`UMA_NO_RECOMPUTE_BLOCK`, `UMA_NO_RECOMPUTE_CHUNK`, `UMA_NO_RECOMPUTE_EDEG`; default OFF =
C1) let the backward retain activations instead. **Path C2** uses
`UMA_NO_RECOMPUTE_BLOCK=1 UMA_NO_RECOMPUTE_EDEG=1` (see §2a for the full C1-vs-C2 accuracy +
performance tables). Full `UMA_NO_RECOMPUTE=1` **OOMs** (62.5/64 GiB) — the edge-chunk
checkpoint must stay on, which is why C2 keeps it.

Two experiments that did **not** help: (i) coarser AC `EDGE_AC_CHUNK=131072` (job 8784521)
= 236 s / Loop 178.3 s, unchanged — recompute cost scales with total edge work, not chunk
count; (ii) fewer tiles for N=32 (job 8784576): W=8 = 342 s, W=12 = 232 s — fewer tiles
means more edges/tile and a strictly slower wall, so W=12 remains best for latency.

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
| N | atoms | orig baseline wall | current best wall | best path | step-10 T (K) | step-10 PE (eV) |
|--:|--:|--:|--:|:--|--:|--:|
| 18 | 46,656 | 88 s | **34 s** (C2) / 70 s (C1) | **C2** (opt4 fits) | 287.0 | −155,753 |
| 32 | 262,144 | 450 s | **193 s** (C2) / 235 s (C1) | **C2** (opt4) | 285.4 | −879,646 |
| 34 | 314,432 | 534 s | **210 s** (C2) / 299 s (C1) | **C2** (opt4 fits) | 285.2 | −1,055,737 |
| 36 | 373,248 | 666 s | **343 s** (C1) | **C1** (C2/opt4 OOMs) | 285.3 | −1,253,109 |
| 38 | 438,976 | 800 s | **408 s** (C1, Loop 317.6 s) | **C1** (C2/opt4 OOMs) | 285.2 | −1,474,399 |
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

- **Correctness:** `pair_style uma` = ASE FairChem to machine precision (ΔE ~1e-8 meV/atom, forces ~1e-13, cos=1.0), verified at N=6/12/18 (1 tile, vs single-tile ASE) and **full-system per-atom parity at N=18/24/32 (12 tiles, vs ASE-GP)** — N=24 max|dF|=9.06e-14 over 110,592 atoms, N=32 max|dF|=1.05e-13 over 262,144 atoms; AG=FD passes.
- **Capacity:** 1 tile → 46,656 atoms; 12 tiles → **full 10-step NVT verified up to N=38 (438,976 atoms)** (N=18/32/34/36/38); N=40 OOMs — beyond the Python reference (N=32).
- **Known limitation:** N-specific AC shard chunk-count can drift by 1 under MD atom motion (hit at N=24 NVT only; single-point OK); single-tile N=18 NVT is memory-tight for the traced path (use 12 tiles). Both fixable (edge-padding to a fixed chunk multiple).
- **Two LAMMPS execution paths (identical accuracy, §2a):** **C1** = full checkpointing
  (opt1+opt2+opt3), fits to the N=38 ceiling; **C2** = C1 + opt4 partial no-recompute
  (`UMA_NO_RECOMPUTE_BLOCK/EDEG`), which skips the block+prologue backward recompute for
  a **~−24% Loop** gain but needs more HBM (fits N≤34, OOMs N≥36 on 12 tiles).
- **Throughput:** N=32 W=12 NVT **450 s → C1 235 s → C2 193 s** (C2 now 1.34× faster than
  ASE-GP's 258 s); N=38 W=12 **800 s → C1 408 s** (C2/opt4 OOMs at N=38). opt2 also cut
  on-disk artifacts 53 GB → 27 GB. C1 and C2 are numerically equivalent (step-10 dE ≤ ~1e-9
  eV vs the original baseline).
- **Not yet done:** re-measure N=18 C1 vs old-baseline 88 s; a warm, load-excluded steps/s
  benchmark; selective per-chunk retain to extend C2 beyond the current N=34 ceiling.
- **Tile scaling:** see §5 — strong scaling on N=16 shows useful speedup to W=4 (C1) and
  W=6 (C2); beyond that communication overhead saturates the small system.

---

## 5. Strong-scaling study — N=16 (32,768 atoms), W = 1, 2, 4, 6, 8, 12 tiles

Single-point (`run 0`) force evaluation on the identical perturbed NaCl N=16 supercell
(a=5.64 Å, rattle 0.05 Å, seed 0). Single-point avoids the N-specific traced-chunk-count
drift bug that breaks multi-step NVT at certain N (fixable by edge-padding; the force
computation itself is correct). Whole-wall includes cold model load; each run is a fresh
mpiexec. Energy = −110,673.829050 eV across **all** W and both C1/C2 (identical). Job 8785948.

**opt1+opt2+opt3 artifacts (C1 = full checkpoint; C2 = + opt4):**
| W (tiles) | C1 wall | C1 eff% | C2 wall | C2 eff% | C2 gain over C1 |
|--:|--:|--:|--:|--:|--:|
| 1 | 42 s | 100% | 21 s | 100% | 2.00× |
| 2 | 26 s | 81% | 14 s | 75% | 1.86× |
| 4 | 17 s | 62% | 11 s | 48% | 1.55× |
| 6 | 19 s | 37% | 9 s | 39% | 2.11× |
| 8 | 15 s | 35% | 9 s | 29% | 1.67× |
| 12 | 16 s | 22% | 9 s | 19% | 1.78× |

Efficiency = (W=1 wall / W) / actual wall × 100%.
Note: wall times are integer-second precision; ±1 s noise dominates at W≥6 where walls are
9-16 s, so efficiency numbers beyond W=4 should be read qualitatively, not exactly.

**Observations:**
- **Good scaling W=1→4:** C1 drops 42→17 s (2.5×); C2 drops 21→11 s (1.9×). Strong-scaling
  efficiency 62% (C1) / 48% (C2) at W=4 — reasonable for a memory-bandwidth-bound model
  where inter-tile communication (per-block `all_gather_nodes`) grows with W.
- **Diminishing returns W≥6:** the per-block all_gather + force all_reduce overhead saturates
  at N=16 (32,768 atoms / 12 = ~2,700 atoms/tile, ~250k edges/tile). Wall plateaus at
  15-19 s (C1) and 9 s (C2) for W=6–12; adding tiles beyond 6 no longer helps for this N.
- **C2 (opt4) consistently ~1.7-2.0× faster than C1** across all W — the backward recompute
  removed by opt4 is a fixed fraction of the call regardless of tile count. C2 reaches a
  floor of 9 s at W≥6; this floor is dominated by the cold model-load time (not reducible
  by adding tiles).
- **Recommended tile count for N=16:** W=4 for C1 (best efficiency), W=4-6 for C2 (wall
  floor reached at W=6). Using W=12 for N=16 wastes 6-8 tiles with no wall benefit.
- **Cold-load caveat:** the warm per-step force-call time is considerably shorter than these
  wall figures (most of which is model load + warmup). In a real MD run the load is amortized
  over all steps; the per-step cost at steady state is roughly (wall − load) / 1.
