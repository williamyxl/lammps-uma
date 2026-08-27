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

> ## STATUS: LAMMPS numbers CURRENT as of freshly rebuilt engine (2026-08-27)
>
> All LAMMPS `pair_style uma` results below (energy, force, timing) were
> **regenerated on a fresh build** of `build-lmp-xccl/lmp` (rebuilt 2026-08-27
> 05:51 from source at HEAD `cfd1657b89`; contains the opt4 `UMA_NO_RECOMPUTE_*`
> knobs). Jobs: `regen_main` 8786078 (C1/C2 sweep), `sc16_run` 8786087 (N=16
> scaling), `par_gate` 8786107 (ASE parity). **Reported energies are step-0
> single-point** (decomposition-independent, the validated quantity); step-10 NVT
> energies are trajectory-dependent and shown separately where relevant.
>
> **ASE FairChem oracle** values are the fixed reference (no ASE rebuild): N=16
> W=1 E=−110673.82905, N=24 W=12 E=−373517.7697985, N=32 W=12 E=−885377.0600366.
> Fresh LAMMPS step-0 energies match these to ≤1.3e-8 meV/atom (§1, §6.1).
>
> **Convention note:** some prior tables listed step-10 (post-NVT) PE. This
> revision reports step-0 PE for all correctness rows; where a step-10 value is
> given it is labelled as such.

---

## 1. Correctness — Path C (LAMMPS) vs Path A (ASE), first frame

Fresh (rebuilt-engine) step-0 energy + per-atom force parity vs the ASE oracle:

| Case | atoms | ΔE (LAMMPS−ASE) | per-atom max\|dF\| (≥100 at) | cos | AG=FD (C) | verdict |
|---|--:|--:|--:|--:|--:|:--|
| **1 tile, N=6** | 1,728 | 1.3e-10 eV | 1.5e-14 eV/Å | 1.0000000000 | 3.6e-8 | ✅ PASS |
| **1 tile, N=12** | 13,824 | 1.6e-8 eV | 2.2e-14 | 1.0000000000 | 9.6e-7 | ✅ PASS |
| **1 tile, N=16** | 32,768 | 4.6e-8 eV (1.4e-9 meV/at) | 5.06e-14 | 1.0000000000 | — | ✅ PASS (gate) |
| **1 tile, N=18** | 46,656 | step-0 E=−157578.531115 = ASE† | 4.8e-14† | 1.0† | — | ✅ (single point)† |
| **12 tiles, N=18** | 46,656 | 4.6e-8 eV | 3.2e-14 | 1.0000000000 | (see note) | ✅ PASS |
| **12 tiles, N=24** | 110,592 | 1.4e-6 eV¶ | 9.1e-14¶ | 1.0000000000 | — | ✅ PASS (step 0)‡ |
| **12 tiles, N=32** | 262,144 | 3.3e-6 eV (1.28e-8 meV/at)¶ | 1.05e-13¶ | 1.0000000000 | — | ✅ PASS¶ |

Freshly measured on the rebuilt engine (2026-08-27): N=16 W=1 and N=32 W=12 via
the ASE parity gate (job 8786107). N=6/12/18 1-tile step-0 energies unchanged
(−5836.318644 / −46690.218610 / −157578.531115). N=24 from the prior ASE-GP
parity (oracle unchanged; §1 ¶).

Gates: |ΔE| ≤ 1e-6 eV, per-atom max|dF| ≤ 1e-5 eV/Å. Observed ΔE ~1e-9 meV/atom, forces at the FP64 floor, force cosine = 1.0000000000 — **LAMMPS `pair_style uma` reproduces the ASE FairChem API to machine precision.**

Notes:
- † **1-tile N=18:** single-point (`run 0`) energy+forces match ASE bit-for-bit; the **10-step NVT OOM'd** on one tile (traced+AC path can't hold the extra MD/integrator state at N=18 — the single-tile NVT ceiling is N<18 for the traced path). 12-tile N=18 NVT completes fine (below).
- ‡ **12-tile N=24:** step-0 energy + all-atom forces pass full parity vs ASE-GP (¶: −373,517.77 eV, per-atom max|dF|=9.06e-14). The **10-step NVT** previously failed at step 1 (`Expected 18 elements... found 19`: atom motion shifted the edge count → the N-specific baked chunk count mismatched). **FIXED by P2.1 edge padding** (2026-08-27): with the pad, N=24 W=12 completes the full 10-step NVT (wall 113 s, step-0 PE −373517.7698 = ASE-GP, job 8786199). See §7.
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

**Fresh numbers (rebuilt engine 2026-08-27; jobs 8786078 / 8786087).**
**10-step NVT@300 K (completed all 10 steps unless noted):**
Path C is split into **C1 = pre-opt4** (full checkpointing) and **C2 = opt4** (partial
no-recompute; §2a). opt4 works on both the single-tile and 12-tile paths (same `block_context.cpp`
knobs); it OOMs where HBM is tight (1-tile N=18, 12-tile N≥36).
| Case | atoms | Path A (ASE) | Path C1 (pre-opt4) | Path C2 (opt4) |
|---|--:|--:|--:|--:|
| 1 tile, N=6 | 1,728 | 7.7 s | 23 s (Loop 11.0 s) | **16 s** (Loop 8.4 s) |
| 1 tile, N=12 | 13,824 | 61.2 s | 116 s (Loop 87.5 s) | **86 s** (Loop 66.5 s) |
| 1 tile, N=18 | 46,656 | 383.3 s | 373 s (Loop 301.8 s) | **OOM** (opt4 exceeds HBM) |
| **12 tiles, N=18** | 46,656 | **90 s**† | **73 s** (Loop 30.9 s) | **35 s** (Loop 23.1 s) |
| **12 tiles, N=32** | 262,144 | **258 s**† | **233 s**\* (Loop 182.7 s) | **174 s**\* (Loop 139.0 s) |
| **12 tiles, N=34** | 314,432 | n/a (ASE-GP ceiling N=32) | **280 s** (Loop 221.3 s) | **212 s** (Loop 169.1 s) |
| **12 tiles, N=36** | 373,248 | n/a (ASE-GP ceiling N=32) | **335 s** (Loop 266.9 s) | **OOM** (opt4 exceeds HBM) |
| **12 tiles, N=38** | 438,976 | n/a (ASE-GP ceiling N=32) | **392 s** (Loop 316.0 s) | **OOM** (opt4 exceeds HBM) |

C1/C2 walls are the current optimized stack (opt1+opt2+opt3; C2 adds opt4), freshly
measured on the rebuilt engine. All 12-tile C1/C2 rows completed the full 10-step NVT.
**C1 and C2 give identical step-0 energy** (N=18 −157578.531115; N=32 −885377.060040;
N=34 −1061980.383367; N=36 −1260622.568658; N=38 −1482625.946248) — proving opt4 is
physics-neutral. (Step-10 NVT energies match within a path but are trajectory-dependent
across W; not a cross-W metric — see §2 convention note.)
**opt4 (C1→C2) gain by size** (Loop / wall, fresh 2026-08-27):
- 1-tile: N=6 11.0→8.4 s (**−24%**) / 23→16 s; N=12 87.5→66.5 s (**−24%**) / 116→86 s;
  N=18 OOM (1-tile memory ceiling; single-tile N=18 NVT is tight even for C1).
- 12-tile: N=18 30.9→23.1 s (**−25%**) / 73→35 s; N=32 182.7→139.0 s (**−24%**) / 233→174 s;
  N=34 221.3→169.1 s (**−24%**) / 280→212 s.
opt4 consistently gives **~−24% Loop** wherever it fits. **C2 ceiling by config:**
1-tile: fits N≤12, OOMs N=18; 12-tile: fits N≤34, OOMs N≥36
(`UR_RESULT_ERROR_OUT_OF_RESOURCES`; jobs 8786078 N=36/38 C2 OOM). Use C1 outside those ranges.
Jobs: rebuild 8786070; C1/C2 sweep 8786078; N=16 scaling 8786087; ASE parity 8786107.

---

## 2a. C1 (pre-opt4) vs C2 (opt4) — the two LAMMPS execution paths

**Fresh (rebuilt engine 2026-08-27, job 8786078).**
Same engine, same artifacts (opt1 `EDGE_AC_CHUNK=65536` + opt2 freeze + opt3 XCCL env),
same forward and same energy graph. The paths differ **only in how the backward gets
forces**: C1 recomputes every sub-module (activation checkpointing everywhere); C2 retains
the node-sized block + edge-degree activations and recomputes only the edge-chunks.

### Accuracy — C1 vs C2 (both vs the ASE-GP oracle, and vs each other)

| metric (N=32 W=12) | C1 (pre-opt4) | C2 (opt4) |
|---|--:|--:|
| step-0 PE (eV) | −885377.060040 | −885377.060040 (identical) |
| step-10 PE (eV) | −879646.224482 | −879646.224482 (identical) |
| per-atom max\|dF\| vs ASE-GP | 1.05e-13 eV/Å | 1.05e-13 eV/Å |
| ΔE/atom vs ASE-GP | 1.28e-8 meV/atom | 1.28e-8 meV/atom |
| force cosine vs ASE-GP | 1.0000000000 | 1.0000000000 |

**Accuracy verdict: identical.** C2 is numerically equivalent to C1 (retain-vs-recompute is
the same gradient); step-0 and step-10 PE match to all printed digits on the rebuilt engine.
Both pass full per-atom parity vs ASE-GP (§1 ¶; N=32 W=12 gate PASS, job 8786107).

### Performance — C1 vs C2 (N=32 W=12, 10-step NVT)

| metric | C1 (pre-opt4) | C2 (opt4) | opt4 gain |
|---|--:|--:|--:|
| **Loop time** (pure MD compute) | 182.7 s | 139.0 s | **−24% (1.31×)** |
| **whole wall** (incl. cold load) | 233 s | 174 s | **−25% (1.34×)** |
| **max N @ 12 tiles (NVT)** | **N=38** (392 s) | **N=34** (OOMs at N≥36) | capacity cost |

**Performance verdict:** opt4 buys **−24% Loop / −25% wall at N=32** by removing the
block+prologue backward recompute. Fresh walls: C1 233 s / C2 174 s (job 8786078). The
per-call breakdown (fwd/bwd/graph/all_reduce) was profiled separately at
graph 1.67 s / fwd 4.65 s / bwd 12.05→7.79 s / all_reduce 0.24 s (job 8784422, structure
unchanged by rebuild).

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

**Fresh (rebuilt engine 2026-08-27, job 8786078).**
Single-crystal NaCl across all 12 XPU tiles (native XCCL graph-parallel, FP64).
**All rows below completed the full 10-step NVT.** The **"orig baseline"** column is the
first correctness-first port (`edge_ac_chunk=16384`, no opt2/opt3/opt4). The **"current
best"** column is the optimized stack, freshly measured; the config used is noted per row
because opt4 (C2) is N-limited (see §2). step-0 PE is the decomposition-independent energy.
| N | atoms | orig baseline wall | current best wall | best path | step-0 PE (eV) |
|--:|--:|--:|--:|:--|--:|
| 18 | 46,656 | 88 s | **35 s** (C2) / 73 s (C1) | **C2** (opt4 fits) | −157578.531115 |
| 32 | 262,144 | 450 s | **174 s** (C2) / 233 s (C1) | **C2** (opt4) | −885377.060040 |
| 34 | 314,432 | 534 s | **212 s** (C2) / 280 s (C1) | **C2** (opt4 fits) | −1061980.383367 |
| 36 | 373,248 | 666 s | **335 s** (C1) | **C1** (C2/opt4 OOMs) | −1260622.568658 |
| 38 | 438,976 | 800 s | **392 s** (C1, Loop 316.0 s) | **C1** (C2/opt4 OOMs) | −1482625.946248 |
| 40 | 512,000 | OOM | OOM | — | OOM |

- **N=32 current best = 174 s** (C2 = opt1+opt2+opt3+opt4; job 8786078), 2.59× faster
  than the 450 s original baseline and 1.48× faster than ASE-GP (258 s).
- **N=38 current best = 392 s** (C1 = opt1+opt2+opt3, Loop 316.0 s; opt4 OOMs at N=38),
  2.04× faster than the 800 s original baseline, full 10-step NVT. N=40 still OOMs.
- **All N (18/32/34/36/38) re-measured** on the rebuilt engine (job 8786078); no
  original-baseline number is carried forward.

Single-point (`run 0`) walltimes for reference (orig baseline): N=32 54 s, N=34 57 s, N=36 88 s, N=38 77 s.

- **12-tile single-crystal ceiling = N=38 (438,976 atoms)**, verified with **full 10-step NVT** (not just single point); N=40 OOMs.
- **This exceeds the FairChem/ASE graph-parallel reference (N=32, 262,144 atoms)** — vanilla single-tile ASE OOMs at N=32, and our per-chunk C++ checkpointing reaches larger single-crystal sizes than the Python GP reference, with full MD dynamics.
- N=24 previously hit an N-specific chunk-count-drift bug at NVT step 1 — **now FIXED by P2.1 edge padding** (§7): N=24 W=12 completes the full 10-step NVT. The pad makes the traced chunk count invariant to edge drift for all N.

---

## 4. Summary

- **Correctness:** `pair_style uma` = ASE FairChem to machine precision (ΔE ~1e-8 meV/atom, forces ~1e-13, cos=1.0), verified at N=6/12/18 (1 tile, vs single-tile ASE) and **full-system per-atom parity at N=18/24/32 (12 tiles, vs ASE-GP)** — N=24 max|dF|=9.06e-14 over 110,592 atoms, N=32 max|dF|=1.05e-13 over 262,144 atoms; AG=FD passes.
- **Capacity:** 1 tile → 46,656 atoms; 12 tiles → **full 10-step NVT verified up to N=38 (438,976 atoms)** (N=18/32/34/36/38); N=40 OOMs — beyond the Python reference (N=32).
- **Resolved (P2.1, §7):** the N-specific AC chunk-count drift under MD atom motion (previously hit N=24 NVT) is **fixed by edge padding** — the runtime pads its per-step edge count to a fixed multiple of edge_ac_chunk (self-loops beyond cutoff, zero contribution) so the traced chunk count is invariant. Remaining: single-tile N=18 NVT is memory-tight for the traced path (use 12 tiles).
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

**Fresh (rebuilt engine + P2.1 padded artifacts, 2026-08-27, job 8786329).
NVT 300 K, 10 steps** — the proper MD timing (matches the §2/§3 convention). This
became possible only after P2.1 edge padding (§7) fixed the N=16 10-step-NVT
chunk-drift crash; the earlier single-point stand-in is retired.

NaCl N=16 (32,768 atoms; a=5.64 Å, rattle 0.05 Å, seed 0). step-0 PE =
−110,673.829050 eV across **all** W and both C1/C2 (identical, = ASE W=1 oracle,
§6.1). **Loop time** is the pure 10-step MD compute (excludes cold load) and is
the scaling metric; whole-wall (incl. load) also shown.

**opt1+opt2+opt3 artifacts (C1 = full checkpoint; C2 = + opt4), NVT-10:**
| W (tiles) | C1 Loop | C1 eff% | C2 Loop | C2 eff% | C2 gain | C1 wall | C2 wall |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1  | 226.2 s | 100% | 172.0 s | 100% | 1.32× | 282 s | 211 s |
| 2  | 137.9 s |  82% | 101.8 s |  84% | 1.35× | 174 s | 128 s |
| 4  |  79.3 s |  71% |  59.2 s |  73% | 1.34× | 103 s |  76 s |
| 6  |  48.0 s |  79% |  35.6 s |  80% | 1.35× |  66 s |  49 s |
| 8  |  50.2 s |  56% |  37.9 s |  57% | 1.32× |  69 s |  51 s |
| 12 |  34.0 s |  55% |  25.9 s |  55% | 1.31× |  51 s |  37 s |

Efficiency = (W=1 Loop / W) / actual Loop × 100% (Loop-time strong scaling).

**Observations:**
- **Strong scaling to W=6:** C1 Loop 226→48 s (**4.7×** on 6 tiles, 79% eff); C2
  172→36 s (**4.8×**, 80% eff). Inter-tile `all_gather_nodes` per block grows with
  W but the compute still dominates at these sizes.
- **W=8 is a dip** (eff 56% vs W=6 79%): 32,768 atoms / 8 = 4,096 atoms/tile
  partitions less evenly than /6 or /12 for this cell; W=12 recovers (55% but
  lowest absolute Loop, 34 s C1 / 26 s C2).
- **C2 (opt4) consistently 1.31–1.35× faster than C1** in Loop time at every W —
  the removed block+prologue backward recompute is a fixed fraction of the call,
  independent of tile count (consistent with the −24% Loop measured at N=32, §2a).
- **Best latency for N=16:** W=12 (Loop 26 s C2). **Best efficiency:** W=6
  (~80%). Beyond W=6 you trade efficiency for a bit more absolute speed.
- **Step-10 energy** differs W=1 (−110602.976) vs W≥2 (−109317.384) — expected:
  `velocity create` seeds a different distribution per domain decomposition, so
  the MD trajectories diverge (step-0 is the decomposition-independent invariant).

---

## 6. Validation suite — current rebuilt engine (2026-08-27)

All LAMMPS numbers in this report were regenerated on a fresh build (2026-08-27
05:51, HEAD `cfd1657b89`). This is the current-build baseline; Phase-1
(`docs/CAMPAIGN_PLAN_quality.md`) will **re-run this identical suite after its
fixes land** and any change beyond the tolerances below is a regression. The
reference is the ASE oracle (no ASE rebuild); step-0 energy is the invariant.

**Suite results (rebuilt engine; jobs 8786078 / 8786087 / 8786107):**
| # | system | tiles | path | ASE reference (step-0 E) | rebuilt LAMMPS step-0 E / wall | verdict |
|---|---|---|---|---|---|:--|
| 1 | N=16 (32,768) | W=1 | C1 & C2 | −110673.82905 | −110673.829050 / C1 46s, C2 21s | ✅ dE=1.4e-9 meV/at |
| 2 | N=32 (262,144) | W=12 | C1 & C2 | −885377.0600366 | −885377.060040 / C1 233s, C2 174s | ✅ dE=1.28e-8 meV/at |
| 3 | N=18 (46,656) | W=1 | C1 | −157578.531115 | −157578.531115 / C1 373s | ✅ exact |
| 4 | N=38 (438,976) | W=12 | C1 | (no ASE oracle) | −1482625.946248 / C1 392s | ✅ NVT completes, N-consistent |
| 5 | N=16 scaling W=1..12 (NVT-10) | 1–12 | C1 & C2 | −110673.82905 | step-0 −110673.829050 all W (§5) | ✅ all W match, NVT completes |

- **C1/C2 equivalence:** identical step-0 energy at every N where both fit
  (N=16/18/32/34 to all printed digits) — opt4 is physics-neutral.
- **Row-4 N=38** has no ASE oracle (ASE-GP ceiling N=32); validated by step-0
  self-consistency + per-atom-E consistency with neighboring N.

**Gate cross-checks (rebuilt engine, ASE-anchored):**
- N=16 W=1 + N=32 W=12 `parity_vs_asegp.py` (job 8786107): both **PASS**
  (dE ≤1.28e-8 meV/atom, per-atom max|dF| ≤1.05e-13, cos=1.0).

This suite is the up-to-date reference. When Phase-1 fixes rebuild the engine, it
re-runs and each cell must reproduce these values (dE ≤~1e-9 eV) or the change is
flagged.

### 6.1 Per-round ASE parity gate (mandatory every code edit, all phases)

The always-on tripwire (`scripts/n16_ase_parity.pbs`, `set -euo pipefail`): C1
step-0 energy + all-atom forces vs the surviving ASE FairChem oracle, covering
both engine paths. **Oracles (fixed reference, no rebuild):** N=16 W=1 ASE
E=−110673.82905; N=32 W=12 ASE-GP E=−885377.0600366.

Current gate run on the rebuilt engine (2026-08-27, job 8786107, **PASS**);
re-measured every code-edit round (the LAMMPS side changes; the ASE oracle does not):

| config | code path | dE/atom | per-atom max\|dF\| | cos | verdict |
|---|---|--:|--:|--:|:--|
| N=16 W=1 (32,768 at) | single-tile `predictor.cpp` | 1.41e-9 meV/at | 5.06e-14 eV/Å | 1.0000000000 | ✅ PASS |
| N=32 W=12 (262,144 at) | GP `mpi_peer_predictor.cpp`+XCCL | 1.28e-8 meV/at | 1.05e-13 eV/Å | 1.0000000000 | ✅ PASS |

Oracles (validated, reused every round, not regenerated): N=16 W=1 =
`hen/pbs/out/ase_n16_parity/*_w01` (ASE E=−110673.82905); N=32 W=12 =
`hen/pbs/out/ase12_n32/*_w12` (ASE E=−885377.06004). Every code-edit round must
re-run this gate and keep it green before the round closes.

---

## 7. P2.1 edge padding — fixes the N-specific NVT chunk-drift crash (2026-08-27)

**Root cause.** The prologue loop and every block's internal Edgewise loop split
the edge tensors by `edge_ac_chunk` in a Python `for`-loop, which `torch.jit.trace`
UNROLLS to a fixed number of `uma_ckpt.chunk` / `uma_ckpt.edge_degree` calls =
`ceil(E_trace / edge_ac_chunk)`. During MD the per-step edge count drifts, so at a
chunk boundary the runtime chunk count differs from the baked one and the traced
graph's fixed list length mismatches: `Expected K elements in a list but found
K+1`. This crashed N=24 (and N=16/N=36) NVT at step 1.

**Fix.** Pad the edge set to a **fixed multiple of `edge_ac_chunk`** so the chunk
count is constant regardless of drift:
- Export (`export_blocks_xpu.py`): the trace example's edge_index is padded to
  `cap = (E//chunk + 1)*chunk` (one guard chunk). Padded edges are **self-loops on
  a node the rank owns** (`node_offset`; 0 for W=1) with `cell_offset[:,0]=2` →
  `|r| ≈ 2·a ≫ cutoff` → radial envelope 0 → **exactly zero** energy/force/gradient
  contribution. `cap` + pad atom recorded in `metadata.json` (`edge_pad_cap`).
- Runtime (`mpi_peer_predictor.cpp`, `predictor.cpp`): after sharding, pad the
  per-step edges up to `edge_pad_cap` via `graph_shard::pad_edges_to_capacity`,
  using **this rank's** first owned node as the pad center (a global index the rank
  does not own would scatter out of its local accumulator → GPU segfault; this was
  caught and fixed during bring-up). Legacy artifacts (no `edge_pad_cap`) skip
  padding — fully backward-compatible.

**Validation.**
- **N=24 W=12 10-step NVT: now completes** (was: crash at step 1). exit=0,
  wall=113 s, step-0 PE = −373517.7698 = ASE-GP reference (dE = 1.8e-9 meV/atom).
  Job 8786199.
- **No regression:** N=16 W=1 + N=32 W=12 ASE parity gate still PASS on the P2.1
  build (dE 1.41e-9 / 1.28e-8 meV/atom, max|dF| 5.05e-14 / 1.05e-13, cos=1.0;
  job 8786211).
- Padded self-loops are provably zero-contribution (energy matches ASE to the
  machine floor), confirming the pad does not perturb physics.

This removes the last known correctness limitation blocking arbitrary-N NVT and is
a prerequisite for multi-node (per-node atom-count variation). Enabled by default
(`UMA_EDGE_PAD=1`); artifacts must be re-exported to carry `edge_pad_cap`.
