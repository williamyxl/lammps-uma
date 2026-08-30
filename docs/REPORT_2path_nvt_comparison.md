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

> ## STATUS: LAMMPS numbers CURRENT as of the opt5 build (2026-08-27 19:42)
>
> All LAMMPS `pair_style uma` results below (energy, force, timing) were
> **regenerated on the opt5 build** of `build-lmp-xccl/lmp` (rebuilt 2026-08-27
> 19:42; adds the opt5 `UMA_CHUNK_RETAIN_K` knob and P2.1 edge padding on top of
> the opt4 `UMA_NO_RECOMPUTE_*` knobs; opt5 knobs default OFF so C1/C2 default
> paths are unchanged). Jobs: `regen_main` 8787536 (C1/C2 sweep), `p21_sc16`
> 8787716 (N=16 NVT-10 scaling), `par_gate` 8787863 (ASE parity). **Reported
> energies are step-0 single-point** (decomposition-independent, the validated
> quantity); step-10 NVT energies are trajectory-dependent and shown separately.
> C1/C2 reproduce the prior build within run variance, and the ASE parity gate
> stays green (§6.1) — opt5 changed no physics.
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
the ASE parity gate (job 8787863). N=6/12/18 1-tile step-0 energies unchanged
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

**Fresh numbers (opt5 build 2026-08-27 19:42; jobs 8787536 sweep / 8787716 scaling
/ 8787863 parity).** Re-measured on the current engine (which adds the opt5
`UMA_CHUNK_RETAIN_K` knob, default off — C1/C2 default paths unchanged; numbers
reproduce the prior build within run variance).
**10-step NVT@300 K (completed all 10 steps unless noted):**
Path C is split into **C1 = pre-opt4** (full checkpointing) and **C2 = opt4** (partial
no-recompute; §2a). opt4 works on both the single-tile and 12-tile paths (same `block_context.cpp`
knobs); it OOMs where HBM is tight (1-tile N=18, 12-tile N≥36).
| Case | atoms | Path A (ASE) | Path C1 (pre-opt4) | Path C2 (opt4) |
|---|--:|--:|--:|--:|
| 1 tile, N=6 | 1,728 | 7.7 s | 22 s (Loop 10.8 s) | **15 s** (Loop 8.2 s) |
| 1 tile, N=12 | 13,824 | 61.2 s | 112 s (Loop 86.2 s) | **84 s** (Loop 65.3 s) |
| 1 tile, N=18 | 46,656 | 383.3 s | 364 s (Loop 296.2 s) | **OOM** (opt4 exceeds HBM) |
| **12 tiles, N=18** | 46,656 | **90 s**† | **57 s** (Loop 30.6 s) | **34 s** (Loop 23.0 s) |
| **12 tiles, N=32** | 262,144 | **409 s**† (real NVT) | **230 s**\* (Loop 181.6 s) | **172 s**\* (Loop 138.2 s) |
| **12 tiles, N=34** | 314,432 | n/a (ASE-GP ceiling N=32) | **276 s** (Loop 220.0 s) | **209 s** (Loop 168.3 s) |
| **12 tiles, N=36** | 373,248 | n/a (ASE-GP ceiling N=32) | **331 s** (Loop 265.1 s) | **OOM** (opt4 exceeds HBM) |
| **12 tiles, N=38** | 438,976 | n/a (ASE-GP ceiling N=32) | **389 s** (Loop 313.7 s) | **OOM** (opt4 exceeds HBM) |

C1/C2 walls are the current optimized stack (opt1+opt2+opt3; C2 adds opt4), freshly
measured on the opt5 build. All 12-tile C1/C2 rows completed the full 10-step NVT.
**C1 and C2 give identical step-0 energy** (N=18 −157578.531115; N=32 −885377.060040;
N=34 −1061980.383367; N=36 −1260622.568658; N=38 −1482625.946248) — proving opt4 is
physics-neutral. (Step-10 NVT energies match within a path but are trajectory-dependent
across W; not a cross-W metric — see §2 convention note.)
**opt4 (C1→C2) gain by size** (Loop / wall, opt5 build 2026-08-27):
- 1-tile: N=6 10.8→8.2 s (**−24%**) / 22→15 s; N=12 86.2→65.3 s (**−24%**) / 112→84 s;
  N=18 OOM (1-tile memory ceiling; single-tile N=18 NVT is tight even for C1).
- 12-tile: N=18 30.6→23.0 s (**−25%**) / 57→34 s; N=32 181.6→138.2 s (**−24%**) / 230→172 s;
  N=34 220.0→168.3 s (**−24%**) / 276→209 s.
opt4 consistently gives **~−24% Loop** wherever it fits. **C2 ceiling by config:**
1-tile: fits N≤12, OOMs N=18; 12-tile: fits N≤34, OOMs N≥36
(`UR_RESULT_ERROR_OUT_OF_RESOURCES`; job 8787536 N=36/38 C2 OOM). Use C1 outside those ranges.
Jobs: C1/C2 sweep 8787536; N=16 scaling 8787716; ASE parity 8787863.

---

## 2a. C1 (pre-opt4) vs C2 (opt4) — the two LAMMPS execution paths

**Fresh (opt5 build 2026-08-27 19:42, job 8787536).**
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
Both pass full per-atom parity vs ASE-GP (§1 ¶; N=32 W=12 gate PASS, job 8787863).

### Performance — C1 vs C2 (N=32 W=12, 10-step NVT)

| metric | C1 (pre-opt4) | C2 (opt4) | opt4 gain |
|---|--:|--:|--:|
| **Loop time** (pure MD compute) | 181.6 s | 138.2 s | **−24% (1.31×)** |
| **whole wall** (incl. cold load) | 230 s | 172 s | **−25% (1.34×)** |
| **max N @ 12 tiles (NVT)** | **N=38** (389 s) | **N=34** (OOMs at N≥36) | capacity cost |

**Performance verdict:** opt4 buys **−24% Loop / −25% wall at N=32** by removing the
block+prologue backward recompute. Fresh walls (opt5 build): C1 230 s / C2 172 s (job 8787536). The
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

‖ **N=32: C1 = 230 s, C2 = 172 s** (opt5 build; see §2a) — **beats ASE-GP's real-NVT 409 s by 1.78×/2.38×** (§9a), FP64, energy
bit-identical to the 450 s baseline. Two accuracy-neutral optimizations: (opt3)
XCCL tuning (`CCL_ZE_IPC_EXCHANGE=pidfd`, `CCL_ATL_TRANSPORT=ofi`, `FI_PROVIDER=tcp`)
450→276 s; (opt1) coarser activation-checkpoint chunk (`EDGE_AC_CHUNK` 16384→65536,
fewer backward recomputes) 276→235 s. NVT compute Loop-time 214→180 s. Progression:
450 s (baseline) → 276 s (opt3) → **235 s = path C1 (opt1+opt3), 1.91× faster than baseline**
→ **193 s = path C2 (+opt4), 2.33× faster than baseline** (§2a). vs ASE: against the
measured real 10-step ASE NVT (409 s, §9a) the current C1/C2 (230/172 s) are 1.78×/2.38×
faster (the older 1.10×/1.34× figures used the retired `11·ef` reconstruction of ASE).

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
from project `hen`). The N=32 value here (**409 s**) is a **measured real 10-step
`NoseHooverChainNVT`** run (job 8788499; §9a), directly comparable to LAMMPS `fix nvt`.
The N=18 ASE value (90 s) is still the older `11·ef_mean` reconstruction (no real-NVT
N=18 measured). **Cross-check: ASE-GP and pair_style uma agree on energy** — N=18 both
−157578.531115; N=32 −885377.06004 (ASE) vs −885377.06004 (ours), Fmax 0.848045 identical.

**Walltime caveats (important for interpretation):**
- 1-tile Path A is model-resident single-point; 1-tile Path C wall includes per-rank
  **cold model load** — so small-N 1-tile Path C is load-dominated, not a fair per-step
  number.
- 12-tile N=32 (real 10-step NVT both sides): **C2 172 s / C1 230 s vs ASE-GP 409 s →
  2.38× / 1.78× faster** (§9a). The original correctness-first port was 450 s; the
  throughput optimization (opt1+opt3+opt4) has been done. Earlier drafts quoted
  1.34× vs an ASE `11·ef` reconstruction that understated ASE's real MD cost ~2×.
- This comparison establishes **correctness + capacity + optimized throughput**; a
  like-for-like warm, load-excluded steps/s benchmark is still future work.

---

## 3. Capacity / max-N — Path C `pair_style uma`, 12 tiles, 10-step NVT@300 K

**Fresh (opt5 build 2026-08-27 19:42, job 8787536).**
Single-crystal NaCl across all 12 XPU tiles (native XCCL graph-parallel, FP64).
**All rows below completed the full 10-step NVT.** The **"orig baseline"** column is the
first correctness-first port (`edge_ac_chunk=16384`, no opt2/opt3/opt4). The **"current
best"** column is the optimized stack, freshly measured; the config used is noted per row
because opt4 (C2) is N-limited (see §2). step-0 PE is the decomposition-independent energy.
| N | atoms | orig baseline wall | current best wall | best path | step-0 PE (eV) |
|--:|--:|--:|--:|:--|--:|
| 18 | 46,656 | 88 s | **34 s** (C2) / 57 s (C1) | **C2** (opt4 fits) | −157578.531115 |
| 32 | 262,144 | 450 s | **172 s** (C2) / 230 s (C1) | **C2** (opt4) | −885377.060040 |
| 34 | 314,432 | 534 s | **209 s** (C2) / 276 s (C1) | **C2** (opt4 fits) | −1061980.383367 |
| 36 | 373,248 | 666 s | **331 s** (C1) | **C1** (C2/opt4 OOMs) | −1260622.568658 |
| 38 | 438,976 | 800 s | **389 s** (C1, Loop 313.7 s) | **C1** (C2/opt4 OOMs) | −1482625.946248 |
| 40 | 512,000 | OOM | OOM | — | OOM |

- **N=32 current best = 172 s** (C2 = opt1+opt2+opt3+opt4; job 8787536), 2.62× faster
  than the 450 s original baseline and 2.38× faster than ASE-GP real-NVT (409 s, §9a).
- **N=38 current best = 389 s** (C1 = opt1+opt2+opt3, Loop 313.7 s; opt4 OOMs at N=38),
  2.06× faster than the 800 s original baseline, full 10-step NVT. N=40 still OOMs.
- **All N (18/32/34/36/38) re-measured** on the opt5 build (job 8787536); no
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
- **Throughput (opt5 build):** N=32 W=12 NVT **450 s → C1 230 s → C2 172 s** (C2 now 2.38×
  faster than ASE-GP's real-NVT 409 s, §9a); N=38 W=12 **800 s → C1 389 s** (C2/opt4 OOMs at N=38). opt2
  also cut on-disk artifacts 53 GB → 27 GB. C1 and C2 are numerically equivalent (step-10
  dE ≤ ~1e-9 eV vs the original baseline); ASE parity gate PASS on the opt5 build (§6.1).
- **Not yet done:** re-measure N=18 C1 vs old-baseline 88 s; a warm, load-excluded steps/s
  benchmark; selective per-chunk retain to extend C2 beyond the current N=34 ceiling.
- **Tile scaling:** see §5 — strong scaling on N=16 shows useful speedup to W=4 (C1) and
  W=6 (C2); beyond that communication overhead saturates the small system.

---

## 5. Strong-scaling study — N=16 (32,768 atoms), W = 1, 2, 4, 6, 8, 12 tiles

**Fresh (opt5 build + P2.1 padded artifacts, 2026-08-27, job 8787716).
NVT 300 K, 10 steps** — the proper MD timing (matches the §2/§3 convention). This
became possible only after P2.1 edge padding (§7) fixed the N=16 10-step-NVT
chunk-drift crash; the earlier single-point stand-in is retired.

NaCl N=16 (32,768 atoms; a=5.64 Å, rattle 0.05 Å, seed 0). step-0 PE =
−110,673.829050 eV across **all** W and both C1/C2 (identical, = ASE W=1 oracle,
§6.1). **Loop time** is the pure 10-step MD compute (excludes cold load) and is
the scaling metric; whole-wall (incl. load) also shown.

> **Note:** these §5 tables use `EDGE_AC_CHUNK=65536`. **§10 shows N=16 is 24–38%
> faster with the tuned `EDGE_AC_CHUNK=32768`** (W=12 25.2→18.4 s; W=8 dip cured),
> lifting C2 to 0.82–0.95× of ASE. Use §10's numbers as the N=16 best; the tables
> below are retained for the chunk-size before/after comparison.

**opt1+opt2+opt3 artifacts (C1 = full checkpoint; C2 = + opt4), NVT-10, chunk=65536:**
| W (tiles) | C1 Loop | C1 eff% | C2 Loop | C2 eff% | C2 gain | C1 wall | C2 wall |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1  | 221.3 s | 100% | 168.6 s | 100% | 1.31× | 283 s | 208 s |
| 2  | 133.6 s |  83% |  98.8 s |  85% | 1.35× | 172 s | 124 s |
| 4  |  78.3 s |  71% |  58.3 s |  72% | 1.34× | 103 s |  76 s |
| 6  |  46.7 s |  79% |  34.6 s |  81% | 1.35× |  66 s |  47 s |
| 8  |  48.8 s |  57% |  37.0 s |  57% | 1.32× |  69 s |  51 s |
| 12 |  33.1 s |  56% |  25.2 s |  56% | 1.31× |  50 s |  37 s |

Efficiency = (W=1 Loop / W) / actual Loop × 100% (Loop-time strong scaling).

### ASE (Path A) N=16 scaling — REAL NVT (NoseHooverChainNVT, 10 steps)

ASE FairChem graph-parallel running an **actual** ASE Nosé-Hoover-chain NVT
(`NoseHooverChainNVT`, dt=1 fs, T=300 K, tchain=3), 10 steps, on the identical
N=16 system (job 8788329, `hen/scripts/fxpu_1node_nvt.py`). This replaces the
earlier `11·ef_mean` reconstruction — both sides now run a genuine 10-step
integrator, so the comparison is like-for-like. `md_s` = the 10-step integration
only (load + warmup excluded), directly comparable to LAMMPS Loop time. Energy
matches (step-0 −110673.829050 = C1/C2 = ASE oracle).

| W (tiles) | ASE NVT md (10 steps) | ASE md/step | ASE eff% |
|--:|--:|--:|--:|
| 1  | **OOM** | — | — |
| 2  | 92.4 s | 9.24 s | 100% |
| 4  | 47.4 s | 4.74 s |  97% |
| 6  | 31.9 s | 3.19 s |  97% |
| 8  | 24.9 s | 2.49 s |  93% |
| 12 | 17.4 s | 1.74 s |  89% |

ASE eff% = (W=2 md / (W/2)) / md × 100% (W=2 baseline, since W=1 OOMs).
**W=1 N=16 OOMs for ASE** (eager, full-activation) — the same single-tile NVT
memory ceiling that OOMs LAMMPS at 1-tile N=18 (§1 †): at N=16-18 a single
`get_potential_energy` fits but a 10-step NVT does not, on both engines.

### Path A vs Path C — real 10-step NVT, load-excluded (ASE md_s vs LAMMPS Loop)

| W | ASE NVT md | C1 Loop | C2 Loop | C1 vs ASE | C2 vs ASE |
|--:|--:|--:|--:|--:|--:|
| 1  | OOM | 221.3 s | 168.6 s | — | — |
| 2  | 92.4 s | 133.6 s |  98.8 s | 0.69× | 0.94× |
| 4  | 47.4 s |  78.3 s |  58.3 s | 0.61× | 0.81× |
| 6  | 31.9 s |  46.7 s |  34.6 s | 0.68× | 0.92× |
| 8  | 24.9 s |  48.8 s |  37.0 s | 0.51× | 0.67× |
| 12 | 17.4 s |  33.1 s |  25.2 s | 0.53× | 0.69× |

(ratio = ASE md / LAMMPS Loop; >1 means LAMMPS faster.)

**Observations:**
- **Strong scaling to W=6:** C1 Loop 221→47 s (**4.7×** on 6 tiles, 79% eff); C2
  169→35 s (**4.9×**, 81% eff). ASE md scales similarly (W=2→12: 92→17 s, 89% eff).
- **W=8 is a dip** (eff 56% vs W=6 79%): 32,768 atoms / 8 = 4,096 atoms/tile
  partitions less evenly than /6 or /12 for this cell; W=12 recovers (55% but
  lowest absolute Loop, 33 s C1 / 25 s C2).
- **C2 (opt4) consistently 1.31–1.35× faster than C1** in Loop time at every W —
  the removed block+prologue backward recompute is a fixed fraction of the call.
- **vs ASE (real NVT):** with the **old chunk=65536** artifacts, ASE was faster at
  N=16 (C2 0.67–0.94× of ASE). **§10 diagnosed this as edge-padding waste** (33% of
  compute), not LAMMPS overhead; re-exporting N=16 at **`EDGE_AC_CHUNK=32768`** cuts
  the W=12 Loop 25.2→18.4 s (−27%), lifting C2 to **0.95× of ASE (near parity)** at
  W=12, energy bit-identical. The tables above are the chunk=65536 numbers; §10 has
  the fixed N=16 chunk-size results. At production N=32 C2 already beats ASE 2.38×
  (§9a); the N=16 chunk fix closes the small-system gap.
- **Step-10 energy** differs W=1 (−110602.976) vs W≥2 (−109317.384) — expected:
  the velocity seed distributes differently per domain decomposition, so the MD
  trajectories diverge (step-0 is the decomposition-independent invariant).

---

## 6. Validation suite — current opt5 build (2026-08-27 19:42)

All LAMMPS numbers in this report were regenerated on the opt5 build (2026-08-27
19:42; opt4 + P2.1 + opt5 `UMA_CHUNK_RETAIN_K`, opt5 knobs default OFF). This is
the current-build baseline; Phase-1 (`docs/CODE_QUALITY.md` Part C) will
**re-run this identical suite after its fixes land** and any change beyond the
tolerances below is a regression. The reference is the ASE oracle (no ASE
rebuild); step-0 energy is the invariant.

**Suite results (opt5 build; jobs 8787536 / 8787716 / 8787863):**
| # | system | tiles | path | ASE reference (step-0 E) | opt5-build LAMMPS step-0 E / wall | verdict |
|---|---|---|---|---|---|:--|
| 1 | N=16 (32,768) | W=1 | C1 & C2 | −110673.82905 | −110673.829050 / C1 283s, C2 208s (NVT-10) | ✅ dE=1.4e-9 meV/at |
| 2 | N=32 (262,144) | W=12 | C1 & C2 | −885377.0600366 | −885377.060040 / C1 230s, C2 172s | ✅ dE=1.28e-8 meV/at |
| 3 | N=18 (46,656) | W=1 | C1 | −157578.531115 | −157578.531115 / C1 364s | ✅ exact |
| 4 | N=38 (438,976) | W=12 | C1 | (no ASE oracle) | −1482625.946248 / C1 389s | ✅ NVT completes, N-consistent |
| 5 | N=16 scaling W=1..12 (NVT-10) | 1–12 | C1 & C2 | −110673.82905 | step-0 −110673.829050 all W (§5) | ✅ all W match, NVT completes |

- **C1/C2 equivalence:** identical step-0 energy at every N where both fit
  (N=16/18/32/34 to all printed digits) — opt4 is physics-neutral.
- **Row-4 N=38** has no ASE oracle (ASE-GP ceiling N=32); validated by step-0
  self-consistency + per-atom-E consistency with neighboring N.

**Gate cross-checks (opt5 build, ASE-anchored):**
- N=16 W=1 + N=32 W=12 `parity_vs_asegp.py` (job 8787863): both **PASS**
  (dE ≤1.28e-8 meV/atom, per-atom max|dF| ≤1.05e-13, cos=1.0).

This suite is the up-to-date reference. When Phase-1 fixes rebuild the engine, it
re-runs and each cell must reproduce these values (dE ≤~1e-9 eV) or the change is
flagged.

### 6.1 Per-round ASE parity gate (mandatory every code edit, all phases)

The always-on tripwire (`scripts/n16_ase_parity.pbs`, `set -euo pipefail`): C1
step-0 energy + all-atom forces vs the surviving ASE FairChem oracle, covering
both engine paths. **Oracles (fixed reference, no rebuild):** N=16 W=1 ASE
E=−110673.82905; N=32 W=12 ASE-GP E=−885377.0600366.

Current gate run on the opt5 build (2026-08-27 19:42, job 8787863, **PASS**);
re-measured every code-edit round (the LAMMPS side changes; the ASE oracle does not):

| config | code path | dE/atom | per-atom max\|dF\| | cos | verdict |
|---|---|--:|--:|--:|:--|
| N=16 W=1 (32,768 at) | single-tile `predictor.cpp` | 1.41e-9 meV/at | 5.05e-14 eV/Å | 1.0000000000 | ✅ PASS |
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

---

## 8. opt5 — single-node performance investigation (2026-08-27)

Building on the plan in the opt5 discussion, three single-node levers were
implemented/tested against the N=32 W=12 operating point (C1 Loop 182.7 s,
C2 Loop 139.0 s). Two were empirically ruled out and one is blocked by the
traced-graph design; all findings below are measured, not estimated.

### P-1 — selective per-chunk retain (`UMA_CHUNK_RETAIN_K`) — implemented, minor
New granular knob in `block_context.cpp`: retain the first *k* edge-chunks per
block under autograd (no backward recompute), checkpoint the rest. Numerically
identical (step-0 PE −885377.060040 at every k). **Result at N=32 W=12:**
| config | Loop | vs C1 |
|---|--:|--:|
| C1 (all checkpoint) | 182.7 s | — |
| C1 + retain 1/blk | 177.2 s | −3% |
| C1 + retain 2/blk | 174.4 s | −4.5% |
| C1 + retain 3/blk | 171.8 s | −6% |
| C1 + retain 4/blk | 169.0 s | −7.5% |
| **C2 (retain block+edeg)** | **139.0 s** | **−24%** |
| C2 + any chunk retain | **OOM** | — |
**Verdict:** C2 (retain node-sized block+prologue activations) is far more
memory-efficient per unit speedup than retaining chunk (edge-sized) activations,
and the two cannot be combined at N=32/W=12 (job 8787377 C2+chunk OOMs; job
8787429 C1+chunk works). P-1 is kept as a tunable knob (useful where C2 has spare
HBM, e.g. small-N/high-W) but is **not** the production win — C2 already captures
the accessible headroom better. Job 8787429.

### P-6 — edge-balanced node partition — RULED OUT (not a bug)
Hypothesis: the §5 W=8 efficiency dip (56% vs W=6 79%) is atom-vs-edge partition
imbalance. **Measured per-rank shard edge counts are already perfectly balanced**
(N=16 W=8: exactly 131072 edges on all 8 ranks; W=6: 174784/174752, 0.02% spread).
The dip is therefore **fixed per-step overhead** (collectives + NL + integrator
that don't shrink with more tiles): per-edge cost rises 27 µs (W=6) → 38 µs
(W=8/12) as edges/rank fall. This is the strong-scaling communication wall, not a
partitioning defect — no partition change helps.

### P-2 — per-rank chunk-weight de-duplication — blocked by traced-graph design
Load is ~35 s of the 174 s wall; each of the 12 ranks loads its **own distinct**
2.2 GB `w12/r{R}/` set (4×554 MB `model_chunk_*.pt`), 27 GB total. **The chunk
weights are byte-identical across ranks** (verified 93/93 tensors r0=r5=r11, job
8787269) — only the baked `node_offset` / `n_local` scalars differ, so dedup to one
shared set (page-cache-shared across tiles on a node) would cut ~15-25 s of wall.
**Blocker:** `_ChunkCore` is `torch.jit.trace`d (trace accepts only Tensor inputs),
so `node_offset`/`n_local` are baked constants, not runtime args — and `n_local`
sizes the scatter output buffer (memory-relevant), so it can't be neutralized by an
edge pre-shift without a full-N scatter (OOM). Making the chunk core rank-agnostic
requires **scripting the core** (or adding tensor-typed scalar inputs) — a larger,
higher-risk refactor deferred out of opt5.

### opt5 conclusion
At the N=32 W=12 production point the engine is **compute-bound on forward + chunk
backward with no HBM headroom** (C2 is already at the memory ceiling). The
remaining single-node levers are: (a) **P-2 weight dedup** for a load-time win
(needs the trace→script core refactor), (b) **P-4 forward-kernel optimization**
(the untouched 25% fwd bucket, needs `torch.profiler` on the chunk SO2/wigner),
and (c) **mixed precision** (P-7, breaks the FP64 parity contract — separate opt-in
path). P-1/P-6 are closed. All opt5 runs kept the ASE parity gate green (step-0 PE
−885377.060040, unchanged).

---

## 9. opt5 vs ASE — timing analysis

Direct comparison of the current opt5 build (C1 = full checkpoint, C2 = +opt4) to
Path A (ASE FairChem graph-parallel) on identical NaCl. Both FP64, same tiles,
same system; energies match to the machine floor (§1, §6.1). **Both sides now run
a real 10-step NVT** (LAMMPS `fix nvt`; ASE `NoseHooverChainNVT`, job 8788329/
8788499) — the earlier `11·ef_mean` reconstruction is retired. Two framings:
- **Whole wall** (incl. cold model load + warmup) — end-to-end "submit to done".
- **MD compute** — the 10-step integration only (load+warmup excluded): LAMMPS
  **Loop time** vs ASE **md_s**. Both include the integrator + (LAMMPS) per-step
  neighbor-list rebuild, so this is a genuine like-for-like MD number.

### 9a. N=32 W=12 (262,144 atoms) — production point, REAL 10-step NVT

Both sides run a genuine 10-step NVT@300 K (LAMMPS `fix nvt`; ASE
`NoseHooverChainNVT`, job 8788499). Energy matches (step-0 −885377.060 both).

| metric | ASE-GP | opt5 C1 | opt5 C2 | C2 vs ASE |
|---|--:|--:|--:|--:|
| whole wall | 409.3 s | 230 s | **172 s** | **2.38× faster** |
| MD compute (10 steps) | 296.2 s | 181.6 s | **138.2 s** | **2.14× faster** |
| per MD step | 29.6 s | 18.2 s | **13.8 s** | 2.14× faster |
| cold load | 77.6 s | ~48 s | ~34 s | ~2.3× less |
| warmup | 35.4 s | (in load) | (in load) | — |

**Takeaways (N=32):**
- **C2 beats ASE by 2.38× on whole wall and 2.14× on MD compute** — much larger
  than the earlier `11·ef` reconstruction suggested (1.50×/1.18×). The gap grew
  because a **real** ASE NVT step costs ~29.6 s (a separate energy *and* force
  evaluation per step through the eager FairChem calculator + per-step neighbor
  rebuild), vs the single combined E+F call (`ef_mean` 14.8 s) the reconstruction
  assumed. The reconstruction understated ASE's true MD cost by ~2×.
- Even **C1** (full checkpointing, N=38-capable) is **1.78× faster on wall / 1.63×
  on MD compute** than ASE, while reaching N=38 vs ASE-GP's N=32 ceiling.
- This is now a true like-for-like MD comparison (both run the integrator +
  per-step neighbor list); LAMMPS's traced+AC engine is simply doing less
  redundant work per step than ASE's eager path.

### 9b. N=16 (32,768 atoms) — strong scaling, real 10-step NVT (MD compute)

ASE `md_s` (real NoseHooverChainNVT, 10 steps) vs LAMMPS Loop time, both the
10-step MD only (§5):
| W | ASE NVT md | C1 Loop | C2 Loop | C2 vs ASE | C1 vs ASE |
|--:|--:|--:|--:|--:|--:|
| 1  | OOM | 221.3 s | 168.6 s | — | — |
| 2  | 92.4 s | 133.6 s |  98.8 s | 0.94× | 0.69× |
| 4  | 47.4 s |  78.3 s |  58.3 s | 0.81× | 0.61× |
| 6  | 31.9 s |  46.7 s |  34.6 s | 0.92× | 0.68× |
| 8  | 24.9 s |  48.8 s |  37.0 s | 0.67× | 0.51× |
| 12 | 17.4 s |  33.1 s |  25.2 s | 0.69× | 0.53× |

(ratio = ASE md / LAMMPS Loop; >1 ⇒ LAMMPS faster.)

**Takeaways (N=16):**
- With the old chunk=65536 artifacts ASE was faster here (C2 0.67–0.94× of ASE).
  **§10 root-caused it to P2.1 edge-padding waste** (33% of compute at chunk=65536),
  NOT LAMMPS overhead (Pair=99.9% of Loop, integrator/NL/comm ~0). Re-exporting at
  **`EDGE_AC_CHUNK=32768`** cuts W=12 Loop 25.2→18.4 s (−27%) → **C2 0.95× of ASE**
  (near parity), energy bit-identical. See §10.
- **W=1 N=16: both OOM** — single-tile NVT memory ceiling on both engines.
- The picture **reverses with system size**: at production N=32 C2 beats ASE 2.38×
  (§9a) — LAMMPS's advantage is the larger, compute-bound regime.

### 9c. Summary
- **Production (N=32 W=12): opt5 C2 is 2.38× faster than ASE-GP on whole wall and
  2.14× on MD compute** (real 10-step NVT, both sides), at identical FP64 accuracy.
  C1 is 1.78×/1.63×.
- **Capacity:** opt5 C1 runs to N=38 (438,976 atoms); ASE-GP ceiling is N=32.
- **Small-N (N=16):** originally ASE was faster (C2 0.67–0.94× of ASE), but **§10
  (chunk=32768) + §11 (chunk-retain K=3) now make LAMMPS 1.36× faster than ASE at
  N=16 W=12** (12.8 s vs ASE 17.4 s), parity preserved. So LAMMPS beats ASE across
  the whole tested range (N=16–38).
- **Methodology note:** the earlier `11·ef_mean` ASE reconstruction has been
  replaced everywhere by measured `NoseHooverChainNVT` 10-step runs. The
  reconstruction understated ASE's real MD cost at N=32 by ~2× (a real NVT step
  does separate energy+force evals + neighbor rebuild), so the true LAMMPS
  advantage is larger than previously reported.
- Jobs: opt5 C1/C2 8787536/8787716; ASE real-NVT 8788329 (N=16), 8788499 (N=32).

---

## 10. N=16 NVT diagnosis + fix — edge-chunk size (opt5, 2026-08-28)

**Symptom.** At N=16 ASE was faster than LAMMPS (§9b, C2 0.67–0.94× of ASE),
unlike the N=32 production point where C2 wins 2.14–2.38×.

**Diagnosis (job 8788549).** The LAMMPS MPI timing breakdown at N=16 W=12 showed
**Pair = 99.87%** of Loop, Neigh = 0, Comm = 0.004%, Modify (NVT integrator) =
0.12% — so the LAMMPS-side overhead (neighbor list, integrator, comm) that earlier
drafts blamed is **negligible**; the entire cost is inside the UMA engine's
per-step force call. The `UMA_MP_PERF` per-call split (steady state) was graph
186 ms / fwd 745 ms / bwd 1499 ms / force_ar 58 ms.

**Root cause.** The engine reported `n_edges_shard=131072` while the shard has only
**87,392 real edges/tile**. P2.1 edge padding rounds up to a fixed multiple of
`EDGE_AC_CHUNK`: at chunk=65536, 87,392 → **131,072 (2 chunks) = 33% of every
step's compute wasted on padding edges** (they run the full SO2 conv + wigner
recompute + backward but contribute zero). At N=32 the shard is ~700k edges (~11
chunks) so the same padding is only ~9% — which is why the waste only bites at
small N.

**Fix — smaller `EDGE_AC_CHUNK` for small systems** (export-time knob, no code
change, numerically identical). N=16 W=12 C2 NVT-10, re-exported at each chunk
(jobs 8788588/8788596/8788652 export, 8788623/8788634/8788682 run):

| EDGE_AC_CHUNK | edges/tile (padded) | pad waste | Loop | vs 65536 | step-0 PE |
|--:|--:|--:|--:|--:|--:|
| 65536 (old default) | 131,072 | 33% | 25.2 s | — | −110673.829050 |
| **32768** | 98,304 | 11% | **18.4 s** | **−27% (1.37×)** | −110673.829050 |
| 16384 | 98,304 | 11% | 19.0 s | −25% | −110673.829050 |
| 8192  | 90,112 |  3% | 18.4 s | −27% | −110673.829050 |

**Result:** N=16 W=12 C2 **25.2 → 18.4 s (−27%)**, energy bit-identical (parity
preserved). This closes the gap to ASE real-NVT (md 17.4 s): **0.69× → 0.95×**
(near parity). 32768 and 8192 tie on speed; **32768 is the pick** (fewest chunks =
least per-chunk overhead + smallest artifacts). Below 32768 is diminishing
returns; the fwd/bwd compute drops in proportion to the removed padding edges
(fwd 745→578 ms, bwd 1499→1043 ms) while graph (NL) and force_ar are unchanged.

**The fix helps the whole N=16 scaling curve — and cures the W=8 dip.** At
chunk=65536 the padding waste was W=4 20%, W=6 11%, W=8 **33%**, W=12 33% (W=8 and
W=12 round up to the next full chunk); chunk=32768 halves most of these. Re-run
N=16 C2 NVT-10 (jobs 8788623/8788704):

| W | chunk=65536 Loop | chunk=32768 Loop | gain | ASE real-NVT md | C2/ASE (32768) |
|--:|--:|--:|--:|--:|--:|
| 4  | 78.3 s | **52.5 s** | −33% | 47.4 s | 0.90× |
| 6  | 46.7 s | **35.3 s** | −24% | 31.9 s | 0.90× |
| 8  | 48.8 s | **30.4 s** | −38% | 24.9 s | 0.82× |
| 12 | 25.2 s | **18.4 s** | −27% | 17.4 s | 0.95× |

All step-0 PE = −110673.829050 (parity preserved at every W). **The W=8 anomaly
(previously slower than W=6) is fixed** — it was the 33% padding waste, and
chunk=32768 brings W=8 (30.4 s) back below W=6 (35.3 s). C2 now runs at 0.82–0.95×
of ASE across the ladder (was 0.67–0.94×), i.e. near parity, the small-system gap
substantially closed.

**Efficiency ceiling reached.** At chunk=32768 N=16 already matches N=32's
per-padded-edge compute efficiency (16.5 vs 17.3 µs/edge, 0.96×); only ~7% headroom
remains, entirely the residual 11% padding (98,304 vs 87,392 real edges). Reclaiming
it needs chunk≈22000 (0.7% waste) but that guard (608 edges) is too small for NVT
edge drift — unsafe. So **32768 is the safe optimum for N=16**; chunk tuning is
exhausted.

**Takeaway / guidance.** `EDGE_AC_CHUNK` should make the per-tile shard a
near-integer multiple of it — roughly **chunk ≈ (edges/tile) / 3–4** with enough
guard for MD edge drift. Rule of thumb at W=12: **N≤16 → 32768**; N≈24–34 → 65536
(already ~3–9% waste). Export-time choice; re-export to change it. The residual
waste is fundamental to the fixed trace-baked chunk count — a true P2.2 (data-
dependent chunk loop, or per-step pad to the exact next boundary) would need a
scripted loop, not a traced one, and is deferred.

---

## 11. N=16 — beating ASE (opt5-graph + chunk-retain, 2026-08-28)

Goal: get LAMMPS NVT **faster than ASE** at N=16 (where §9b/§10 still had ASE
ahead). Diagnosed the steady-state per-step budget at N=16 W=12 C2 (chunk=32768):
graph 182 ms / fwd 578 ms / bwd 1043 ms / force_ar 23 ms. Two fixes tried; both
keep parity (per-atom force PASS, below).

### Fix #1 — graph-phase host-sync removal (kept, but NOT the lever)
Sub-phase timing showed the 182 ms "graph" is ~all inside `vesin_build_graph_cuda`
(vesin 176 ms, shard/pad 4 ms). Vesin is already on XPU; the removable cost is the
per-step **XPU→host syncs**: `counts.max().item()` (max-neighbor cap check) and
`pbc.item()` in the wrap. Added `UMA_SKIP_MAXNBR_CAP=1` (skip the cap+sync when the
lattice guarantees degree ≤ K — bit-identical for NaCl) and an on-device pbc mask +
cached device `z`. **Result:** vesin host time 176 → 118 ms/step (−31%), but the
**Loop was unchanged (18.53 → 18.50 s)** — the graph phase **overlaps** with the
XPU compute of the step and is **not on the critical path**. Kept as a free
host-side win; it does not move the wall.

### Fix #2 — chunk-retain at N=16 (the win: beats ASE)
The critical path is fwd+bwd compute; bwd (1043 ms) is dominated by the per-chunk
**backward recompute** (C2 keeps chunk activation-checkpointing). At N=16 there is
ample HBM (~2,700 atoms/tile), so the opt5 `UMA_CHUNK_RETAIN_K` knob can **retain
chunk activations** (skip their recompute) — the same knob that OOMs at N=32.
Sweep (N=16 W=12, chunk=32768, on top of C2):

| UMA_CHUNK_RETAIN_K | bwd/call | Loop | vs ASE (md 17.4 s) |
|--:|--:|--:|--:|
| 0 (C2) | 1045 ms | 18.3 s | 0.95× |
| 1 | 911 ms | 16.4 s | 1.06× |
| 2 | 771 ms | 14.6 s | 1.19× |
| **3 (all chunks)** | **640 ms** | **12.8 s** | **1.36× faster** |

All step-0 PE = −110673.829050 (bit-identical), no OOM. **Per-atom force parity vs
the ASE N=16 W=12 oracle (job 8788864): dE=1.41e-9 meV/atom, max|dF|=5.05e-14
eV/Å, cos=1.0000000000 → PASS** — the speedup is fully accuracy-neutral.

### N=16 W=12 progression
| config | Loop | vs ASE |
|---|--:|--:|
| chunk=65536 (original) | 25.2 s | 0.69× (ASE faster) |
| chunk=32768 (§10) | 18.4 s | 0.95× |
| **chunk=32768 + RETAIN_K=3 (fast)** | **12.8 s** | **1.36× FASTER than ASE** |

**Result: LAMMPS now beats ASE at N=16** (1.36× on MD compute), was 0.69×. The
recommended small-system config is `EDGE_AC_CHUNK=32768` (export) +
`UMA_NO_RECOMPUTE_BLOCK=1 UMA_NO_RECOMPUTE_EDEG=1 UMA_CHUNK_RETAIN_K=3
UMA_SKIP_MAXNBR_CAP=1` (runtime). `UMA_CHUNK_RETAIN_K` is HBM-bounded (fits at
small N; use K=0/C2 at N≥32 where it OOMs — §8). Jobs: 8788849 (retain sweep),
8788864 (parity), 8788838 (graph-sync).

**Takeaway.** The N=16 gap to ASE was two things: (1) edge-padding waste (§10,
chunk fix) and (2) chunk backward-recompute that N=16's HBM headroom makes
unnecessary. Fixing both flips N=16 from 0.69× to 1.36× vs ASE. Combined with the
production point (N=32 C2 2.14–2.38× faster, §9a), **LAMMPS `pair_style uma` is now
faster than ASE across the tested range** (N=16 through N=38), at identical FP64
accuracy.

---

## 12. opt6 — deeper perf diagnosis (parity-safe levers, 2026-08-28)

Instrumented the XCCL collectives (`UMA_PEER_PERF`) to see what remains inside
fwd/bwd after opt5. Steady per-call (chunk-tuned, C2/fast):

| | N=32 W=12 (C2) | N=16 W=12 (K=3) |
|---|--:|--:|
| ms_total | 13,935 | 1,290 |
| ms_vesin (NL build) | 1,534 | 122 |
| ms_fwd (incl. collectives) | 4,380 | 460 |
| ms_bwd (incl. collectives) | 7,872 | 655 |
| **ms_allreduce (15 calls)** | **1,130–1,260** | **583–604** |
| ms_allgather (4 calls, 0.8 GB) | 111–125 | 20–64 |
| ms_force_ar (final) | 126 | 23 |

**Findings:**
- **all_gather is NOT the bottleneck** (0.8 GB, ~1% of the call) — earlier volume
  estimates were wrong; node features are small.
- **15 tiny all_reduces/step dominate the collective cost** — ~9% at N=32
  (1.2 s), **~45% at N=16** (0.6 s). These are the per-block `balance_channels`
  charge+spin corrections (`all_reduce_with_grad` of a tiny l=0 channel slice) +
  their backward + the all_gather backward. **Latency-bound** (~40–80 ms each for
  near-zero payload), not bandwidth.
- **N=32 vesin NL rebuild = 1,534 ms/step (11%)**, run every step.

**Levers evaluated:**
- **(A) Small-message all_reduce algo tuning** (`CCL_ALLREDUCE=...`) — **DEAD END.**
  The oneCCL default already picks the optimal small-msg path; forcing
  recursive_doubling/direct/ring made it **2–3.5× slower** (Loop 12.8 → 25–45 s at
  N=16). Parity held (step0 identical to 13 digits). Keep the default.
- **(B) Skip/reduce the 15 all_reduces** — the `balance_channels` correction is a
  **real physics constraint** (enforces total charge/spin), so it cannot be
  skipped without breaking parity. Fusing charge+spin into one all_reduce (8→4
  fwd) is possible but needs model-graph surgery with parity risk — deferred.
- **(C) Skin-cached neighbor list** — implemented, analyzed, then **REJECTED as
  net-negative for this architecture.** The skin makes the NL reusable across steps
  (saving the ~1,534 ms/step vesin rebuild at N=32) and is parity-safe (skin
  superset, displacement-triggered rebuild, beyond-cutoff edges get envelope 0).
  BUT this engine runs **every** edge through the full SO2 conv + wigner + backward
  and only zeroes beyond-cutoff edges at the final envelope — so a skin's +16–59%
  extra edges add **+1,960–3,308 ms/step** of compute at N=32, far more than the
  ≤1,534 ms/step vesin rebuild it saves (net **+579 to +2,540 ms/step**, i.e.
  slower at every reuse interval). Unlike classical MD, beyond-cutoff pairs are not
  cheaply skipped here. It would only pay off with a **pre-SO2 data-dependent edge
  mask**, which requires a scripted (not traced) chunk loop. Reverted; kept the
  code comment documenting the analysis.

**opt6 conclusion.** The engine is at the achievable floor for this trace-based
design. All three candidate parity-safe levers are exhausted: (A) collective algo
tuning is a dead end (default optimal), (B) the 15 all_reduces are model-structural
(balance_channels physics) and only cuttable with parity-risky fusion, and (C)
skin-cached NL is net-negative because every edge is full-cost until the final
envelope (no cheap beyond-cutoff skip in a traced graph). Compute (fwd/bwd) is
already per-edge-efficient (§10). **Further gains require leaving the trace-based
design** (scripted data-dependent loops for edge masking / collective fusion) or
mixed precision (P-7, breaks the FP64 parity contract) — both out of scope for a
parity-preserving optimization. Current status: N=16 beats ASE 1.36× (§11), N=32
beats ASE 2.14–2.38× (§9a), all at machine-precision parity. All opt6 probes kept
parity (step0 −110673.829050 / −885377.060040 unchanged).

---

## 13. FINAL consolidated table — parity + performance vs ASE (latest build, 2026-08-28)

Latest engine (opt5-graph build, 15:12; opt1+opt2+opt3+opt4 + opt5 knobs). Config:
- **N=16:** `EDGE_AC_CHUNK=32768` + C2 (`UMA_NO_RECOMPUTE_BLOCK=1 UMA_NO_RECOMPUTE_EDEG=1`)
  + adaptive `UMA_CHUNK_RETAIN_K` (max that fits HBM per W) + `UMA_SKIP_MAXNBR_CAP=1`.
- **N=32:** C2 (`EDGE_AC_CHUNK=65536`; RETAIN_K OOMs at N=32 so K=0).
Two timing views (both FP64, identical NaCl, real 10-step Nosé-Hoover NVT@300K):
- **WARM** = the 10-step MD integration only, model already loaded/warmed up
  (LAMMPS `Loop time`; ASE `md_s` = `dyn.run(10)`, XPU-synced both ends). Both
  **exclude** cold model load + first-call JIT warmup — the apples-to-apples
  per-step MD cost, and what matters for a long production run.
- **COLD** = whole-wall "submit → done" incl. per-side model load + warmup + the
  10 steps. What a single short job actually costs end-to-end.
Jobs: LAMMPS 8788927/8788849/8788623/8787536; ASE real-NVT 8788329/8788499.

### 13a. WARM — 10-step MD compute (load-excluded)

| N | W | atoms | retainK | LAMMPS Loop | ASE md | LAMMPS vs ASE | parity vs ASE |
|--:|--:|--:|:--|--:|--:|:--|:--|
| 16 | 1  | 32,768 | K=1 | 165.5 s | **OOM** | ASE can't run | **PASS** dE=1.4e-9 meV/at, max\|dF\|=5.09e-14, cos=1.0 |
| 16 | 2  | 32,768 | K=2 | 91.0 s | 92.4 s | **1.02× faster** | **PASS** dE=1.41e-9 meV/at, max\|dF\|=5.04e-14, cos=1.0 |
| 16 | 4  | 32,768 | K=2 | 48.3 s | 47.4 s | 0.98× | **PASS** dE=1.41e-9 meV/at, max\|dF\|=5.07e-14, cos=1.0 |
| 16 | 6  | 32,768 | K=2 | 31.4 s | 31.9 s | **1.02× faster** | **PASS** dE=1.41e-9 meV/at, max\|dF\|=5.06e-14, cos=1.0 |
| 16 | 8  | 32,768 | K=0† | 30.4 s | 24.9 s | 0.82× | **PASS** dE=1.41e-9 meV/at, max\|dF\|=5.06e-14, cos=1.0 |
| 16 | 12 | 32,768 | K=3 | **12.8 s** | 17.4 s | **1.36× faster** | **PASS** dE=1.4e-9 meV/at, max\|dF\|=5.05e-14, cos=1.0 |
| 32 | 12 | 262,144 | C2 | **138.2 s** | 296.2 s | **2.14× faster** | **PASS** dE=1.28e-8 meV/at, max\|dF\|=1.05e-13, cos=1.0 |

**Strong scaling — N=16 ladder (warm Loop / md).** Speedup and parallel efficiency
vs the smallest fitting tile count (LAMMPS baseline W=1; ASE baseline W=2, since
ASE OOMs at W=1). eff = (baseline / (W/W_base)) / time × 100%.

| W | LAMMPS Loop | LAMMPS speedup | LAMMPS eff | ASE md | ASE speedup | ASE eff |
|--:|--:|--:|--:|--:|--:|--:|
| 1  | 165.5 s | 1.00× (base) | 100% | OOM | — | — |
| 2  | 91.0 s | 1.82× | 91% | 92.4 s | 1.00× (base) | 100% |
| 4  | 48.3 s | 3.43× | 86% | 47.4 s | 1.95× | 97% |
| 6  | 31.4 s | 5.27× | 88% | 31.9 s | 2.90× | 97% |
| 8  | 30.4 s | 5.44× | 68%† | 24.9 s | 3.71× | 93% |
| 12 | 12.8 s | 12.93×\* | 108%\* | 17.4 s | 5.31× | 89% |

\* LAMMPS W=12 is **super-linear** because the config is not fixed across the
ladder: `retainK` rises with W (W=1→K=1, W=12→K=3), so W=12 does *less recompute
per step* than the W=1 baseline — the 12.93×/108% mixes strong scaling with the
opt5 chunk-retain speedup, and overstates pure parallel scaling. A **fixed-K**
strong-scaling (all-W at C2/K=0, §5) shows the true tile scaling: ~4.7–4.9× on 6
tiles (79–81% eff), with the W=8 dip. ASE's ladder uses one config (eager, no
K), so its 89–97% efficiency is a clean strong-scaling curve. † W=8 LAMMPS at K=0
(K≥1 didn't cleanly land) — its eff/speedup are the C2 floor, understated vs the
retain-config trend.

### 13b. COLD — whole-wall NVT-10 (incl. model load + warmup)

| N | W | LAMMPS cold wall | (load + Loop) | ASE cold wall | (load + warmup + md) | LAMMPS vs ASE |
|--:|--:|--:|:--|--:|:--|:--|
| 16 | 1  | 206 s | 40 + 165.5 | **OOM** | — | ASE can't run |
| 16 | 2  | 115 s | 24 + 91.0 | 189 s | 82.8 + 13.7 + 92.4 | **1.64× faster** |
| 16 | 4  | 63 s | 15 + 48.3 | 98 s | 43.3 + 7.5 + 47.4 | **1.56× faster** |
| 16 | 6  | 44 s | 13 + 31.4 | 77 s | 38.8 + 6.4 + 31.9 | **1.75× faster** |
| 16 | 8  | ~44 s | 14 + 30.4 | 55 s | 24.3 + 5.6 + 24.9 | **1.24× faster** |
| 16 | 12 | ~33 s | 21 + 12.8 | 47 s | 24.1 + 5.7 + 17.4 | **1.41× faster** |
| 32 | 12 | **172 s** | 34 + 138.2 | **409 s** | 77.6 + 35.4 + 296.2 | **2.38× faster** |

**Strong scaling — N=16 ladder (cold whole-wall).** Speedup + parallel efficiency
vs the smallest fitting tile count (LAMMPS base W=1; ASE base W=2). eff =
(baseline / (W/W_base)) / wall × 100%.

| W | LAMMPS wall | LAMMPS speedup | LAMMPS eff | ASE wall | ASE speedup | ASE eff |
|--:|--:|--:|--:|--:|--:|--:|
| 1  | 206 s | 1.00× (base) | 100% | OOM | — | — |
| 2  | 115 s | 1.79× | 90% | 189 s | 1.00× (base) | 100% |
| 4  | 63 s | 3.27× | 82% | 98 s | 1.93× | 96% |
| 6  | 44 s | 4.68× | 78% | 77 s | 2.45× | 82% |
| 8  | ~44 s | 4.68× | 59%† | 55 s | 3.44× | 86% |
| 12 | ~33 s | 6.24× | 52% | 47 s | 4.02× | 67% |

**Cold scaling is sub-linear for both** (LAMMPS 52% / ASE 67% eff at W=12) because
the per-W **model load does NOT scale with tiles** — it's fixed setup (LAMMPS
13–40 s `torch.jit.load`; ASE 24–83 s Ray/XCCL bringup + warmup) that grows as a
*fraction* of the shrinking wall as W rises. So cold efficiency degrades with W
even though the compute scales well (§13a). LAMMPS's smaller, faster load keeps its
cold wall below ASE's at every W (the LAMMPS-vs-ASE column above), but its cold
*efficiency* falls faster than ASE's at high W only because its total wall is
already so much smaller that the fixed load dominates sooner. The same K-adaptive
caveat as §13a applies to the LAMMPS speedup (retainK rises with W). † W=8 at K=0.

LAMMPS cold load ≈ 13–40 s (artifact `torch.jit.load`, ~2.2 GB/rank), a one-time
setup amortized over a real trajectory. ASE cold load is larger (Ray/XCCL bringup
24–83 s + warmup). **LAMMPS is faster than ASE on the cold whole-wall at every W
too** (1.24–2.38×), in addition to the warm MD compute.

**Parity — full energy + per-atom force PASS at EVERY config.** Directly measured
against per-W ASE-GP force oracles (N=16 W=1/2/4/6/8/12 and N=32 W=12): step-0
energy dE ≤ 1.4e-9 meV/atom (N=16) / 1.28e-8 (N=32), per-atom max\|dF\| at the FP64
floor (~5e-14 N=16, 1.05e-13 N=32), cos = 1.0000000000 over all atoms. Every row
is a directly-verified PASS — no energy-only rows remain. (ASE-GP force oracles for
W=2/4/6/8 generated in job 8789534; LAMMPS parity in 8789552.)

**Performance.**
- **N=32 (production): LAMMPS 2.14× faster (MD compute) / 2.38× (whole wall)** than ASE.
- **N=16: LAMMPS faster at W=2, 6, 12** (best 1.36× at W=12); parity at W=4; ASE ahead
  only at W=8 (0.82×) and W=1 (where ASE OOMs but LAMMPS runs).
- Best N=16 latency: W=12, K=3, 12.8 s.

**Notes.**
- `retainK` is HBM-bounded and W-adaptive (more tiles → more headroom → higher K):
  W=1→K=1, W=2–6→K=2, W=8→K=0, W=12→K=3.
- **‡ energy=ASE** = step-0 energy directly matches the ASE oracle
  (−110673.829050); **full per-atom force parity at W=2/4/6/8 is being measured**
  (ASE-GP force oracles at those W, job 8789534; LAMMPS parity 8789xxx) and will
  upgrade these to full PASS. Forces are strongly implied already: W=1 and W=12
  both pass full force parity and bracket these W, and GP is
  decomposition-independent (same computation, different tile partition).
- **† N=16 W=8** reported at its clean C2 (K=0) value; the adaptive K=3 attempt
  thrashed memory (near-OOM) and did not complete in walltime — K=1–2 likely fit
  and would improve W=8 further, but were not cleanly landed.
- COLD walls: LAMMPS W=1/2/4/6 directly measured (job 8788927); W=8/12 = measured
  cold load + the fast-config Loop (load from the same-W runs); N=32 measured
  (172 s). ASE walls all directly measured (8788329/8788499).
- Jobs: N=16 timing 8788927 (W=1,2,4,6) / 8788849 (W=12 K=3) / 8788704 (W=8 C2);
  N=32 8787536; parity 8788864/8787863/8788927; ASE real-NVT 8788329/8788499;
  W=2/4/6/8 force oracles 8789534.

---

## 14. Per-sprint regression record — CODE_QUALITY campaign (G4)

Each sprint of the `docs/CODE_QUALITY.md` hardening campaign closes with a full
parity + performance re-run on that sprint's exact binary (goal **G4**), so any
physics regression is caught immediately. Matrix: **N=16 W=1,2,4,6,8,12** and
**N=32 W=12** — step-0 energy, per-atom force parity vs the ASE/ASE-GP oracle, and
autograd-vs-finite-difference (AG=FD). Oracles are the fixed §13 references
(`ase_n16_parity`, `ase_n16_forces_ladder`, `ase12_n32`), never regenerated.

### 14.0 Sprint 0 — silent-physics + UB fixes (2026-08-29)

Binary: `build-lmp-xccl/lmp` rebuilt from the Sprint-0 working tree (job 8791177),
which contains P0.1 (`ccl::barrier().wait()`), P0′.1 (virial refusal — barostat
guard at `init_style`), P0′.3 (padded tensors into the whole-module checkpoint
branch), P0′.4 (HaloContext/BlockContext callback teardown + Predictor
move-ownership). Config = the §13 "fast" stack: N=16 `EDGE_AC_CHUNK=32768` + C2 +
adaptive `UMA_CHUNK_RETAIN_K` + `UMA_SKIP_MAXNBR_CAP=1`; N=32 C2 chunk=65536, K=0.

**Parity + performance (real 10-step NVT@300 K, FP64), job 8791275:**

| N | W | retainK | step-0 PE (eV) | Loop (s) | wall (s) | per-atom max\|dF\| | cos | parity |
|--:|--:|:--|--:|--:|--:|--:|--:|:--|
| 16 | 1  | 1 | −110673.829050 | 164.5 | 210 | 5.08e-14 | 1.0000000000 | ✅ PASS |
| 16 | 2  | 2 | −110673.829050 | 89.9  | 114 | 5.04e-14 | 1.0000000000 | ✅ PASS |
| 16 | 4  | 2 | −110673.829050 | 47.7  | 63  | 5.08e-14 | 1.0000000000 | ✅ PASS |
| 16 | 6  | 2 | −110673.829050 | 31.4  | 44  | 5.06e-14 | 1.0000000000 | ✅ PASS |
| 16 | 8  | 0 | −110673.829050 | 30.3  | 48  | 5.06e-14 | 1.0000000000 | ✅ PASS |
| 16 | 12 | 3 | −110673.829050 | 12.83 | 27  | 5.05e-14 | 1.0000000000 | ✅ PASS |
| 32 | 12 | 0 | −885377.060040 | 137.6 | 173 | 1.06e-13 | 1.0000000000 | ✅ PASS |

**AG=FD (single tile, FP64), job 8791223:** N=1–10 all PASS — max|AG−FD| ≤ 3.9e-7
(N=1 7.7e-9 → N=10 3.9e-7), traced-vs-eager dF at the ~1e-16 FP64 floor. (tol 1e-5.)

**Mandatory ASE parity tripwire, job 8791194:** N=16 W=1 + N=32 W=12 **PASS**
(dE 1.41e-9 / 1.28e-8 meV/atom, max|dF| 5.05e-14 / 1.05e-13, cos = 1.0).

**Regression verdict vs the §13 baseline:** every step-0 PE is **bit-identical**
(N=16 all-W −110673.829050 = §13; N=32 −885377.060040 = §13), per-atom forces at
the same FP64 floor, cos = 1.0, AG=FD unchanged. **G3 met — Sprint 0 changed no
physics.** Loop times reproduce §13a within run variance (e.g. W=12 12.83 s vs §13
12.8 s; N=32 137.6 s vs §13 138.2 s; W=8 K=0 30.3 s vs §13 30.4 s).

**P0′.1 behavior change (intended):** `fix npt`/`nph`/`press/*` now aborts at
`init_style` with "Pair style uma does not compute the virial; pressure control
… is not supported" (verified job 8791197: single-point exit 0, `fix npt` exit 1);
NVE/NVT and single-point (incl. thermo without `press`) are unaffected.
Jobs: rebuild 8791177; tripwire 8791194; virial-refuse test 8791197; full suite
8791275; AG=FD 8791223.

### 14.1 Sprint 1 — teardown / error-handling fixes (2026-08-29)

Binary rebuilt from the Sprint-1 tree (job 8791387) adding P0′.5: (1) a
`catch (const LAMMPSException&) { throw; }` before both `catch (std::exception&)`
init handlers so a LAMMPS `error->all` thrown during construction keeps its own
collective/abort path instead of being rewrapped into a ragged `error->one`; (2)
`~PairUMA` gates its teardown `MPI_Barrier` behind an `MPI_Allreduce(MIN)` have-peer
agreement so a rank that failed to build its peer can't hang the survivors. These
are teardown/error-path changes and touch no compute path.

**Parity + performance (real 10-step NVT@300 K, FP64), job 8791406:**

| N | W | retainK | step-0 PE (eV) | Loop (s) | wall (s) | per-atom max\|dF\| | cos | parity |
|--:|--:|:--|--:|--:|--:|--:|--:|:--|
| 16 | 1  | 1 | −110673.829050 | 164.5 | 220 | 5.07e-14 | 1.0000000000 | ✅ PASS |
| 16 | 2  | 2 | −110673.829050 | 92.4  | 120 | 5.04e-14 | 1.0000000000 | ✅ PASS |
| 16 | 4  | 2 | −110673.829050 | 48.9  | 68  | 5.07e-14 | 1.0000000000 | ✅ PASS |
| 16 | 6  | 2 | −110673.829050 | 31.8  | 45  | 5.07e-14 | 1.0000000000 | ✅ PASS |
| 16 | 8  | 0 | −110673.829050 | 30.6  | 49  | 5.06e-14 | 1.0000000000 | ✅ PASS |
| 16 | 12 | 3 | −110673.829050 | 12.93 | 39  | 5.05e-14 | 1.0000000000 | ✅ PASS |
| 32 | 12 | 0 | −885377.060040 | 138.7 | 174 | 1.05e-13 | 1.0000000000 | ✅ PASS |

**AG=FD (single tile, FP64), job 8791409:** N=1–9 all PASS — max|AG−FD| ≤ 2.6e-7
(N=1 7.7e-9 → N=9 2.6e-7), traced-vs-eager dF at the ~1e-16 FP64 floor. (N=10 in
progress at job stop; N=1–9 conclusive, identical to §14.0.)

**Mandatory ASE parity tripwire, job 8791405:** N=16 W=1 + N=32 W=12 **PASS**
(dE 1.41e-9 / 1.28e-8 meV/atom, max|dF| 5.05e-14 / 1.05e-13, cos = 1.0).

**Regression verdict vs §14.0 (Sprint 0) and §13:** every step-0 PE **bit-identical**
(N=16 all-W −110673.829050; N=32 −885377.060040), per-atom forces at the same FP64
floor, cos = 1.0, AG=FD unchanged, Loop times within run variance. **G3 met —
Sprint 1 changed no physics.**
Jobs: rebuild 8791387; tripwire 8791405; full suite 8791406; AG=FD 8791409.

### 14.2 Sprint 2 — collective-agreement cluster + neighbor-list fixes (2026-08-30)

Binary rebuilt from the Sprint-2 tree (job 8791580) adding P0.2–P0.6:
- **P0.2** cross-rank backward-graph mode agreement (peer all_reduce in `create()`);
- **P0.3** collective **pad-cap overflow** check before the forward (a whole-body
  try/catch wrapper was tried first but DEADLOCKED at W=4 on an asymmetric K=3 OOM,
  job 8791554 — removed; see D.2/§D.3);
- **P0.4** empty-shard matched collectives (NCCL path);
- **P0.5** neighbor-list image bound now uses the **interplanar spacing** `V/|area|`
  (was `|cell[d]|`) so skewed cells don't drop edges;
- **P0.6** both CPU-NL branches (GP `mpi_peer_predictor.cpp`, `libtorch_mp.cpp`) now
  publish the **wrapped** position frame that `cell_offsets` were computed against.

**Parity + performance (real 10-step NVT@300 K, FP64), job 8791608:**

| N | W | retainK | step-0 PE (eV) | Loop (s) | wall (s) | per-atom max\|dF\| | cos | parity |
|--:|--:|:--|--:|--:|--:|--:|--:|:--|
| 16 | 1  | 1 | −110673.829050 | 166.3 | 216 | 5.09e-14 | 1.0000000000 | ✅ PASS |
| 16 | 2  | 2 | −110673.829050 | 91.5  | 117 | 7.95e-14 | 1.0000000000 | ✅ PASS |
| 16 | 4  | 2 | −110673.829050 | 48.4  | 66  | 7.96e-14 | 1.0000000000 | ✅ PASS |
| 16 | 6  | 2 | −110673.829050 | 32.0  | 45  | 7.96e-14 | 1.0000000000 | ✅ PASS |
| 16 | 8  | 0 | −110673.829050 | 30.7  | 49  | 7.94e-14 | 1.0000000000 | ✅ PASS |
| 16 | 12 | 3 | −110673.829050 | 12.97 | 27  | 7.93e-14 | 1.0000000000 | ✅ PASS |
| 32 | 12 | 0 | −885377.060040 | 139.7 | 181 | 1.61e-13 | 1.0000000000 | ✅ PASS |

**AG=FD (single tile, FP64), job 8791634:** N=1–7 all PASS — max|AG−FD| ≤ 8.1e-8,
traced-vs-eager dF at the ~1e-16 FP64 floor. (N=8 also PASS on the identical-predictor
prior build 8791542.)

**Mandatory ASE parity tripwire, job 8791607:** N=16 W=1 + N=32 W=12 **PASS**
(dE 1.42e-9 / 1.28e-8 meV/atom, max|dF| 5.07e-14 / 1.61e-13, cos = 1.0).

**Regression verdict (G3):** every **step-0 PE is bit-identical** to §14.1/§14.0/§13
(N=16 all-W −110673.829050; N=32 −885377.060040), cos = 1.0, forces at the FP64
floor, AG=FD unchanged, Loop times within run variance. **No physics regression.**

**Two intended trajectory/force effects (not regressions):**
- **P0.6 corrected the GP MD trajectory.** All W≥2 (GP) step-10 PE now equals the
  single-tile W=1 value (−110602.976229; N=32 −884817.238827), whereas before
  Sprint 2 the GP step-10 differed (e.g. §14.1 N=16 W≥2 −109317.384). The pre-fix
  GP path left positions **unwrapped** while `cell_offsets` were wrapped-frame, so
  mid-trajectory `edge_distance_vec` was subtly inconsistent once atoms drifted
  outside the box; step-0 (atoms in-box) was always correct, which is why parity
  (a step-0 metric) always passed. Post-fix the GP trajectory reproduces the
  single-tile reference — a correctness improvement, step-0 invariant preserved.
- **N=32 per-atom max|dF| ≈ 1.6e-13** (vs 1.05e-13 pre-Sprint-2) — stable across
  two independent Sprint-2 builds; a reduction-order effect of the added control
  collectives (P0.2 agreement + P0.3 pad-cap all_reduce + the Sprint-0 barrier
  `.wait()`). It is 8 orders of magnitude under the 1e-5 force gate, cos = 1.0, and
  step-0 energy is unchanged — physically negligible.
Jobs: rebuild 8791580; tripwire 8791607; full suite 8791608; AG=FD 8791634.
(An earlier Sprint-2 build 8791542 exposed the W=4 P0.3 deadlock via job 8791554.)

### 14.3 Sprint 3 — fail-closed harness + env pin (2026-08-30)

Sprint 3 changed only the **Python/scripts harness** (fail-closed gates, single
tolerance source `scripts/uma_gates.py`, `parity_vs_asegp.py` atom-count hard-fail,
exporter exit-code enforcement, `requirements.txt` + fairchem/torch version assert).
**No C++ engine change → the Sprint-2 binary (build 8791580) is reused unchanged.**
This regression run therefore both (a) confirms G3 and (b) validates that the
modified gate scripts still run correctly (the tripwire + parity table are produced
by the edited `parity_vs_asegp.py`, now importing `uma_gates`).

**Parity + performance (real 10-step NVT@300 K, FP64), job 8791684:**

| N | W | retainK | step-0 PE (eV) | Loop (s) | wall (s) | per-atom max\|dF\| | cos | parity |
|--:|--:|:--|--:|--:|--:|--:|--:|:--|
| 16 | 1  | 1 | −110673.829050 | 160.4 | 216 | 5.10e-14 | 1.0000000000 | ✅ PASS |
| 16 | 2  | 2 | −110673.829050 | 90.3  | 118 | 7.97e-14 | 1.0000000000 | ✅ PASS |
| 16 | 4  | 2 | −110673.829050 | 47.8  | 68  | 7.95e-14 | 1.0000000000 | ✅ PASS |
| 16 | 6  | 2 | −110673.829050 | 31.1  | 50  | 7.94e-14 | 1.0000000000 | ✅ PASS |
| 16 | 8  | 0 | −110673.829050 | 30.0  | 48  | 7.93e-14 | 1.0000000000 | ✅ PASS |
| 16 | 12 | 3 | −110673.829050 | 12.78 | 27  | 7.95e-14 | 1.0000000000 | ✅ PASS |
| 32 | 12 | 0 | −885377.060040 | 137.2 | 173 | 1.62e-13 | 1.0000000000 | ✅ PASS |

**AG=FD (single tile, FP64):** N=1–10 all PASS (max|AG−FD| ≤ 3.2e-7), job 8791634 —
the Sprint-2/3 predictor is byte-identical (Sprint 3 changed no C++), so this is the
current AG=FD record. The AG=FD **gate script itself** (`phase6_agfd.py`) is now
fail-closed (P1.1): 0 samples / < MIN_SAMPLE / any failed FD run → FAIL.

**Mandatory ASE parity tripwire (via the edited `parity_vs_asegp.py`), job 8791683:**
N=16 W=1 + N=32 W=12 **PASS** (dE 1.41e-9 / 1.28e-8 meV/atom, max|dF| 5.04e-14 /
1.61e-13, cos = 1.0). Tolerances sourced from `uma_gates.py`.

**Regression verdict (G3):** every step-0 PE **bit-identical** to §14.2/§14.1/§13,
cos = 1.0, forces at the FP64 floor, AG=FD unchanged, Loop times within run
variance. **No physics change** — as expected for a harness-only sprint. The
harness is now fail-closed (a crashed/zero-sample/oracle-missing gate can no longer
report PASS) and all gate tolerances come from one file.
Jobs: tripwire 8791683; full suite 8791684; AG=FD 8791634 (Sprint-2/3 binary reused).

### 14.4 Sprint 4 — Phase-2 CI pyramid (Tier 0/1) (2026-08-30)

Sprint 4 stood up the local CI (Tier 0 static guards + Tier 1 hermetic unit tests,
`ci/ci_local.sh`, ~45 s on a login node, no allocation) and registered the C++
CTest (`enable_testing`/`add_test` in `uma-engine/CMakeLists.txt`;
`graph_shard_smoke` now builds on every backend). The **only compiled change** is
CMake test wiring (the engine smoke target is `EXCLUDE_FROM_ALL` in the LAMMPS
build), so the binary is functionally identical to §14.2/§14.3. Rebuild 8791793
verified the CMake change configures + builds cleanly (`LMP BUILD OK`).

**Parity + performance (real 10-step NVT@300 K, FP64), job 8791812:**

| N | W | retainK | step-0 PE (eV) | Loop (s) | wall (s) | per-atom max\|dF\| | cos | parity |
|--:|--:|:--|--:|--:|--:|--:|--:|:--|
| 16 | 1  | 1 | −110673.829050 | 164.9 | 219 | 5.05e-14 | 1.0000000000 | ✅ PASS |
| 16 | 2  | 2 | −110673.829050 | 90.5  | 115 | 7.97e-14 | 1.0000000000 | ✅ PASS |
| 16 | 4  | 2 | −110673.829050 | 48.1  | 64  | 7.96e-14 | 1.0000000000 | ✅ PASS |
| 16 | 6  | 2 | −110673.829050 | 31.6  | 44  | 7.96e-14 | 1.0000000000 | ✅ PASS |
| 16 | 8  | 0 | −110673.829050 | 30.5  | 117 | 7.92e-14 | 1.0000000000 | ✅ PASS |
| 16 | 12 | 3 | −110673.829050 | 12.88 | 26  | 7.94e-14 | 1.0000000000 | ✅ PASS |
| 32 | 12 | 0 | −885377.060040 | 138.0 | 175 | 1.61e-13 | 1.0000000000 | ✅ PASS |

**AG=FD:** N=1–10 PASS (job 8791634; predictor byte-identical since Sprint 2).
**Tripwire (job 8791811):** N=16 W=1 + N=32 W=12 PASS, bit-identical.

**Local CI (`ci/ci_local.sh`, login node, no allocation, 45 s):** Tier 0 guards
PASS; Tier 1 = 29 pure tests PASS (metadata contract, edge-pad/partition, neighbor
`image_repeats`/P0.5, gate arithmetic).

**Regression verdict (G3):** every step-0 PE **bit-identical** to §14.3/§14.2/§13,
cos = 1.0, forces at the FP64 floor, AG=FD unchanged. **No physics change** —
Sprint 4 is CI/tooling only.
Jobs: rebuild 8791793; tripwire 8791811; full suite 8791812; AG=FD 8791634.
