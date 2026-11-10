# Nitro-inspired activities — LibTorch UMA LAMMPS (authoritative plan, revision 4)

Scope: **only activities whose idea comes from UMA Nitro**
(`fairchem/.../uma_nitro/_inference.py`). Nothing else is planned here.
This file is **self-contained**: the shared test protocol (correctness bars, timing, ABBA, effect
threshold, results template) is in section P below. The earlier full plan
(`uma_lammps_perf_test_plan.md` in the workdir root) is reference only.

Status: harness written (`pbs/nitro/`); baseline not yet verified in this tree. Gains are estimates.

Fixed conditions: FP64 only; stock UMA model unchanged; gate = NaCl 14³ (21,952 atoms), W=12,
NVT + NPT 10 steps; one branch per activity, immutable tag before each submit.

---

> **Revision 4 (2026-11-10).** All work now happens in a fresh clone,
> `/lus/flare/projects/MatSciAI/xiaoliyan/workdir/lammps-uma-nitro`, on branch `uma-nitro`.
> This file is the authoritative plan and lives in that clone as `NITRO_PLAN.md`.
> `uma_lammps_perf_test_plan.md` (workdir root) is the earlier full plan, kept for reference only;
> its `perf/*` names are superseded by the `nitro/*` names below.

## R. Workspace and version control (revision 4 — overrides anything below that says otherwise)

### R.1 Repository state (done)

| Item | Value |
|---|---|
| Path | `/lus/flare/projects/MatSciAI/xiaoliyan/workdir/lammps-uma-nitro` |
| Source | `https://github.com/williamyxl/lammps-uma.git`, branch `lmp-uma-release` (single-branch clone; `origin` = GitHub) |
| Base commit | `a240ba3485` ("fix export MPI ranks"). Verified identical to the HEAD of `lammps-uma-release`. |
| Working branch | `uma-nitro` (created from `lmp-uma-release`; local only, **not pushed**) |
| Clone size | 1.4 GB |

### R.2 Where things may be edited

| Tree | Rule |
|---|---|
| `lammps-uma-nitro/` | **All** edits, builds, PBS scripts and outputs for this plan |
| `lammps-uma-release/` | Not touched (the PR-cleanup wave rule still applies to it). Its Wave E artifacts may be **read** as an input to the BASE arm (see R.4) |
| `lammps-uma/` | Read-only (campaign clone) |
| `hen/sqs_sampling/` | Read-only |

### R.3 Branch and tag names

A branch called `uma-nitro` and a branch called `uma-nitro/<id>` cannot coexist in git (a ref cannot be both a file and a directory). So:

| Name | Meaning |
|---|---|
| `uma-nitro` | Integration branch. Carries the harness (PBS templates, scripts, this plan). |
| `nitro-origin` (tag) | `a240ba3485`, the starting point. Create before any change. |
| `nitro-base` (annotated tag) | The verified `uma-nitro` head after the harness commit. This is the **BASE arm** and never moves. |
| `nitro/<id>` (branches) | One per activity, cut from `nitro-base`: `nitro/n1-a`, `nitro/n1-b`, `nitro/n2-a`, `nitro/n2-b`, `nitro/n3-cap`, `nitro/n4-a`, `nitro/n4-b`, `nitro/n5-f1` … `nitro/n5-f7`, `nitro/n5-proto` |
| `nitro/<id>/r<k>` (annotated tags) | Immutable measurement tag, created before every `qsub`. Never amend a tagged commit. |
| `nitro/stack/s<k>` | Merge-commit stacks of passing options (no squash, no rebase). |

The rest of the protocol (flag-gated arms with BASE as default, per-arm build dir, `meta.json`, failed arms stay, no cherry-pick of result commits) is unchanged from the earlier plan's §0.1.

### R.4 Effects of the fresh clone on earlier blockers

| Earlier item | Effect |
|---|---|
| B5 (untracked `pbs/` and `PR_CLEANUP_PLAN.md`) | **Resolved.** The clone has no untracked files, so a clean-tree check is meaningful. Rule: commit harness files on `uma-nitro`; keep `pbs/out/`, `pbs/bin/`, `build-*/` out of git via `.git/info/exclude`. |
| Missing build script | **Resolved.** The clone has no `scripts/build_lammps_uma.sh` either. The working recipe is the `cmake` call in `lammps-uma-release/pbs/pr_gate/run_waveE.pbs` (GCC 13.4, MPICH 5.0, torch cmake path from `fxpu`, `UMA_ENGINE_USE_XPU=ON`, `UMA_ENGINE_USE_XCCL=ON`, `UMA_ICPX`, `ONECCL_ROOT=$CONDA_PREFIX`, `PKG_ML-UMA=ON`, `PKG_KOKKOS=OFF`). Copy it into the clone as `pbs/nitro/build.inc`, with `LMP_BUILD` set per arm (`build-nitro/<sha7>/`). Wave E built inside the debug job; keep that. |
| `pbs/` harness missing | **Written** in `pbs/nitro/` (see P.4); not yet committed or verified on a compute node. |
| Existing W=12 N=14 artifact | Lives in `lammps-uma-release/pbs/out/pr_gate_wE_9040639.../n14/artifact`. Valid as a BASE input only while the exporter and engine sources equal `a240ba3485`; any arm that changes export re-exports into the nitro tree. |

### R.5 Standing rules for every job in this tree

- Queues: `debug` or `debug-scaling` only; `walltime=01:00:00`.
- FP64 only for all model numerics and custom kernels.
- No torch/XPU compute on login nodes. Compiling is fine.
- Syntax-check subagent (`grok-4.7-high`, type `shell`) after every code edit and before every `qsub`.
- Disk: Flare project quota is 27.57T used of 40T (69%). Each W=12 N=14 artifact is about 27 GB.

### R.6 Execution sequence (revised)

| Step | Action | Needs compute node |
|---|---|---|
| 0 | Clone and create `uma-nitro` | no (done) |
| 1 | Tag `nitro-origin`; add `.git/info/exclude`; add `pbs/nitro/` (build include, env include, export script, Wave E job as a BASE template, ABBA runner, `meta.json` writer, force/virial compare script); syntax-check; commit on `uma-nitro`; tag `nitro-base` after step 2 | no |
| 2 | Verify BASE: build in the nitro tree, run the NaCl 14³ W=12 NVT+NPT gate once, expect step-0 PE −74142.191907631001 eV and step-10 PEs matching Wave E to ≤ 1e-8 eV | yes (`debug`) |
| 3 | E0 noise floor and bitwise BASE repeat (3 runs in one job, then 3 jobs) | yes (`debug`) |
| 4 | S0 of the SYCL prototype (§4), on branch `nitro/n5-proto` | yes (`debug`) |
| 5 | N1 control: time the host graph build against the device build, ABBA | yes (`debug`) |
| 6 | Continue with S1–S5, then N2, N5 microbenchmarks, N3, N4, stack (§1, §5) | yes |

Steps 1–2 close "plan ready for execution" for the harness; the activities then proceed in the order in §1.

---

## P. Shared test protocol (self-contained)

### P.1 Correctness bars

Every arm is compared with a BASE run **in the same allocation**. Reference values for absolute sanity
(NaCl 14³, W=12, FP64, `uma-s-1p2`, seed 0): step-0 PE −74142.191907631001 eV; step-10 PE NVT
−74095.578889731943 eV, NPT −74098.199120459016 eV.

| Check | Pass bar | Notes |
|---|---|---|
| `lmp` exit code | 0 for NVT and NPT | |
| Thermo | steps 0…N present, `pe` and `temp` finite | |
| NPT | volume moves, no "virial refused" | needs virial, AC off |
| No hen code | no `UMA_HEN_ROOT` and no hen code paths (patches, shim, scripts) in the log | the checkpoint path under `hen/uma-cache` is allowed |
| Step-0 \|ΔPE\| vs BASE | ≤ 1e−8 eV | reorder-only changes gave 1.5e−11…5.8e−11 in Waves A–E |
| Forces | max\|ΔF\| ≤ max(1e−9 eV/Å, 10 × BASE-vs-BASE noise) | calibrate in E0 (`diff_forces.py`) |
| Virial (NPT) | max\|ΔW\| ≤ 1e−7 eV | calibrate in E0 |
| Bit-identity | exact | only for arms claiming "same ops, same order"; **first measure BASE-vs-BASE bitwise** (BASE has `index_add_`, which may not be reproducible); if not, relax to tolerance |
| Independent checks | relative force error to force RMS; finite-difference check on a few atoms; NVE 200-step drift | a noise-scaled bar alone can hide a real bug |
| Worker ladder | any arm touching the graph, autograd or kernels passes on a small DIAG cell at W=1 first, then W=2 → 4 → 6 → 12, before the W=12 gate counts | DIAG cells are labelled and never used to claim a result |
| NPT/virial parity | mandatory for N1, N2, N5 | stress path is `dE/dcell` through edge vectors |
| Forced edge cases | atoms outside the box, tiny cell (multi-image), triclinic cell, pair at exactly the cutoff, a >300-neighbor dense case, pad present, multi-chunk | the 10-step gate on uniform NaCl does not exercise these |

### P.2 Timing

- Timing runs: **30 steps** NVT; steps 0–4 are warm-up and are discarded. Gate runs of 10 steps are for correctness only.
- Statistic: **median** over steps 5–29 of `ms_total` and each `MP_PERF` field, with MAD, min and max (`compare.py perf`).
- Instrumentation: `MP_PERF` synchronizes, so compare arms with the same instrumentation on, and take one extra pair of runs with it off to confirm.
- **Effect threshold:** an arm wins only if the paired step-time difference has a 95% interval excluding 0 **and** the median gain exceeds 3 × the pooled MAD; treat 5% as a priority signal, not a pass/fail line, so small deterministic savings (N3, N4) can still be judged in a stack.
- Speedup = `median(BASE) / median(arm)`, labelled with cell, W and queue.

### P.3 Control of variables

1. One PBS job holds several arms in **ABBA** order (BASE, A, A, BASE).
2. One factor changes per comparison; binary, artifact, env, deck, seed, run length, rank-to-tile mapping stay frozen.
3. Gate new behaviour behind env vars or `pair_style` keywords with default = BASE, so one binary serves both arms where possible.
4. If an arm changes export, re-export and also run BASE on that re-exported artifact.
5. Fixed `velocity ... loop geom`-equivalent seed and an identical `data.nacl` (`make_nacl.py`).
6. Record hostnames and tile state; discard a job whose nodes throttle.
7. Every run writes `meta.json` (`meta.py`): git describe/SHA, branch, tag, `nitro-base` SHA, binary sha256, artifact id, env, hosts, queue, jobid.
8. Every PBS job uses a per-run `timeout`, records exit codes and keeps going after a failing arm; risky arms run last.

### P.4 Harness files (in `pbs/nitro/`)

| File | Purpose |
|---|---|
| `env.inc` | compute-node environment (`fxpu` conda env, per-arm `NITRO_BUILD`, checkpoint path) |
| `lib.inc` | clean-tree/tag check, runtime env, LAMMPS deck writer (optional force dump), `mpiexec` runner |
| `make_nacl.py` | NaCl N×N×N data and xyz files (same construction as Wave E) |
| `export_one_rank.sh` | one rank of the W=12 export |
| `run_gate.pbs` | build into `build-nitro/<sha7>/`, export (or reuse), NVT+NPT 10 steps, PE bars against the recorded values, force dump at steps 0 and 10, `SUMMARY.txt`, `meta.json`, `run.env` |
| `run_timing.pbs` | E0: `REPS` identical 30-step NVT runs in one job; MP_PERF medians, bitwise PE and force repeat |
| `compare.py` | `pe`, `bar`, `perf`, `bit` subcommands over LAMMPS and MP_PERF logs |
| `diff_forces.py` | frame-by-frame force dump comparison (max, rms, relative, bitwise atom count) |
| `meta.py` | writes `meta.json` |

Submission pattern (queue `debug` or `debug-scaling`, `walltime=01:00:00`, from `pbs/nitro/`, after the tag exists):

```bash
cd pbs/nitro
qsub -v NITRO_TAG=nitro/base-verify/r1 run_gate.pbs
qsub -v NITRO_TAG=nitro/e0/r1,NITRO_REF=<gate OUTROOT>,REPS=3 run_timing.pbs
```

### P.5 Results template

| Exp | Arm | Cell/W | Jobs | Median step (ms) | MAD | Speedup vs BASE | \|ΔPE\| (eV) | max\|ΔF\| (eV/Å) | Mem (GiB) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|

---

## 1. Activities

Each activity has: Nitro source idea, what we do, a **control** (the Nitro-referenced comparison that
decides go/no-go), and its own failure points.

### N1 — Device-side radius-graph build with fixed capacity and overflow flag
- **Nitro idea:** `padded_radius_graph` (`bucket_fill_direct` → `fused_candidates` → `graph_scan` → `edge_emit`) rebuilds the graph on the device every step; an overflow flag is checked after the build.
- **Do:**
  - N1-a: build the graph on XPU with composed torch ops, FP64. **Output order must equal the reference builder's order: by center ascending, then (distance, neighbor, ox, oy, oz)** (`emit_graph` in `neighbor_list.cpp`). That order is also the `max_neighbors` truncation tie-break, and it keeps the summation order identical to BASE.
  - N1-b: custom SYCL kernels with the same stages, only if N1-a is correct but slow.
  - Reuse the existing cross-rank pad-cap overflow check in `mpi_peer_predictor.cpp` (all-reduce of an overflow flag before the forward); add the device builder's own overflow flag to it. On overflow all ranks stop with an explicit error.
- **Control:** time the current host `build_neighbor_graph` + H2D copy with device events, then N1-a/N1-b in the same job (ABBA). Proceed only if N1-a ≤ ~25% of the host time.
- **Tests:** edge-set equality vs the current builder; ΔPE/ΔF/Δvirial vs BASE; NPT parity; W ladder.
- **Expected gain:** up to ~110 ms/step (~13%) if the host-build hypothesis holds (unverified).

### N2 — Target-sorted edge→node segmented reduction
- **Nitro idea:** `EdgeCSR` (edges sorted by target node) with `_segreduce_tgt_*` / `fused_m0_segreduce` instead of atomic scatter.
- **Starting point (verified in code):** the current host graph is already emitted by center ascending, `shard_edges` preserves that order, and the pad edges are **prepended** with center = the rank's first owned node, so the padded shard is already sorted by center. N2 does **not** depend on N1, as long as N1 keeps that order.
- **Do:**
  - N2-a: per-segment reduction with torch ops over the already center-sorted edges (offsets from a bincount/cumsum of centers).
  - N2-b: SYCL kernel, one work-group per segment, with custom backward.
- **Control:** microbenchmark vs the current `index_add_` on the real chunk shapes, forward and forward+backward. Proceed only if ≥ 1.3×.
- **Expected gain:** a few % if `index_add_` is hot, ~0 otherwise.

### N3 — Measured edge capacity with margin
- **Nitro idea:** `plan_capacities` / `_measured_caps` with `EDGE_MARGIN` (1.30) instead of a worst-case cap.
- **Do:** export with cap = measured max per-rank shard edges × {1.05, 1.10, 1.30}, rounded up (N3-a/b/c). Nitro's remeasure-and-retry is **not** copied (capacity is baked into the traced model); overflow is a hard error.
- **Control:** the three margins against the current cap 65 536.
- **Expected gain:** ≤ ~11% of edge work (unmeasured). Lowest priority.

### N4 — MOLE merge at export, keyed by composition
- **Nitro idea:** merge the mixture-of-linear-experts per system (`_composition_key`, `prepare_for_inference`).
- **Do:**
  - N4-a: fold the per-system expert coefficients at export; store composition + charge + spin in metadata and refuse a mismatch.
  - N4-b: fold other composition-constant subgraphs.
- **Control:** unfolded vs folded artifact; bit-level or tolerance equality of PE/F.
- **Expected gain:** ~1% time, some memory. Optional.

### N5 — Fused edge-chain blocks (Nitro kernel set)
Each block: **microbenchmark → one layer → all layers → gate**. Drop a block whose FP64 microbenchmark is under ~1.5× the unfused torch sequence.

| ID | Block | Nitro source |
|---|---|---|
| F1 | LayerNorm + SiLU | `_ln_silu_*` |
| F2 | Gate | `_gate_op` |
| F3 | Radial MLP chain | `_merged_radials`, `_unified_radials` |
| F4 | Wigner rotation + SO2 conv1 | `wigner_conv1_fused_*` |
| F5 | Inverse rotation + conv2 + edge→node reduce (uses N2) | `invrot_conv2_*`, `fused_m0_segreduce` |
| F6 | Hand-written backward for F1–F5 | `*_bwd_*` |
| F7 | Side-queue concurrency of independent sub-graphs | `_parallel`, `_side_streams` |

- **Backend:** must run in the Python-free C++ engine, so SYCL/oneDNN custom ops registered in C++ with an exporter change (Triton and CuTe are not usable as-is). F7 is **deferred** by default.
- **Expected gain:** ~100–200 ms (12–25%) if the edge chain is memory-bound; low confidence.

### Order
N1 → N2 → N5 microbenchmarks (F1, F2, F4, F5, F6, F3) → N5 in-model for blocks that passed → N3 → N4 → stack the passing activities (`nitro/stack/s<k>`) and run the gate.

### Handles
`nitro/n1-a`, `nitro/n1-b`, `nitro/n2-a`, `nitro/n2-b`, `nitro/n3-cap`, `nitro/n4-a`, `nitro/n4-b`, `nitro/n5-f1` … `nitro/n5-f7`, `nitro/stack/s<k>`; tags `nitro/<id>/r<k>` before each `qsub`.

---

## 2. Failure points and robustness (Nitro activities only)

Severity: **S** can ship wrong physics or invalidate results; **H** likely to block or waste effort; **M** reduces confidence.

### Applies to all
| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| X1 | H | Upside is capped: Nitro has nothing for the in-graph all-reduce wall (~390 of 828 ms). Realistic total ≈ 1.2–1.5×. | State the cap before work starts |
| X2 | H | Nitro's Triton/CuTe kernels cannot run in the Python-free C++ engine. | Prototype one trivial SYCL op end-to-end (export → load → autograd) first |
| X3 | S | Checkpoint mismatch: Nitro is validated for `uma-s-1p2p1`; local is `uma-s-1p2` (composition embedding differs). | Reuse edge-side designs only; parity vs the pinned fairchem 2.21.0 forward |
| X4 | H | Every compute activity needs export, re-export (27 GB artifacts) and re-validation. | Disk and job budget up front |
| X5 | M | Nitro's speedups are NVIDIA-tuned; PVC FP64 behaviour differs. | Per-block microbenchmark go/no-go |
| X6 | M | Licence: adapting MIT/BSD code into the GPL-2 tree needs notices; CUTLASS DSL is NVIDIA-licensed. | Clean-room re-implementation |

### N1
| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| N1-1 | S | Edge set differs from the current builder (cutoff inclusivity, 300-neighbor truncation, self-edges, tiny/triclinic cells, atoms outside the box). | Edge-set equality tests across crafted cases |
| N1-2 | S | Non-deterministic or different edge order → irreproducible forces or a different summation order than BASE; breaks N2. | Reproduce the reference order (center, distance, neighbor, ox, oy, oz) |
| N1-3 | H | Overflow cannot be remeasured (baked cap). | Read the flag every step; fail fast |
| N1-4 | M | Ranks could build different edge sets. | Cross-rank checksum on debug runs |
| N1-5 | M | Boundary flips at r ≈ 6.0 (negligible energy effect, not bit-identical). | Tolerance band in the equality test |
| N1-6 | M | Host-build premise unverified. | The N1 control measures it |

### N2
| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| N2-1 | M | (Corrected) Pad edges are prepended with the first owned node as center, so the padded shard stays sorted; but this holds only while the graph is center-sorted (true for the CPU builder, **not guaranteed** for a vesin/device builder) and chunk boundaries can still split a segment. | Assert sortedness at runtime in debug; test multi-chunk |
| N2-2 | S | Global cumsum-difference loses FP64 precision; custom kernel needs a custom backward. | Per-segment reduction; FP64 `gradcheck` |
| N2-3 | H | May not be hot at all. | Microbenchmark gate |

### N3
| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| N3-1 | H | Tighter cap turns a density/NPT excursion into a mid-run crash for ≤ ~11% gain. | Opt-in only; likely drop |
| N3-2 | M | Cap must cover the max over ranks and is coupled to chunk size. | Existing cross-rank agreement; one export per cap |

### N4
| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| N4-1 | S | Silent wrong energy if the folded artifact is used with a different composition/charge/spin, or composition changes mid-run. | Metadata key; refuse mismatch |
| N4-2 | M | Coefficients are a runtime input in some modules and constant in one; interface changes. | Per-module audit |
| N4-3 | M | ~1% gain will not clear a threshold alone. | Judge in a stack, or drop |

### N5
| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| N5-1 | H | SYCL/DPC++ build integration, AOT PVC target, ABI match with the torch 2.13+xpu in use. | X2 prototype |
| N5-2 | S | Hand-written backward errors, including virial through edge vectors. | FP64 `gradcheck`, FD spot checks, NPT parity, W ladder |
| N5-3 | H | If FP64 GEMMs dominate compute, fusion gains little. | Block microbenchmarks decide |
| N5-4 | M | Exporter drift (fused ops replace subgraphs in a hand-reimplemented, version-pinned forward). | Per-block parity vs stock eager FP64 |
| N5-5 | S | F7 can deadlock: a traced module cannot fork queues, and multi-queue execution can reorder oneCCL collectives across ranks. | Deferred; if attempted, one ordered queue for collectives |

---

## 3. Second review round — blocking items and robustness (checked against the code)

### 3.0 Status update (2026-11-10, after your answers)

Decisions: goal = step time on the NaCl 14³ W=12 gate case; SYCL kernel dependency acceptable; N3 and N4 kept as low-priority options; harness committed on `uma-nitro` before any measurement.

| Item | Status |
|---|---|
| B1 architecture | **Resolved** from `hen/uma-cache/uma-s-1p2.pt` pickle: `lmax=2`, `mmax=2`, `sphere_channels=128`, `hidden_channels=128`, `edge_channels=128`, `num_layers=4`, `cutoff=6`, `max_neighbors=300`, `num_experts=64`. The `[Ec,25,25]` comments are stale. Nitro's edge-side shapes match. |
| B4 host build | **Resolved** from the Wave E build log: `uma-engine: vesin not found; using CPU neighbor list`. The 118 ms is the single-threaded host builder on all 12 ranks. |
| B5 VCS | **Decided**: commit harness on `uma-nitro`; run outputs/binaries excluded via `.git/info/exclude`; clean check = tracked files only. |
| B2 SYCL op | Open, but now permitted. Prior art found (PyTorch `SyclExtension`) and the engine already compiles `icpx` objects that use torch's XPU queue (`xccl_peer.cpp`). Needs the prototype in §4 before N5. |
| B3 device-build speed | Open. Decided by the N1 go/no-go. |

### 3.1 Blocking items (resolve before any implementation)

| # | Blocking item | Evidence | Required resolution |
|---|---|---|---|
| B1 | **Architecture match is inferred, and the code contradicts itself.** The traced chunk shows 9×9 Wigner and `[3,4,2]` m-splits (lmax=2), but `export_blocks_xpu.py` docstrings and comments repeatedly say `[Ec,25,25]` wigner (lmax=4). Every N5 kernel shape depends on this. | `export_blocks_xpu.py` lines 20, 145, 336, 385 vs traced `torch.zeros([1,9,9])` | Read `lmax`, `mmax`, `sphere_channels`, `num_layers` from the actual checkpoint config (`backbone.lmax`) on a compute node and write them into this plan. Do not start N5 before this |
| B2 | **No SYCL/C++ custom compute op exists in the engine yet.** Only SYCL use is `xccl_peer.cpp` compiled by `icpx` into an object; the registered ops (`uma_ckpt`, `uma_halo`, `uma_peer`) are checkpoint/communication ops. A compute kernel needs an `icpx` kernel object, a `TORCH_LIBRARY` registration with Autograd, an exporter-side stand-in, and a validated trace. | `CMakeLists.txt` lines 97–156; `TORCH_LIBRARY` greps | Run the X2 prototype (one trivial elementwise SYCL op through export → traced load → autograd → W=12) and record whether it works. N5 is blocked until it does |
| B3 | **Device-build speed is not a given.** The code comment in `mpi_peer_predictor.cpp` reports a vesin rebuild of ~1534 ms/step at N=32 (CUDA vesin), i.e. a device build can be slower than the 118 ms host build. | `mpi_peer_predictor.cpp` ~lines 375–381 | Keep the N1 go/no-go (≤ ~25% of host time) as a hard gate; do not assume the Nitro-style builder wins |
| B4 | **Host build premise still unconfirmed in the actual build.** The vesin branch needs `third_party/vesin/lib/libvesin_torch.so`; it is empty here, so the CPU `build_neighbor_graph` branch should be the one compiled, but the actual `build-uma` CMake output was not located. The host builder is single-threaded (no OpenMP) and runs on all 12 ranks. | `CMakeLists.txt` 196–211; `neighbor_list.cpp` (no OpenMP) | Confirm from the build's CMake cache/log, or from the first N1 control run that times the host build |
| B5 | **Resolved by the fresh clone (R.4).** The old tree had untracked `PR_CLEANUP_PLAN.md` and `pbs/`; the new clone has none, so the clean-tree check is meaningful. | — | Commit the harness on `uma-nitro`; exclude `pbs/out/`, `pbs/bin/`, `build-*/` |
| B6 | **Gate rule status.** The PR-cleanup wave rule applies to `lammps-uma-release/`, which this plan no longer edits. Every `nitro/*` branch still has to pass the NaCl 14³ FP64 W=12 NVT+NPT gate before it counts as closed. | `PR_CLEANUP_PLAN.md` lines 24–26 | Never merge `nitro/*` into `lmp-uma-release` unreviewed |

### 3.2 Corrections made to this plan in round 2

1. **N1 ordering requirement changed.** The earlier "sort by (center, neighbor, image)" was wrong. The reference order is by center, then (distance, neighbor, ox, oy, oz); using another order changes the summation order and the 300-neighbor truncation.
2. **N2 no longer depends on N1.** The host graph is already sorted by center, shards preserve it, and pad edges are prepended with the first owned node as center, so the padded shard is still sorted. The old claim "pad edges appended at the end break sortedness" was incorrect.
3. **N1 overflow handling** now reuses the engine's existing cross-rank overflow all-reduce instead of adding a new mechanism.

### 3.3 Additional robustness points

| # | Sev | Point | Mitigation |
|---|---|---|---|
| R1 | H | **Truncation semantics.** `max_neighbors = 300` keeps the 300 nearest per center with a total-order tie-break. The gate cell (~32 neighbors) never triggers it, so a wrong device top-k would pass the gate. | Dense >300-neighbor test case is mandatory for N1 |
| R2 | M | **Multi-image and non-orthorhombic cells.** The host builder switches between all-pairs, cell-list (orthorhombic with ≥3 bins), and image repeats. The device builder must match all three regimes or refuse them explicitly. | Equality tests per regime; explicit error for unsupported cells |
| R3 | M | **Pad edge validity** relies on `|2·a| > cutoff` (offset 2 along x). Not an issue for 5.64 Å NaCl, but an N1/N3 change must not shrink cells or alter the pad convention silently. | Keep pad convention unchanged |
| R4 | M | **Chunk-count baking.** The trace unrolls `ceil(edge_pad_cap / edge_ac_chunk)` chunks. Any activity that changes the cap (N3) or the per-chunk structure (N5) needs a fresh export per N and W; reused artifacts will crash or mis-sum. | Store cap/chunk in metadata; engine asserts at load |
| R5 | M | **Memory headroom.** Peak is ~37 GiB/tile at the gate; fused blocks and a device graph build add temporaries. | Record peak memory per arm |
| R6 | M | **Per-step host sync.** Existing guards already sync (`.item()` after the overflow all-reduce); N1's flag read must not add a second sync. | Fold into the existing check |


---

## 4. B2 prototype plan — a compute SYCL op in the Python-free engine

### 4.1 Does a "compute SYCL op" exist? Yes (checked online, 2026-11-10)

| Source | What it gives us |
|---|---|
| PyTorch tutorial "Custom SYCL Operators" (docs.pytorch.org/tutorials/advanced/cpp_custom_ops_sycl.html), PyTorch ≥ 2.8 | `.sycl` sources built with `torch.utils.cpp_extension.SyclExtension` (compiled by `icpx`), registered with `TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL(ns, XPU, m)`, called as `torch.ops.ns.op` |
| PyTorch PR #132945 | `SyclExtension` / `.sycl` support; arch list via `TORCH_XPU_ARCH_LIST` (default `pvc,xe-lpg`); AOT per device, otherwise JIT |
| `intel/torch-xpu-ops` | Most ATen XPU operators are SYCL kernels; confirms the approach is the supported route and that no Intel Extension for PyTorch is needed |
| Local check | `fxpu` env has `torch.utils.cpp_extension.SyclExtension`, `c10/xpu` headers and `libc10_xpu.so`; `icpx` 2026.1 is on the login path |

So we do **not** invent a mechanism. What is project-specific:

1. The tutorial's op is FP32 and gets autograd from **Python** (`torch.library.register_autograd`). The engine is Python-free, so autograd must be registered in **C++**. The engine already does this for `uma_ckpt` (`TORCH_LIBRARY_IMPL(uma_ckpt, Autograd, ...)` in `block_context.cpp`).
2. The engine already builds an `icpx` object and links it into `uma_engine` (`xccl_peer.o` via a custom CMake command), and that object already uses torch's XPU queue (`c10::xpu::getCurrentXPUStream().queue()` in `xccl_peer.cpp`). This reduces the toolchain risk of B2 considerably. It is not zero risk: `env.inc` symlinks `libsycl.so.9` as `libsycl.so.8`, which signals a runtime-version seam.
3. Export-time tracing needs a Python stand-in for the op, as `uma_ckpt_ops.py` does for `uma_ckpt`.

### 4.2 Design (one op, mirrors the existing `uma_ckpt` pattern)

| Piece | Implementation |
|---|---|
| Kernel | `src/ML-UMA/uma-engine/src/uma_sycl_ops.sycl.cpp`, templated on `double`, `sycl::nd_range<1>`, queue from `c10::xpu::getCurrentXPUStream()`. Prototype op: fused elementwise `y = silu(x * w + b)` with analytic backward kernel (small, has a non-trivial backward, exercises FP64) |
| Registration | `TORCH_LIBRARY(uma_sycl, m)` with the forward and backward schemas; `TORCH_LIBRARY_IMPL(uma_sycl, XPU, m)` for both kernels; `TORCH_LIBRARY_IMPL(uma_sycl, Autograd, m)` with a C++ `torch::autograd::Function` whose backward calls the registered backward op |
| Build | Same pattern as `xccl_peer.o`: `add_custom_command` running `icpx -fsycl -O2 -std=c++17 -fPIC` plus AOT for PVC (`-fsycl-targets=spir64_gen -Xs "-device pvc"`), object linked into `uma_engine`. Build into a per-arm build dir, never the shared `build-uma/` |
| Export stand-in | `python/uma_sycl_ops.py`: `torch.library.Library("uma_sycl", "DEF")` with the **identical schema** and a reference torch implementation (for tracing only), as `uma_ckpt_ops.py` does |
| Trace | A small module calling `torch.ops.uma_sycl.silu_affine(...)` traced with `torch.jit.trace`; confirm the graph records the op, with no baked sizes |
| Branch | `nitro/n5-proto` from `nitro-base`; no model code touched |

### 4.3 Stages and pass bars

Each stage must pass before the next. Compile/syntax work can run on the login node (no torch compute); anything that touches the XPU runs in a `debug` job (1 node, `walltime=01:00:00`).

| Stage | What | Where | Pass |
|---|---|---|---|
| S0 | Toolchain check: build the PyTorch tutorial `mymuladd` in **FP64** with `SyclExtension` into a scratch dir (`TORCH_XPU_ARCH_LIST=pvc`) and run it | compile on login, run on 1 compute node | Compiles with the `fxpu` torch; result equals `a*b+c` in FP64 (rel. error ≤ 1e-15); no ABI or `libsycl` errors |
| S1 | Engine-style C++ test (no Python): standalone exe built like the engine links the new `.o`; call the op on XPU FP64 tensors, then `torch::autograd::grad` | 1 compute node | Forward equals the torch reference (≤ 1e-15 rel.); gradient equals the analytic formula and a central finite difference (≤ 1e-9); bitwise reproducible across 10 repeats |
| S2 | Export round trip: trace the small module with the Python stand-in, save the `.pt`, load it in a C++ process that links the op, run forward+backward | 1 compute node | Loads without "unknown builtin op"; outputs and gradients equal eager torch (≤ 1e-15 / 1e-9) |
| S3 | Link into `uma_engine` and rebuild `lmp`; run the existing NaCl 14³ W=12 NVT+NPT gate with the op linked but **unused** | `debug` job, ABBA vs BASE | Step-0 PE = −74142.191907631001 eV (|ΔE| ≤ 1e-8), identical to BASE; no regression in step time beyond noise; no new static-init or load errors |
| S4 | 12-tile run: `mpiexec -n 12` test that calls the op repeatedly interleaved with an `uma_peer` all-reduce | 1 compute node | No hang; results identical on all ranks; op and collective ordering unchanged |
| S5 | Cost measurement: launch overhead per call, FP64 bandwidth vs `torch` elementwise on the same tensors at the real chunk shapes | 1 compute node | Report only; this is the N5 go/no-go input (a fused op must beat the unfused torch sequence by ≥ 1.5× per block) |

**Gate to N5:** S0–S4 pass. If S1 or S2 fail for a reason that cannot be fixed within the engine's build, N5 and N2-b are dropped, and the plain-torch options (N1-a, N2-a, N3, N4) are what remains.

### 4.4 Failure points specific to the prototype

| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| P-1 | H | **SYCL runtime seam**: the system `icpx` (2026.1) vs the `libsycl.so.9` shipped in `fxpu` (the engine already symlinks 9 as 8). A `sycl::queue` crossing library boundaries can misbehave. | S1 and S3 explicitly run on a compute node; reuse the exact flags and link line of `xccl_peer.o`; record `icpx --version` and the resolved `libsycl` in `meta.json` |
| P-2 | H | **ABI mismatch** with torch (C++11 ABI flag, include dirs). | Copy the engine's include/flag handling; S1 fails fast if the op cannot be loaded |
| P-3 | H | **Schema mismatch** between the Python stand-in and the C++ registration breaks the traced graph at load. | One schema string in a shared header; S2 loads the traced file in C++ |
| P-4 | M | **JIT vs AOT**: without AOT for PVC, 12 ranks JIT-compile on first call and distort timing. | AOT `-device pvc`; discard first-call timing |
| P-5 | M | **Queue ordering** with oneCCL: the op runs on torch's current XPU stream while collectives use their own queue/stream handling. | S4; same-stream usage as `xccl_peer.cpp` |
| P-6 | M | **Autograd under checkpoint recompute**: chunk modules re-run forward with grad enabled in backward; the op must have the `Autograd` key registered or it silently loses gradients. | S1 gradient test through `CheckpointModuleFn`-style no_grad forward then recompute |
| P-7 | M | **FP64 contraction**: `icpx` may fuse multiply-add differently from the torch kernel, so results are not bit-identical. | Tolerance bars above; do not claim bit-identity for fused ops |
| P-8 | M | **Disk and build isolation**: building into the shared `build-uma/` would overwrite the BASE binary. | Per-arm build dir, binary copied to `pbs/bin/` |
| P-9 | M | **Process rules**: any edit needs the syntax-check subagent (`grok-4.7-high`) first; any `qsub` needs queue `debug`, `walltime=01:00:00`, and a clean tracked tree on the tagged commit. | Follow the execution checklist before each step |

### 4.5 What the prototype deliberately does not do

It does not fuse any real UMA block, change the exported model, or alter `max_neighbors`/edge sets. It is a toolchain and integration test only. Real fusions (F1–F6) start only after the gate in §4.3 passes and the per-block microbenchmark clears the 1.5× bar.
