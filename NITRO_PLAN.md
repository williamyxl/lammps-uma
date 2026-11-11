# Nitro-inspired activities — LibTorch UMA LAMMPS (authoritative plan, revision 4)

Scope: **only activities whose idea comes from UMA Nitro**
(`fairchem/.../uma_nitro/_inference.py`). Nothing else is planned here.
This file is **self-contained**: the shared test protocol (correctness bars, timing, ABBA, effect
threshold, results template) is in section P below. The earlier full plan
(`uma_lammps_perf_test_plan.md` in the workdir root) is reference only.

Status (revision 5, 2026-11-11): harness committed (`7da48ce5`). BASE gate job 9072484 passed the **PE bars**
(step-0 |ΔE| 0 / 1.5e-11, step-10 exact) but its **timing path is dead** (0 `MP_PERF` lines, empty `perf_nvt.json`;
see section H, finding F1). `nitro-base` is **not yet tagged**. No activity work starts before stages H1–H6 are closed.
Gains are estimates and come from the pre-`getenv`-removal tree; they are re-measured in H4/H5.

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
| Working branch | `uma-nitro` (created from `lmp-uma-release`). **Revision 5:** pushed to `origin/uma-nitro` at the end of every stage (H.7); `lmp-uma-release` is never pushed to from this clone. |
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
- Disk: Flare project quota is 27.57T used of 40T (69%). Each W=12 N=14 artifact is about 27 GB. **Artifacts exported by a job are deleted at job exit unless `KEEP_ARTIFACT=1`** (H2.6).
- Before any measurement: stages H1–H6 closed (section H).

### R.6 Execution sequence (revision 5 — supersedes revision 4)

Steps 0–1 are done. **Stages H1–H6 (section H) must all be closed, in order, before step 7.**

| Step | Action | Needs compute node | Status |
|---|---|---|---|
| 0 | Clone and create `uma-nitro` | no | done |
| 1 | Harness in `pbs/nitro/` committed on `uma-nitro` (`7da48ce5`); `nitro-origin` tag | no | done (ABBA runner was **missing**, see H2) |
| H1 | Restore perf/debug instrumentation (nitro tree only) | build only | open |
| H2 | Harness hardening: ABBA runner, replicate-aware compare, `meta.json` completeness, tree check, artifact auto-delete | no | open |
| H3 | Re-verify BASE gate with working `MP_PERF`; tag `nitro-base` | yes (`debug`) | open |
| H4 | E0: noise floor, bitwise repeat, 3 jobs on different nodes; minimum detectable effect | yes (`debug`) | open |
| H5 | Step-time decomposition: is the ~400 ms all-reduce communication or waiting? | yes (`debug`) | open |
| H6 | Refresh plan numbers: Amdahl ledger, N-activity expectations | no | open |
| 7 | S0 of the SYCL prototype (§4), on `nitro/n5-proto` | yes (`debug`) | blocked by H6 |
| 8 | N1 control: time the host graph build against the device build, ABBA | yes (`debug`) | blocked by H6 |
| 9 | Continue with S1–S5, then N2, N5 microbenchmarks, N3, N4, stack (§1, §5) | yes | blocked by H6 |

H1–H6 close "plan ready for execution". The activities then proceed in the order in §1.

---

## H. Hardening stages before any activity (revision 5, 2026-11-11)

Source: robustness review of revision 4 checked against the tree and BASE gate job 9072484.

### H.0 Decisions recorded (user answers, 2026-11-11)

| Topic | Decision |
|---|---|
| Instrumentation fix location | **Nitro tree only** (`uma-nitro`). The PR tree (`lammps-ml-uma`) is not changed. Consequence: `nitro-base` = `a240ba3485` + harness + instrumentation commit, so "BASE equals the PR branch head" is **no longer true** and must be stated as "equals PR head + instrumentation-only commit `<sha>`". |
| Overall target | **≥ 1.5× end-to-end step time** (BASE median / best stack median, NaCl 14³, W=12, FP64). Continue to N5 regardless of earlier gains; N5 blocks are still individually gated by their microbenchmarks (≥ 1.5×). The Amdahl ledger (H6) states whether 1.5× is reachable on paper; if it is not, say so in the final report. |
| Replication | Every go/no-go claim needs **≥ 3 PBS jobs on different nodes**; the replicate unit is the job (per-job medians), not the step. |
| Disk | Artifacts exported by a job are **deleted at job exit by default**; a flag keeps them (H2.6). |
| Host-build control | The shard-local / OpenMP host-build control arm was **declined**. N1 stays "device build vs current host build". |

### H.1 Findings and where each is closed

| ID | Finding | Closed in |
|---|---|---|
| F1 | `getenv` → in-process map (`b6ca79f8`) silently disabled `UMA_MP_PERF`, `UMA_MEM_PEAK`, `UMA_MP_NATOMS`, `UMA_MN_CKPT`: 0 `MP_PERF` lines, empty `perf_nvt.json`; all quoted ms figures are from the old tree | H1, H3 |
| F2 | P.3 item 3 ("gate new behaviour behind env vars") does not work for any `UMA_*` name that is not set by a `pair_style` keyword | H1.4 |
| F3 | `nitro-base` not tagged; `meta.json` has `nitro_base_sha: null` | H3 |
| F4 | No ABBA runner (P.3 and R.6 assumed one); `run_gate.pbs` / `run_timing.pbs` are single-commit | H2.1 |
| F5 | Statistics treat steps as independent; one job, one node, autocorrelated | H2.2, H4 |
| F6 | No artifact disk policy (27 GB each; `/lus/flare` 65% used) | H2.6 |
| F7 | `nitro_check_tree` ignores untracked files; PBS `.o`/`.e` files land in `pbs/nitro/` | H2.4 |
| F8 | MPICH path hard-coded in `run_gate.pbs` | H2.5 |
| F9 | `UMA_ALLOW_FAIRCHEM_MISMATCH=1` is set globally; the fairchem version actually used is not recorded | H2.5 |
| F10 | The ~400 ms all-reduce (0.81 GB over 15 calls ≈ 2 GB/s) is probably waiting, not transfer; it is the largest slice and unmeasured | H5 |
| F11 | Plan numbers (828 / 390 / 118 ms) come from a different tree; Amdahl ceiling not stated | H6 |
| F12 | Stale statements (baseline "not verified", R.4 rows) | H3, H6 |
| F13 | Inherited PR-tree issues (CPU compile untested; `devices > 1` workers cannot see the runtime map) | Out of scope here (see H.8) |

### H1 — Restore instrumentation in the nitro tree (code; `uma-nitro`)

Goal: perf and debug knobs work again, and new arm toggles have a legal, documented mechanism.

1. **Audit.** List every name read via `uma::runtime_get` / `runtime_flag_1` (`git grep`). Classify each as: (a) set by `apply_runtime_config()` from a keyword (`UMA_AC`, `UMA_DD`, `UMA_ENGINE_BUILD_GRAPH`, `UMA_COMPUTE_VIRIAL`); (b) perf/debug (`UMA_MP_PERF`, `UMA_MEM_PEAK`, …); (c) functional (`UMA_MP_NATOMS`, `UMA_MN_CKPT`, `UMA_ALLOW_LEGACY_METADATA`, …). Record the table in `pbs/nitro/ENV_AUDIT.md`.
2. **Fix.** In `runtime_config.cpp`, `runtime_get(name)` returns the map value; if absent and `name` is in a compiled-in **allowlist**, fall back to `std::getenv(name)`. The allowlist holds the (b) and (c) names that the harness exports plus every arm toggle added later (`UMA_NITRO_*`). Names in (a) never fall back (keywords win). This is nitro-only and is not to be ported to the PR tree.
3. **Mark the commit** `nitro: instrumentation-only env fallback` and keep it free of any numerics change; it becomes part of `nitro-base`.
4. **Arm-toggle rule (replaces P.3 item 3 wording).** A new behaviour is added as `UMA_NITRO_<ID>` in the allowlist, default = BASE (unset ⇒ BASE path), read once at construction and logged on rank 0 (`NITRO_FLAGS: ...`) so a log proves which arm ran.
5. **Syntax-check subagent** (`grok-4.7-high`, `shell`) after the edit and before any `qsub`.

Exit: build succeeds; `git grep` shows each allowlisted name documented in `ENV_AUDIT.md`; no change outside `runtime_config.cpp`, its header if needed, and `ENV_AUDIT.md`. Compute proof is H3.

### H2 — Harness hardening (no compute)

1. **ABBA runner** `pbs/nitro/run_abba.pbs` + `abba.inc`. Input: a spec file of arms, each = (label, tag, optional env toggles, artifact path). It builds each distinct tag (or reuses `pbs/bin/<sha7>/lmp` if its sha256 matches), runs the order A B B A (BASE first), 30-step NVT per slot, and keeps going after a failing arm (risky arms last). Output per slot: log, `MP_PERF` JSON, hostnames, `meta.json`.
2. **Replicate-aware compare.** `compare.py abba` takes the outputs of ≥ 3 jobs and reports per-job medians per arm, the paired per-job difference, the between-job spread, and a verdict under the **revised effect threshold**: an arm wins only if (i) all replicate jobs show the same sign of gain, **and** (ii) the median per-job gain exceeds `max(3 × pooled within-job MAD, between-job range)`. Per-step intervals remain descriptive only. P.2 is updated accordingly (H6).
3. **`meta.json` completeness.** Fill `nitro_base_sha` (from the `nitro-base` tag; fail if absent in timing jobs), `nitro_tag`, and record versions: `torch`, `fairchem` (actual vs expected 2.21.0), `icpx --version`, resolved `libsycl`, MPICH path, `ONECCL_ROOT`, plus `NITRO_FLAGS` from the log.
4. **Tree check and stray files.** `nitro_check_tree` must also fail on **untracked, non-ignored** files under `src/` and `pbs/nitro/`. Add `pbs/nitro/*.o*`, `pbs/nitro/*.e*` to `.git/info/exclude` (or set `#PBS -o`/`-e` into `pbs/out/`). Remove the existing `nitro-gate.o9072484` from the work tree after copying it into its `pbs/out/` run directory.
5. **Environment portability.** Resolve the MPICH path from a variable in `env.inc` with a recorded fallback instead of a literal in `run_gate.pbs`; fail with a clear message if missing. Keep `UMA_ALLOW_FAIRCHEM_MISMATCH=1` but log the actual fairchem version and a warning line in `SUMMARY.txt`, so a mismatch is visible in every result.
6. **Artifact policy (feature).** Every job that exports an artifact **deletes it at exit** by default (`trap ... EXIT`; also on failure and on `qdel`/walltime where the shell still runs). Flag: `KEEP_ARTIFACT=1` (set with `qsub -v`) keeps it. Rules:
   - A job deletes only artifacts **it created** (marker file written at export start); artifacts reused through `NITRO_REF` are read-only inputs and are never removed.
   - `meta.json` records `artifact_kept: true|false` and the artifact size and sha-of-manifest before deletion.
   - The existing BASE artifact from job 9072484 (about 27 GB) is kept as the BASE reference through H5; delete it when H5 closes unless later arms still need it.
   - Microbenchmark and single-rank export jobs export one rank only.
7. **Syntax-check subagent** on all edited `.sh/.pbs/.inc/.py` (`python3.10 -m py_compile`, `bash -n`) before commit and before any `qsub`.

Exit: syntax check COMPILE_OK; dry-run of `run_abba.pbs` logic with a fake spec on the login node (no torch compute) produces the expected slot order and metadata skeleton; `compare.py abba` unit-tested on synthetic per-job data (sign-disagreement ⇒ no-win; spread larger than gain ⇒ no-win).

### H3 — Re-verify BASE and freeze it (`debug`, 1 node)

1. Commit H1 and H2 on `uma-nitro`; tag `nitro/base-verify/r2`.
2. Run `run_gate.pbs` on that tag (clean tracked tree; `KEEP_ARTIFACT` unset; reuse the 9072484 artifact via `NITRO_REF`).
3. Pass bars: all P.1 PE bars unchanged (step-0 |ΔE| ≤ 1e-8, step-10 NVT/NPT PEs equal the recorded values to ≤ 1e-8); `MP_PERF rank=0` rows present for every step; `perf_nvt.json` non-empty; `NITRO_FLAGS` line present; `meta.json` has no null fields and records `nitro_base_sha`.
4. If the PE differs from the earlier gate beyond 1e-11, the instrumentation commit changed numerics: revert it and redo H1.
5. On pass: create annotated tag **`nitro-base`** on that commit; update R.1 ("Base commit" row) and R.3 to say "PR head `a240ba3485` + instrumentation commit `<sha>`".

Exit: `nitro-base` exists and is the commit that H4/H5 and all arms are built from.

### H4 — E0: noise floor and replication (`debug`, 3 jobs on different nodes)

1. Use `run_timing.pbs` (REPS = 3 per job) in **3 separate jobs**, submitted so the node sets differ (check `hostnames` in `meta.json`; resubmit a job that lands on a repeated node set or a throttled node).
2. Measure: (a) bitwise repeat of PE and forces within a job and across jobs (`compare.py bit`, `diff_forces.py`); (b) BASE-vs-BASE max|ΔF| and max|ΔW| (these set the force/virial bars in P.1); (c) per-job median step time and `MP_PERF` fields; within-job MAD and between-job range; (d) one extra pair with `UMA_MP_PERF` off vs on (instrumentation effect).
3. Output `pbs/out/nitro_e0/E0_SUMMARY.md` with the **minimum detectable effect** (the larger of 3 × within-job MAD and the between-job range, in ms and %), and the calibrated force/virial tolerances. Copy the values into P.1/P.2.
4. If BASE is not bitwise reproducible, P.1 "Bit-identity" is relaxed to the calibrated tolerance for all arms and this is written down.
5. If the minimum detectable effect exceeds 5% of the step time, add a rule to P.2: arms with an expected gain below the MDE (N3, N4) are judged only inside a stack, with ≥ 5 jobs.

Exit: E0_SUMMARY.md written; tolerances and MDE entered in the plan.

### H5 — Step-time decomposition (`debug`, 1 node; diagnostic only)

Question: is the ~400 ms in-graph all-reduce real communication or waiting for other ranks (compute imbalance, stream sync, `MP_PERF` event syncs)? It is the largest slice (≈ 48%) and no Nitro activity touches it.

1. Add an opt-in diagnostic knob `UMA_NITRO_PERF_BARRIER=1` (allowlisted, default off, perf-only): a timed `MPI_Barrier` / device sync immediately before each collective, so `ms_wait` and `ms_xfer` are reported separately. This is a small code change on branch `nitro/h5-diag` cut from `nitro-base` (not part of `nitro-base`); syntax-check first.
2. Run in one job, ABBA: BASE with `UMA_MP_PERF` on, BASE with it off (step time only), BASE with the barrier diagnostic. Three jobs on different nodes (H4 rule), same scripts.
3. Report: `ms_wait` vs `ms_xfer` per collective; effective GB/s per rank against the expected link/XCCL rate; per-rank compute time spread (is one rank always last?).
4. Decision rules (recorded, not acted on here): if waiting dominates, the cause is imbalance or sync, and the finding is added to X1 with the per-rank evidence; if transfer dominates at ≈ 2 GB/s, the finding is added to X1 as "transport inefficiency, outside Nitro scope". Either way the result feeds H6; it does not change the user's decision to continue to N5.

Exit: `pbs/out/nitro_h5/H5_SUMMARY.md` with the wait/transfer split and per-rank spread.

### H6 — Refresh the plan (no compute)

1. **Amdahl ledger.** Replace the 828 / 390 / 118 ms narrative with a table from H3–H5 of the BASE step: host graph build, compute (fwd + bwd without collectives), collective wait, collective transfer, other; each with median, MAD and share. For each activity, add its upper bound as a function of the ledger (e.g. N1 ≤ host-build share; N5 ≤ compute share) and the cumulative ceiling versus the 1.5× target.
2. **Update expectations** in N1–N5 ("Expected gain" lines) and X1 with the measured ledger; mark N3 as "small by design (real edges ≈ 58–60k vs cap 65 536)".
3. **Update P.1–P.4**: calibrated tolerances and MDE (H4), replicate rule and revised effect threshold (H2.2), arm-toggle rule (H1.4), ABBA runner row and `run_abba.pbs`, `abba.inc`, `ENV_AUDIT.md` rows in the harness file table.
4. **Fix stale text**: "baseline not yet verified", R.4 rows ("`pbs/` harness missing", the artifact row), R.5 disk line (artifacts auto-deleted; project at the measured usage), P.3 item 3.
5. **Close-out table** at the end of this section: each F-item with its closing stage and result.

Exit: no "TBD" or stale statement left in R, P, 1; H-table all closed; then step 7 may start.

### H.7 Rules that apply to every H stage

- **Sequential close (rule, 2026-11-11).** Never open an item in a later stage while any item in an earlier stage is not fully closed. A stage is fully closed only when **every** item in it is done: its exit criteria met, its plan status updated, and its commit pushed (or the push explicitly waived for that stage per the rule below). This applies to items within a stage as well as to whole stages: no H2 work starts while any H1 item is open, and so on through H6, then step 7 onward. A blocked item stays in its own stage; do not skip ahead and come back.
- Compute only in `debug` / `debug-scaling`, `walltime=01:00:00`, FP64, no login-node compute, tagged immutable commit before every `qsub`, syntax-check subagent before every `qsub`.
- A failed gate means: revert that stage's commit, do not tag, do not advance.
- **Commit and push at the end of every stage (rule, 2026-11-11).** A stage is closed only after all of:
  1. Its exit criteria are met and the plan's stage-status table (R.6) and any numbers it produced are updated **in the same commit**.
  2. Syntax-check subagent passed for any code touched (H.7 rule above).
  3. `git commit` on the branch the stage's work lives on (`uma-nitro` for H1–H4 and H6; `nitro/h5-diag` for the H5 diagnostic code, with the H5 summary and plan update committed on `uma-nitro`). Message: `nitro: stage H<n> closed — <one line>`. Include small result files (`E0_SUMMARY.md`, `H5_SUMMARY.md`, `ENV_AUDIT.md`); never `pbs/out/` data, artifacts, binaries or `build-*/`.
  4. `git push origin <same branch>` (remote `origin` = `github.com/williamyxl/lammps-uma`), plus `git push origin <tag>` for any tag created in the stage (`nitro-base`, `nitro/<id>/r<k>`). Tags are immutable: never move or re-push a tag.
  5. Never force-push; never push to `lmp-uma-release` or any branch that is not the stage's own branch; never push before the syntax check.
  - If the push fails (this login node has had no GitHub credentials), the agent records `push pending` in the R.6 status column, gives the user the exact `git push` commands, and the **next stage does not start until the user confirms the push** (or explicitly waives it for that stage).

### H.8 Deliberately out of scope here

- Inherited PR-tree issues (F13): CPU-only compile test and `devices > 1` worker environment propagation. They do not affect the W=12 `devices 1` gate; they are tracked in `lammps-ml-uma` and not fixed in the nitro tree.
- The shard-local / OpenMP host-build control arm (declined in H.0).
- Any change to the stock UMA model or to the PR tree.

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
- **Replication (revision 5, H.0/H2.2):** the replicate unit is the PBS job; every claim needs ≥ 3 jobs on different nodes. Verdict: the gain has the same sign in all jobs **and** the median per-job gain exceeds `max(3 × pooled within-job MAD, between-job range)`. The per-step interval below is descriptive only.
- **Effect threshold (per-step, descriptive):** the paired step-time difference has a 95% interval excluding 0 **and** the median gain exceeds 3 × the pooled MAD; treat 5% as a priority signal, not a pass/fail line, so small deterministic savings (N3, N4) can still be judged in a stack.
- Speedup = `median(BASE) / median(arm)`, labelled with cell, W and queue.

### P.3 Control of variables

1. One PBS job holds several arms in **ABBA** order (BASE, A, A, BASE).
2. One factor changes per comparison; binary, artifact, env, deck, seed, run length, rank-to-tile mapping stay frozen.
3. Gate new behaviour behind `UMA_NITRO_<ID>` toggles (allowlisted env fallback, H1.4) or `pair_style` keywords with default = BASE, so one binary serves both arms where possible. Plain `UMA_*` env vars are **ignored** by the engine unless allowlisted.
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
| X1 | H | Upside is capped: Nitro has nothing for the in-graph all-reduce wall (~390 of 828 ms, old tree). Even zero compute gives only ≈ 1.57×; target is ≥ 1.5× (H.0), which is near that ceiling. | H5 splits wait from transfer; H6 writes the Amdahl ledger before work starts |
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
