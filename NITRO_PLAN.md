# Nitro-inspired activities — LibTorch UMA LAMMPS (authoritative plan, revision 10)

Scope: **only activities whose idea comes from UMA Nitro**
(`fairchem/.../uma_nitro/_inference.py`). Nothing else is planned here.
This file is **self-contained**: the shared test protocol (correctness bars, timing, ABBA, effect
threshold, results template) is in section P below. The earlier full plan
(`uma_lammps_perf_test_plan.md` in the workdir root) is reference only.

Status (revision 10, 2026-11-12): **steps 1–10 are closed** (through Z1, the ZIF-8 test system). **Step 11 (N1) is next.** The one numbered order is the master table in R.6; all other labels are names, not order. The dedicated N5 step plan is §5. **There is no speedup target**: the objective is to try every option and report the best correct configuration (see H.0, "Objective and decision policy"). `nitro-base` is `efae85ff72` (job 9077571). The BASE step is 825 ms. The ledger is in H6.

Fixed conditions: FP64 only; stock UMA model unchanged; gate systems = NaCl 14³ (21,952 atoms) and ZIF-8 3×3×3 (7,452 atoms; Z1), W=12,
NVT + NPT 10 steps, identical protocol for both; one branch per activity, immutable tag before each submit.

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
| Base commit | Verified BASE is tag `nitro-base` = `efae85ff72`. That is PR head `a240ba3485` plus the harness, the H1 instrumentation commit `87adda2a47`, H2, and the environ link fix. It is not identical to the PR head. |
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
| `nitro-base` (annotated tag) | `efae85ff72`, the verified BASE arm (job 9077571). Never moves. |
| `nitro/<id>` (branches) | One per activity, cut from `nitro-base`: `nitro/n1-a`, `nitro/n1-b`, `nitro/n2-a`, `nitro/n2-b`, `nitro/n3-cap`, `nitro/n4-a`, `nitro/n4-b`, `nitro/n5-f1` … `nitro/n5-f7`, `nitro/n5-proto` |
| `nitro/<id>/r<k>` (annotated tags) | Immutable measurement tag, created before every `qsub`. Never amend a tagged commit. |
| `nitro/stack/s<k>` | Merge-commit stacks of passing options (no squash, no rebase). |

The rest of the protocol (flag-gated arms with BASE as default, per-arm build dir, `meta.json`, failed arms stay, no cherry-pick of result commits) is unchanged from the earlier plan's §0.1.

### R.4 Effects of the fresh clone on earlier blockers

| Earlier item | Effect |
|---|---|
| B5 (untracked `pbs/` and `PR_CLEANUP_PLAN.md`) | **Resolved.** The clone has no untracked files, so a clean-tree check is meaningful. Rule: commit harness files on `uma-nitro`; keep `pbs/out/`, `pbs/bin/`, `build-*/` out of git via `.git/info/exclude`. |
| Missing build script | **Resolved.** The clone has no `scripts/build_lammps_uma.sh` either. The working recipe is the `cmake` call in `lammps-uma-release/pbs/pr_gate/run_waveE.pbs` (GCC 13.4, MPICH 5.0, torch cmake path from `fxpu`, `UMA_ENGINE_USE_XPU=ON`, `UMA_ENGINE_USE_XCCL=ON`, `UMA_ICPX`, `ONECCL_ROOT=$CONDA_PREFIX`, `PKG_ML-UMA=ON`, `PKG_KOKKOS=OFF`). Copy it into the clone as `pbs/nitro/build.inc`, with `LMP_BUILD` set per arm (`build-nitro/<sha7>/`). Wave E built inside the debug job; keep that. |
| `pbs/` harness | **Committed and used.** Gate 9077571, E0 jobs 9078185/9078330/9078397, and H5 jobs 9079409/9079449/9079543 ran from `pbs/nitro/`. |
| Existing W=12 N=14 artifact | The BASE artifact is `lammps-uma-nitro/pbs/out/nitro_gate_9072484.../n14/artifact`, reused by the H3 gate and by H4/H5. Any arm that changes export re-exports into the nitro tree and deletes that export at job exit unless `KEEP_ARTIFACT=1`. |

### R.5 Standing rules for every job in this tree

- Queues: `debug` or `debug-scaling` only; `walltime=01:00:00`.
- FP64 only for all model numerics and custom kernels.
- No torch/XPU compute on login nodes. Compiling is fine.
- Syntax-check subagent (`grok-4.7-high`, type `shell`) after every code edit and before every `qsub`.
- Disk: each W=12 N=14 artifact is about 27 GB. **Artifacts exported by a job are deleted at job exit unless `KEEP_ARTIFACT=1`** (H2.6). On 2026-11-12, `lfs quota` reports user xiaoliyan using 7.9T on `/lus/flare/projects/MatSciAI`. The quota fields came back unlimited, so the older 27.57T-of-40T project figure was not remeasured.
- Queues for H3 onward: submit on `debug-scaling`. If the queue refuses the job, poll every 10 minutes and submit there again. Do not use any other queue. While a job is in the queue or running, poll its status every 10 minutes.
- Steps 1–10 are closed. Step 11 (N1) is next.

### R.6 Master sequence (revision 10) — the only numbered order in this plan

**Read this table first.** The **Step** column is the one and only order of work. Every other label (H1, Z1, S0, N1, N5.4, F2, E0) is just the *name* of what a step does. A name never says when something happens. Steps run strictly in number order, and a step opens only when every lower step is closed (or its wait is explicitly waived by the user, which is recorded in the status cell).

| Step | Name in this plan | What it is | Compute | Status |
|---|---|---|---|---|
| 1 | clone | Clone and create `uma-nitro` | no | closed |
| 2 | harness | Harness in `pbs/nitro/` (`7da48ce5`), `nitro-origin` tag | no | closed (ABBA runner was missing, built in step 4) |
| 3 | **H1** | Restore perf/debug instrumentation (allowlist in `ENV_AUDIT.md`) | build only | closed |
| 4 | **H2** | Harness hardening: ABBA runner, replicate-aware compare, `meta.json`, tree check, artifact auto-delete | no | closed |
| 5 | **H3** | Re-verify the BASE gate with working `MP_PERF`; tag `nitro-base` | yes | closed. Job 9077571, `nitro-base` = `efae85ff72` |
| 6 | **H4** (E0) | Noise floor, bitwise repeat, 3 nodes; minimum detectable effect | yes | closed. MDE 6.27 ms (0.76% of 823 ms) |
| 7 | **H5** | Step-time decomposition of the ~400 ms all-reduce | yes | closed. ~360 ms local drain, ~20 ms skew, ~12 ms transfer |
| 8 | **H6** | Refresh plan numbers: Amdahl ledger | no | closed. BASE step 825 ms; compute 659 ms (80%); N1 at most 118 ms |
| 9 | **S0** | SYCL toolchain check: FP64 `mymuladd` (§4) | yes | closed. Job 9083937, tag `nitro/s0/r2`; rel. error 1.537e-16 |
| 10 | **Z1** | ZIF-8 3×3×3 as a second test system: BASE reference PEs, verify gate, noise floor, ledger | yes | closed. Record job 9084279, verify job 9084359 (all four \|ΔPE\| ≤ 3.6e-10 eV), 3 timing jobs; MDE 3.11 ms (0.35% of 876 ms); tags `nitro/z1/r1`, `nitro/z1/r2` |
| 11 | **N1-a** | Device-side graph build with torch ops: selftest, W=1 energy check, then the W ladder and both-system gates | yes | in progress. DIAG NaCl 4³ passed at W=1 (9087586), W=2 (9087641), W=4 (9087698), and W=6 (9087772). Next rung is W=12. Not a gate result. |
| 12 | **S1** | Engine-style C++ test of a custom op (§4.3) | yes | not started |
| 13 | **S2** | Export round trip of the custom op | yes | not started |
| 14 | **S3** | Link the op into `uma_engine`, run the gate with it unused | yes | not started |
| 15 | **S4** | 12-tile test beside the all-reduce | yes | not started |
| 16 | **N1-b** | Same graph builder as step 11, as custom SYCL kernels. Same name as before; the user placed it here on 2026-11-12, after S4 and before S5 | yes | not started |
| 17 | **S5** | Cost measurement (report only) | yes | not started |
| 18 | **N2** | Target-sorted edge→node reduction (N2-a torch ops, then N2-b SYCL) | yes | not started |
| 19 | **N5.0** | N5 entry check | no | not started |
| 20 | **N5.1** | Op-level profile of the 659 ms compute row | yes | not started |
| 21 | **N5.2** | Fixtures and reference harness | yes | not started |
| 22 | **N5.3** | Shared infrastructure for fused ops (exporter flags, op header, load check) | build + 1 node | not started |
| 23 | **N5.4** (F2) | Fused gate | yes | not started |
| 24 | **N5.5** (F4) | Fused Wigner rotation + conv1 | yes | not started |
| 25 | **N5.6** (F5) | Fused inverse rotation + conv2 + reduce | yes | not started |
| 26 | **N5.7** (F1) | Fused LayerNorm + SiLU | yes | not started |
| 27 | **N5.8** (F3) | Fused radial MLP chain | yes | not started |
| 28 | **N5.9** | Stack gate for the N5 blocks | yes | not started |
| 29 | **N3** | Measured edge capacity (opt-in) | yes | not started |
| 30 | **N4** | MOLE merge at export | yes | not started |
| 31 | **final stack** | All wins together, leave-one-out and add-one-in, vs BASE, both systems | yes | not started |
| 32 | **F7** | Side-queue concurrency (deferred; opens only if the user asks) | n/a | deferred |

Name legend: **H** = hardening, **Z** = ZIF-8 test system, **S** = SYCL prototype stage, **N** = optimization activity (N5.x = a stage inside N5), **F** = a fused block built inside an N5 step. Sub-items such as Z1.3 or N1-a belong to the step that names them.

**Labeling rule (user, 2026-11-12).** No new label letters or prefixes are added to this plan. New work takes the next free plain step number, or becomes a sub-item of an existing step under its existing label. If a new letter seems necessary, ask the user first. Every status message names work as "step N (name)".


---

## H. Hardening stages before any activity (revision 5, 2026-11-11)

Source: robustness review of revision 4 checked against the tree and BASE gate job 9072484.

### H.0 Decisions recorded (user answers, 2026-11-11)

| Topic | Decision |
|---|---|
| Instrumentation fix location | **Nitro tree only** (`uma-nitro`). The PR tree (`lammps-ml-uma`) is not changed. Consequence: `nitro-base` = `a240ba3485` + harness + instrumentation commit, so "BASE equals the PR branch head" is **no longer true** and must be stated as "equals PR head + instrumentation-only commit `<sha>`". |
| Objective and decision policy (revision 8, replaces the earlier ≥ 1.5× target) | **No speedup target and no speed threshold anywhere in this plan.** The objective is to exhaust the options and report the best correct configuration: BASE median / best stack median (NaCl 14³, W=12, FP64) with its run-to-run range, as a measurement and not a pass/fail. Rules: (1) **Correctness is the only hard gate** (P.1 bars). (2) **Every planned activity and block is attempted**, whatever its expected size; ordering by measured share is for priority only, never for skipping. (3) **Each result gets a verdict by the replication rule**: win (beyond the MDE and the between-job range), flat, or loss. A win goes into the stack. Flat and loss are recorded with their numbers and stay out. (4) **Every custom kernel gets a tuning round** (work-group and sub-group shape, local-memory use, vendor library versus custom) before it is closed flat or loss. (5) A stage can also close as **infeasible** (cannot be built or cannot be made correct) with the blocker documented. (6) **There is no stop rule by speed.** The program ends when every item has a verdict and the final stack (with leave-one-out and add-one-in checks) is measured. The Amdahl ledger (H6) is information, not a goal. |
| Test systems (revision 9) | Two systems, **identical protocol** (W=12, FP64, NVT and NPT, 300 K, velocity seed 4928459, timestep 0.001 ps, same fixes, same bars); only the structure differs. **NaCl** rocksalt 14³ (21,952 atoms, `make_nacl.py`). **ZIF-8** 3×3×3 (7,452 atoms: 3,240 H, 2,592 C, 1,296 N, 324 Zn; 50.97 Å cube; `make_zif8.py` from the rosen_uma_ZIF 4×4×4 CIF, same 0.05 Å rattle with seed 0 as NaCl). Selected per job with `NITRO_SYSTEM=nacl|zif8` (`pbs/nitro/systems.inc`). Defaults, change if you disagree: (1) **correctness** gates run on both systems and both must pass; (2) **timing verdicts are per system**, reported side by side, never pooled (each has its own ledger and MDE); (3) a change enters the stack if it is correct on both and is a win on at least one and a loss on neither; (4) per-activity ABBA and the stack gates run on both systems, micro stages use NaCl shapes with ZIF-8 shapes reported alongside. |
| Replication | Every verdict (win, flat, loss) needs **≥ 3 PBS jobs on different nodes**; the replicate unit is the job (per-job medians), not the step. |
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

### Step 3 · H1 — Restore instrumentation in the nitro tree (code; `uma-nitro`)

Goal: perf and debug knobs work again, and new arm toggles have a legal, documented mechanism.

1. **Audit.** List every name read via `uma::runtime_get` / `runtime_flag_1` (`git grep`). Classify each as: (a) set by `apply_runtime_config()` from a keyword (`UMA_AC`, `UMA_DD`, `UMA_ENGINE_BUILD_GRAPH`, `UMA_COMPUTE_VIRIAL`); (b) perf/debug (`UMA_MP_PERF`, `UMA_MEM_PEAK`, …); (c) functional (`UMA_MP_NATOMS`, `UMA_MN_CKPT`, `UMA_ALLOW_LEGACY_METADATA`, …). Record the table in `pbs/nitro/ENV_AUDIT.md`.
2. **Fix.** In `runtime_config.cpp`, `runtime_get(name)` returns the map value; if absent and `name` is in a compiled-in **allowlist**, fall back to `std::getenv(name)`. The allowlist holds the (b) and (c) names that the harness exports plus every arm toggle added later (`UMA_NITRO_*`). Names in (a) never fall back (keywords win). This is nitro-only and is not to be ported to the PR tree.
3. **Mark the commit** `nitro: instrumentation-only env fallback` and keep it free of any numerics change; it becomes part of `nitro-base`.
4. **Arm-toggle rule (replaces P.3 item 3 wording).** A new behaviour is added as `UMA_NITRO_<ID>` in the allowlist, default = BASE (unset ⇒ BASE path), read once at construction and logged on rank 0 (`NITRO_FLAGS: ...`) so a log proves which arm ran.
5. **Syntax-check subagent** (`grok-4.7-high`, `shell`) after the edit and before any `qsub`.

Exit: build succeeds; `git grep` shows each allowlisted name documented in `ENV_AUDIT.md`; code changes are `runtime_config.cpp`, its header, and `ENV_AUDIT.md` (this plan's R.6 row is updated in the same commit by the H.7 close rule). Compute proof is H3.

Status: **closed** in the H1 commit. `MP_PERF` rows are still unproven until H3.

### Step 4 · H2 — Harness hardening (no compute)

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

Status: **closed** in the H2 commit. Files: `abba.inc`, `run_abba.pbs`, `compare.py abba` / `abba-test`, `meta.py` (`nitro_tag`, versions, `artifact_kept`), tree check, MPICH from `env.inc`, exit-trap deletion. The 9072484 artifact is not marked owned and stays until H5.

### Step 5 · H3 — Re-verify BASE and freeze it (`debug`, 1 node)

1. Commit H1 and H2 on `uma-nitro`; tag `nitro/base-verify/r2`.
2. Run `run_gate.pbs` on that tag (clean tracked tree; `KEEP_ARTIFACT` unset; reuse the 9072484 artifact via `NITRO_REF`).
3. Pass bars: all P.1 PE bars unchanged (step-0 |ΔE| ≤ 1e-8, step-10 NVT/NPT PEs equal the recorded values to ≤ 1e-8); `MP_PERF rank=0` rows present for every step; `perf_nvt.json` non-empty; `NITRO_FLAGS` line present; `meta.json` has no null fields and records `nitro_base_sha`.
4. If the PE differs from the earlier gate beyond 1e-11, the instrumentation commit changed numerics: revert it and redo H1.
5. On pass: create annotated tag **`nitro-base`** on that commit; update R.1 ("Base commit" row) and R.3 to say "PR head `a240ba3485` + instrumentation commit `<sha>`".

Exit: `nitro-base` exists and is the commit that H4/H5 and all arms are built from.

Attempt 1 (2026-11-11): job 9077136 on `debug-scaling`, tag `nitro/base-verify/r2`, failed at link. `extern char **environ` inside `namespace uma` became `uma::(anonymous namespace)::environ`. The fix keeps that declaration at global scope with C linkage. `nitro/base-verify/r2` stays on the failing commit. The MPI library CMake found (`87e2045`) matches the earlier successful `7da48ce` build, so it is not part of this failure.

Attempt 2 passed. Job 9077571 on `debug-scaling`, 1 node, tag `nitro/base-verify/r3` = `efae85ff72`. NVT and NPT exit 0. Step-0 and step-10 \|ΔPE\| versus the recorded bars are 1.5e-11 to 5.8e-11 eV, inside the 1e-8 gate. Versus job 9072484 the same steps differ by 1.5e-11 to 5.8e-11 eV. That is the band P.1 already records for run-to-run noise, so H1 is not reverted. `MP_PERF rank=0` has 11 lines in each log (steps 0 through 10). `perf_nvt.json` has 6 rows after the 5-step skip; median `ms_total` is 822 ms with `MP_PERF` on. Rank 0 logged `NITRO_FLAGS: (none)`. `fairchem` is 2.21.0. The only null in that job's `meta.json` is `nitro_base_sha`, because this tag is created after the run. `nitro-base` is `efae85ff72`.

Status: **closed**. The 10-minute status poll for 9077571 is stopped.

### Step 6 · H4 — E0: noise floor and replication (`debug`, 3 jobs on different nodes)

1. Use `run_timing.pbs` (REPS = 3 per job) in **3 separate jobs**, submitted so the node sets differ (check `hostnames` in `meta.json`; resubmit a job that lands on a repeated node set or a throttled node).
2. Measure: (a) bitwise repeat of PE and forces within a job and across jobs (`compare.py bit`, `diff_forces.py`); (b) BASE-vs-BASE max|ΔF| and max|ΔW| (these set the force/virial bars in P.1); (c) per-job median step time and `MP_PERF` fields; within-job MAD and between-job range; (d) one extra pair with `UMA_MP_PERF` off vs on (instrumentation effect).
3. Output `pbs/out/nitro_e0/E0_SUMMARY.md` with the **minimum detectable effect** (the larger of 3 × within-job MAD and the between-job range, in ms and %), and the calibrated force/virial tolerances. Copy the values into P.1/P.2.
4. If BASE is not bitwise reproducible, P.1 "Bit-identity" is relaxed to the calibrated tolerance for all arms and this is written down.
5. If the minimum detectable effect exceeds 5% of the step time, add a rule to P.2: arms with an expected gain below the MDE (N3, N4) are judged only inside a stack, with ≥ 5 jobs.

Exit: E0_SUMMARY.md written; tolerances and MDE entered in the plan.

Result: three nodes (x4114, x4504, x4500), all different, jobs 9078185, 9078330, 9078397, `fail=0`. Perf-off job 9078443 on x4010 ran and wrote no `MP_PERF` lines; its loop time does not show a sync cost above the node spread. Summary: `pbs/out/nitro_e0/E0_SUMMARY.md`. The 10-minute poll is stopped.

Status: **closed**.

### Step 7 · H5 — Step-time decomposition (`debug`, 1 node; diagnostic only)

Question: is the ~400 ms in-graph all-reduce real communication or waiting for other ranks (compute imbalance, stream sync, `MP_PERF` event syncs)? It is the largest slice (≈ 48%) and no Nitro activity touches it.

1. Add an opt-in diagnostic knob `UMA_NITRO_PERF_BARRIER=1` (allowlisted, default off, perf-only): a timed `MPI_Barrier` / device sync immediately before each collective, so `ms_wait` and `ms_xfer` are reported separately. This is a small code change on branch `nitro/h5-diag` cut from `nitro-base` (not part of `nitro-base`); syntax-check first.
2. Run in one job, ABBA: BASE with `UMA_MP_PERF` on, BASE with it off (step time only), BASE with the barrier diagnostic. Three jobs on different nodes (H4 rule), same scripts.
3. Report: `ms_wait` vs `ms_xfer` per collective; effective GB/s per rank against the expected link/XCCL rate; per-rank compute time spread (is one rank always last?).
4. Decision rules (recorded, not acted on here): if waiting dominates, the cause is imbalance or sync, and the finding is added to X1 with the per-rank evidence; if transfer dominates at ≈ 2 GB/s, the finding is added to X1 as "transport inefficiency, outside Nitro scope". Either way the result feeds H6; it does not change the user's decision to continue to N5.

Exit: `pbs/out/nitro_h5/H5_SUMMARY.md` with the wait/transfer split and per-rank spread.

Result: jobs 9079409, 9079449, 9079543 on x4010, x4018, and x4112. Code is `nitro/h5-diag` `d14cdc36b5`, not `nitro-base`. Post-warmup medians: local drain 358–362 ms, all-reduce barrier 17–27 ms, all-reduce transfer 12.4–12.5 ms (0.809 GB, ~65 GB/s). Ranks 6 and 7 are last most often. The `fail=1` flag is the `UMA_MP_PERF=0` arm. The 10-minute poll is stopped.

Status: **closed**.

### Step 8 · H6 — Refresh the plan (no compute)

1. **Amdahl ledger.** Replace the 828 / 390 / 118 ms narrative with a table from H3–H5 of the BASE step: host graph build, compute (fwd + bwd without collectives), collective wait, collective transfer, other; each with median, MAD and share. For each activity, add its upper bound as a function of the ledger (e.g. N1 ≤ host-build share; N5 ≤ compute share) and the cumulative ceiling versus the BASE step.
2. **Update expectations** in N1–N5 ("Expected gain" lines) and X1 with the measured ledger; mark N3 as "small by design (real edges ≈ 58–60k vs cap 65 536)".
3. **Update P.1–P.4**: calibrated tolerances and MDE (H4), replicate rule and revised effect threshold (H2.2), arm-toggle rule (H1.4), ABBA runner row and `run_abba.pbs`, `abba.inc`, `ENV_AUDIT.md` rows in the harness file table.
4. **Fix stale text**: "baseline not yet verified", R.4 rows ("`pbs/` harness missing", the artifact row), R.5 disk line (artifacts auto-deleted; project at the measured usage), P.3 item 3.
5. **Close-out table** at the end of this section: each F-item with its closing stage and result.

Exit: no "TBD" or stale statement left in R, P, 1; H-table all closed; then step 9 (S0) may start.

### Measured BASE step (H6)

Source: H5 ON slots, three nodes, 104 post-warmup steps each. The number below is the median of the three job medians. Within-job MAD of `ms_total` is about 0.6–1.1 ms. The between-job range of `ms_total` is 6.5 ms (H4 MDE is 6.27 ms).

`ms_fwd` + `ms_bwd` include the in-graph collective. H5 showed that collective's 390 ms timer is 358 ms of local queue drain, 20.5 ms of rank skew, and 12.5 ms of all-reduce transfer. The drain stays in compute. All-gather adds 0.5 ms of wait and 6.7 ms of transfer.

| Row | Median (ms) | MAD across the 3 jobs (ms) | Share of 825 ms |
|---|---|---|---|
| Host graph build (`ms_graph`; 118 ms of it is `ms_vesin`) | 124 | 0.1 | 15% |
| Compute (forward + backward, without rank-wait and transfer) | 659 | 1 | 80% |
| Collective wait (MPI barrier after the local drain) | 21 | 3 | 2.5% |
| Collective transfer (all-reduce 12.5 ms + all-gather 6.7 ms) | 19 | 0.1 | 2.3% |
| Other (force all-reduce) | 1 | 0.2 | 0.1% |
| Step (`ms_total`) | 825 | 3 | 100% |

N1 can remove at most the 118 ms host neighbor build (14% of the step). The rest of the reachable time is in the 659 ms compute row (80%), through N2 and N5. Communication is 40 ms (5%), and Nitro does not target it. Removing graph and compute entirely would leave 41 ms. That ceiling is not a forecast. There is no speedup target (H.0); this ledger shows where the time is.

| Activity | Upper bound | Note |
|---|---|---|
| N1 | 118 ms (`ms_vesin`) | 14% of the step |
| N2 | unmeasured; only the part of the 659 ms that is `index_add_` | part of the compute row |
| N3 | small by design: real edges are about 58–60k against the 65,536 cap | cleanup, not a large lever |
| N4 | about 1% of the step, ~8 ms | small; near the 6.27 ms MDE, judged in the stack |
| N5 | at most the 659 ms compute row | the largest lever; the FP64 GEMM part of it is not fusable |

### Close-out

| ID | Result |
|---|---|
| F1 | Closed in H1 and proved in H3: `MP_PERF` rows are present. |
| F2 | Closed in H1: new arms use `UMA_NITRO_*`. |
| F3 | Closed in H3: `nitro-base` is `efae85ff72`. |
| F4 | Closed in H2: `run_abba.pbs`. |
| F5 | Closed in H2 and H4: the job is the replicate. MDE is 6.27 ms. |
| F6 | Closed in H2: exports are deleted unless `KEEP_ARTIFACT=1`. |
| F7 | Closed in H2: untracked source fails the tree check; PBS logs go to `pbs/out/`. |
| F8 | Closed in H2: `MPICH` comes from `env.inc`. |
| F9 | Closed in H2 and H3: fairchem 2.21.0 is recorded and matches. |
| F10 | Closed in H5: the 390 ms timer is 358 ms local drain, 20 ms rank skew, 12.5 ms transfer at ~65 GB/s. |
| F11 | Closed in H6: this ledger. |
| F12 | Closed in H6: the stale harness and baseline lines in R and the status header are updated. |
| F13 | Still out of scope (H.8). |

Status: **closed**. Step 9 (S0) may start.

### Step 10 · Z1 — ZIF-8 3×3×3 as a second test system (`debug-scaling`, 1 node per job)

Added in revision 9. Same protocol as NaCl; only the structure differs. The H.7 rules (sequential close, commit and push, closure report) apply.

The structure: the P1 4×4×4 CIF in the read-only project `rosen_uma_ZIF` is reduced to the 276-atom conventional cell (a = 16.99 Å) and replicated 3×3×3, with the same 0.05 Å Gaussian rattle (seed 0) as NaCl. Without the rattle the data file equals the existing `zif8_3x3x3_npt_nvt_chain/sim/data.zif8` exactly (checked). Element order H, C, N, Zn (types 1–4). Timestep stays 0.001 ps as for NaCl (the rosen runs use 0.0005; for a 10-step parity gate this does not matter, and it keeps the protocol identical).

| Item | Work | Exit |
|---|---|---|
| Z1.1 Code | `make_zif8.py`, `systems.inc`, and `lib.inc`, `run_gate.pbs`, `run_timing.pbs`, `run_abba.pbs` take `NITRO_SYSTEM` (default `nacl`; the NaCl deck is unchanged). Without recorded reference PEs, `run_gate.pbs` runs record-only. | Done: atom count 7,452 and type counts match; unrattled data equals the rosen file; the NaCl and ZIF-8 decks print correctly; an unknown system name fails. |
| Z1.2 Syntax check | Subagent (`grok-4.7-high`) on the edited `.py`, `.inc`, `.pbs` files before any `qsub`. | `COMPILE_OK`. Done for this commit (`py_compile` with 3.11, `bash -n`, sourcing test for `nacl`, `zif8` and an unknown name). Re-run after any further edit and before the first `qsub`. |
| Z1.3 Record job | Tag `nitro/z1/r1`; `qsub -v NITRO_TAG=nitro/z1/r1,NITRO_SYSTEM=zif8,KEEP_ARTIFACT=1 run_gate.pbs`. | NVT and NPT exit 0, volume moves, no "virial refused"; step-0 PE equal in NVT and NPT; the three reference values printed (`RECORD_ONLY`). |
| Z1.4 Record | Write step-0, NVT step-10 and NPT step-10 PE into `systems.inc`; commit; tag `nitro/z1/r2`. | References in the repo. |
| Z1.5 Verify job | `qsub -v NITRO_TAG=nitro/z1/r2,NITRO_SYSTEM=zif8 run_gate.pbs` on a different node. | All four \|ΔPE\| < 1e−8 eV against the recorded values. |
| Z1.6 Noise floor and ledger | Three jobs on different nodes with `run_timing.pbs` (reps 3, 30 steps, reusing the kept artifact through `NITRO_REF`). Report ZIF-8 MDE (between-job range and 3 × pooled MAD), bitwise or not, force noise, and the step ledger rows (graph, compute, wait, transfer, other) with shares. Record real edges per rank against the 65,536 cap, and the maximum neighbours per atom against `max_neighbors` = 300. | `pbs/out/nitro_z1/Z1_SUMMARY.md`. |
| Z1.7 Plan | Put the ZIF-8 numbers into P.1, P.2 and a ZIF-8 ledger next to the NaCl one; update R.6; commit and push. | Z1 marked closed; step 11 may start. |

Failure points: (a) the exported artifact is system-specific, so never reuse a NaCl artifact for ZIF-8 (and the reverse); (b) a ZIF-8 reference taken from one node must reproduce on another (Z1.5 checks); (c) ZIF-8 has light H atoms, so 0.001 ps with H is a looser integrator than the rosen runs, which is acceptable for 10–30 steps but not for physics claims; (d) the rattle is not in the rosen data files, so ZIF-8 PEs here are not comparable with the rosen logs.

**Result (2026-11-12).** Full detail in `pbs/out/nitro_z1/Z1_SUMMARY.md`.

- Z1.3 record job 9084279 (`nitro/z1/r1`): NVT and NPT exit 0, NPT volume 132,417 → 132,590 Å^3, no "virial refused", step-0 PE NVT vs NPT equal to 2.4e-10 eV. Reference PEs (eV): step 0 −45916.897506276422; NVT step 10 −46244.412436485371; NPT step 10 −46245.409267788964.
- Z1.4: references written to `systems.inc`; tag `nitro/z1/r2` (`62d5daa`).
- Z1.5 verify job 9084359 on a different node: |ΔPE| = 1.164e-10, 5.821e-11, 3.638e-10, 7.276e-11 eV against the 1e-8 bar. Pass.
- Z1.6 three timing jobs on three nodes (9084446, 9084482, 9084541). Median step 876.1 ms. **MDE 3.11 ms (0.35%)**: between-job range 3.11 ms, 3 × pooled MAD 2.49 ms. PE not bitwise: max |ΔPE| 4.657e-10 eV; max |ΔF| 1.717e-12 eV/Å. The NaCl same-ops tolerance of 1e-10 eV does not transfer; the ZIF-8 same-ops tolerance is **1e-9 eV** (the 1e-8 pass bar and 1e-9 eV/Å force bar are unchanged).
- ZIF-8 step ledger (ms; median of three job medians): graph 54.2 (6.2%; `ms_vesin` 49.8), forward 262.4 (29.9%), backward 556.4 (63.5%), force all-reduce 1.05, other 2.2. The collective timers (`ms_allreduce` 499.3, `ms_allgather` 3.05) sit inside forward+backward. Their split into drain, wait and transfer needs the `nitro/h5-diag` build and was not run for ZIF-8.
- Edges: engine edges are padded to 65,536 in every run and the engine throws on overflow, so none overflowed. Real edges from geometry (6 Å, periodic): 437,170 directed in total, about 36.4k per rank (36.0–36.8k), so about 44% of the edge slots are padding (NaCl: under 12%). Max neighbours per atom is 68, far below `max_neighbors` = 300. The per-rank edge count is an estimate, not an in-engine count.
- Provenance: the timing jobs pair the `r2` binary with the artifact exported by the `r1` record job (`r1` → `r2` changed only `systems.inc` values and one echo string; the `r2` job's own export reproduced the PEs).

Status: **closed.** Step 11 may start.

### H.7 Rules that apply to every H stage

- **Report closure (rule, 2026-11-12).** Every status report says whether the stage in progress is closed, and lists the earlier stages that are already closed.
- **Sequential close (rule, 2026-11-11).** Never open an item in a later stage while any item in an earlier stage is not fully closed. A stage is fully closed only when **every** item in it is done: its exit criteria met, its plan status updated, and its commit pushed (or the push explicitly waived for that stage per the rule below). This applies to items within a stage as well as to whole stages: no H2 work starts while any H1 item is open, and so on through step 8 (H6), then step 9 onward. A blocked item stays in its own stage; do not skip ahead and come back.
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
(NaCl 14³, W=12, FP64, `uma-s-1p2`, seed 0; ZIF-8 3×3×3 values are in `systems.inc`, step 10): step-0 PE −74142.191907631001 eV; step-10 PE NVT
−74095.578889731943 eV, NPT −74098.199120459016 eV. ZIF-8: step-0 PE −45916.897506276422 eV; step-10 PE NVT −46244.412436485371 eV, NPT −46245.409267788964 eV.

| Check | Pass bar | Notes |
|---|---|---|
| `lmp` exit code | 0 for NVT and NPT | |
| Thermo | steps 0…N present, `pe` and `temp` finite | |
| NPT | volume moves, no "virial refused" | needs virial, AC off |
| No hen code | no `UMA_HEN_ROOT` and no hen code paths (patches, shim, scripts) in the log | the checkpoint path under `hen/uma-cache` is allowed |
| Step-0 \|ΔPE\| vs BASE | ≤ 1e−8 eV | reorder-only changes gave 1.5e−11…5.8e−11 in Waves A–E |
| Forces | max\|ΔF\| ≤ max(1e−9 eV/Å, 10 × BASE-vs-BASE noise) | E0 max noise 7.35e−14 eV/Å, so 10× noise is below 1e−9 and the bar stays **1e−9 eV/Å** |
| Virial (NPT) | max\|ΔW\| ≤ 1e−7 eV | E0 was NVT and did not dump the virial; bar unchanged |
| Bit-identity | \|ΔPE\| ≤ 1e−10 eV (NaCl), ≤ 1e−9 eV (ZIF-8) | E0 is not bitwise (NaCl max \|ΔPE\| 5.821e−11 eV; ZIF-8 4.657e−10 eV). Arms that claim the same ops use the tolerance of their system |
| Independent checks | relative force error to force RMS; finite-difference check on a few atoms; NVE 200-step drift | a noise-scaled bar alone can hide a real bug |
| Worker ladder | any arm touching the graph, autograd or kernels passes on a small DIAG cell at W=1 first, then W=2 → 4 → 6 → 12, before the W=12 gate counts | DIAG cells are labelled and never used to claim a result |
| NPT/virial parity | mandatory for N1, N2, N5 | stress path is `dE/dcell` through edge vectors |
| Forced edge cases | atoms outside the box, tiny cell (multi-image), triclinic cell, pair at exactly the cutoff, a >300-neighbor dense case, pad present, multi-chunk | the 10-step gate on uniform NaCl does not exercise these |

### P.2 Timing

- Timing runs: **30 steps** NVT; steps 0–4 are warm-up and are discarded. Gate runs of 10 steps are for correctness only.
- Statistic: **median** over steps 5–29 of `ms_total` and each `MP_PERF` field, with MAD, min and max (`compare.py perf`).
- Instrumentation: `MP_PERF` synchronizes, so compare arms with the same instrumentation on, and take one extra pair of runs with it off to confirm.
- **Replication (revision 5, H.0/H2.2):** the replicate unit is the PBS job; every claim needs ≥ 3 jobs on different nodes. Verdict: the gain has the same sign in all jobs **and** the median per-job gain exceeds `max(3 × pooled within-job MAD, between-job range)`. The per-step interval below is descriptive only.
- **E0 MDE (H4):** 6.27 ms, 0.76% of the 822.6 ms median `ms_total`. It is the between-job range. 3 × the pooled within-run MAD is 1.57 ms. The MDE is under 5% of the step, so N3 and N4 are not forced onto a five-job stack by that rule.
- **ZIF-8 MDE (step 10, Z1.6):** 3.11 ms, 0.35% of the 876.1 ms median `ms_total`; it is the between-job range (3 × pooled MAD is 2.49 ms). NaCl and ZIF-8 MDEs are per system and never pooled.
- **Effect threshold (per-step, descriptive):** the paired step-time difference has a 95% interval excluding 0 **and** the median gain exceeds 3 × the pooled MAD; treat 5% as a priority signal, not a pass/fail line, so small deterministic savings (N3, N4) can still be judged in a stack.
- Speedup = `median(BASE) / median(arm)`, labelled with cell, W and queue.

### P.3 Control of variables

1. One PBS job holds several arms in **ABBA** order (BASE, A, A, BASE).
2. One factor changes per comparison; binary, artifact, env, deck, seed, run length, rank-to-tile mapping stay frozen.
3. Gate new behaviour behind `UMA_NITRO_<ID>` toggles (allowlisted env fallback, H1.4) or `pair_style` keywords with default = BASE, so one binary serves both arms where possible. Plain `UMA_*` env vars are **ignored** by the engine unless allowlisted.
4. If an arm changes export, re-export and also run BASE on that re-exported artifact.
5. Fixed `velocity ... loop geom`-equivalent seed and an identical data file for the chosen system (`make_nacl.py` or `make_zif8.py`, `NITRO_SYSTEM`).
6. Record hostnames and tile state; discard a job whose nodes throttle.
7. Every run writes `meta.json` (`meta.py`): git describe/SHA, branch, tag, `nitro-base` SHA, binary sha256, artifact id, env, hosts, queue, jobid.
8. Every PBS job uses a per-run `timeout`, records exit codes and keeps going after a failing arm; risky arms run last.

### P.4 Harness files (in `pbs/nitro/`)

| File | Purpose |
|---|---|
| `env.inc` | compute-node environment (`fxpu` conda env, per-arm `NITRO_BUILD`, checkpoint path) |
| `lib.inc` | clean-tree/tag check, runtime env, LAMMPS deck writer (optional force dump), `mpiexec` runner |
| `make_nacl.py` | NaCl N×N×N data and xyz files (same construction as Wave E) |
| `make_zif8.py` | ZIF-8 N×N×N data and xyz files from the rosen 4×4×4 CIF (same rattle as NaCl) |
| `systems.inc` | per-system settings (`NITRO_SYSTEM` = `nacl` or `zif8`): data file, elements, masses, atom count, reference PEs |
| `export_one_rank.sh` | one rank of the W=12 export |
| `run_gate.pbs` | build into `build-nitro/<sha7>/`, export (or reuse), NVT+NPT 10 steps, PE bars against the recorded values, force dump at steps 0 and 10, `SUMMARY.txt`, `meta.json`, `run.env` |
| `run_timing.pbs` | E0: `REPS` identical 30-step NVT runs in one job; MP_PERF medians, bitwise PE and force repeat |
| `run_abba.pbs` | ABBA slots for one job (H2). H5 used it. |
| `abba.inc` | Parses an arm spec into BASE, arm, arm, BASE order |
| `ENV_AUDIT.md` | Allowlist of `UMA_*` names that may fall back to the environment (H1) |
| `compare.py` | `pe`, `bar`, `perf`, `bit`, `abba`, `abba-test` |
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
decides keep or drop by the H.0 verdict), and its own failure points.

### N1 [step 11] — Device-side radius-graph build with fixed capacity and overflow flag
- **Nitro idea:** `padded_radius_graph` (`bucket_fill_direct` → `fused_candidates` → `graph_scan` → `edge_emit`) rebuilds the graph on the device every step; an overflow flag is checked after the build.
- **Do:**
  - N1-a: build the graph on XPU with composed torch ops, FP64. **Output order must equal the reference builder's order: by center ascending, then (distance, neighbor, ox, oy, oz)** (`emit_graph` in `neighbor_list.cpp`). That order is also the `max_neighbors` truncation tie-break, and it keeps the summation order identical to BASE.
  - N1-b is step 16, after step 15 (S4). Same stages as N1-a, as custom SYCL kernels. It runs whether N1-a was a win or not, once N1-a is correct. The user set this order on 2026-11-12. Step 11 closes on N1-a and the energy gates.
  - Reuse the existing cross-rank pad-cap overflow check in `mpi_peer_predictor.cpp` (all-reduce of an overflow flag before the forward); add the device builder's own overflow flag to it. On overflow all ranks stop with an explicit error.
- **Control:** time the current host `build_neighbor_graph` + H2D copy with device events, then N1-a/N1-b in the same job (ABBA). Record the device-build time as a fraction of the host time. There is no cutoff: N1-a is adopted only if it is correct and a win by the H.0 verdict.
- **Tests:** edge-set equality vs the current builder; ΔPE/ΔF/Δvirial vs BASE; NPT parity; W ladder.
- **Expected gain:** at most 118 ms (`ms_vesin`), 14% of the 825 ms step, 1.17×. The host build is measured (H6). The device build is adopted only if it is correct and a win.

**N1-a selftest (job 9085554, node x4300c3s6b0n0, not a verdict).** `pbs/nitro/n1/n1_selftest` against `build_neighbor_graph`. Passed on CPU tensors and on XPU: rocksalt 4³, atoms shifted outside the box, random orthorhombic N=128, tiny cell (multi-image), dense truncation at `max_neighbors=5`, a pair at exactly the cutoff, triclinic N=20, and a deliberate cell-cap overflow (code 1). Wrap of out-of-box positions matches the scalar wrap to 5.3e-15 Å. NaCl-sized 14³ (21,952 atoms, 702,464 edges) matches exactly. One-node timing, synchronized chrono, three reps: host build + H2D copy 112 / 91 / 74 ms; device build 136 / 68 / 60 ms. The first rep is cold. This is one job, so it is not a win or a loss. The candidate set is chosen with torch ops on the tensor device; order and `max_neighbors` truncation use the reference comparator, because a torch distance and the scalar distance differ by an ulp inside an equal shell and that reordered edges (job 9085272).

**W=1 DIAG (job 9087586, tag `nitro/n1/r2`, node x4613c0s2b0n0, not a gate result).** NaCl 4³, 512 atoms, FP64, one tile. Same binary and same W=1 artifact. The base arm logged `NITRO_FLAGS: (none)`. The n1 arm logged `NITRO_FLAGS: UMA_NITRO_N1=1`. Against the same-job base arm: NVT |ΔPE| 2.3e-13 eV at step 0 and 6.8e-13 eV at step 10; NPT |ΔPE| 0 at step 0 and 9.1e-13 eV at step 10 (bar 1e-8). Max |ΔF| 6.0e-15 eV/Å (bar 1e-9). NPT volume 11482 → 11492 Å³. No "virial refused". First submit 9087543 failed the harness check because a W=1 export writes `model_traced.pt` into the artifact root, not `w1/r0`; the export itself had succeeded. That check now accepts the W=1 layout. W=12 export is unchanged.

**W=2 DIAG (job 9087641, tag `nitro/n1/r3`, node x4502c2s7b0n0, not a gate result).** Same cell and bars. Base arm `NITRO_FLAGS: (none)`, n1 arm `UMA_NITRO_N1=1`. NVT |ΔPE| 2.3e-13 eV at step 0 and 0 at step 10. NPT |ΔPE| 4.5e-13 eV at step 0 and 0 at step 10. Max |ΔF| 2.4e-14 eV/Å. NPT volume moved. No "virial refused".

**W=4 DIAG (job 9087698, tag `nitro/n1/r4`, node x4211c0s3b0n0, not a gate result).** Same cell and bars. Base arm had no nitro flags. The n1 arm logged `UMA_NITRO_N1=1`. NVT |ΔPE| 2.3e-13 eV at step 0 and 0 at step 10. NPT |ΔPE| 4.5e-13 eV at step 0 and 0 at step 10. Max |ΔF| 1.9e-14 eV/Å. NPT volume moved. No "virial refused".

**W=6 DIAG (job 9087772, tag `nitro/n1/r5`, node x4613c6s3b0n0, not a gate result).** Same cell and bars. Base arm had no nitro flags. The n1 arm logged `UMA_NITRO_N1=1`. NVT |ΔPE| 2.3e-13 eV at step 0 and 4.5e-13 eV at step 10. NPT |ΔPE| 2.3e-13 eV at both step 0 and step 10. Max |ΔF| 1.9e-14 eV/Å. NPT volume moved. No "virial refused".


### N2 [step 18] — Target-sorted edge→node segmented reduction
- **Nitro idea:** `EdgeCSR` (edges sorted by target node) with `_segreduce_tgt_*` / `fused_m0_segreduce` instead of atomic scatter.
- **Starting point (verified in code):** the current host graph is already emitted by center ascending, `shard_edges` preserves that order, and the pad edges are **prepended** with center = the rank's first owned node, so the padded shard is already sorted by center. N2 does **not** depend on N1, as long as N1 keeps that order.
- **Do:**
  - N2-a: per-segment reduction with torch ops over the already center-sorted edges (offsets from a bincount/cumsum of centers).
  - N2-b: SYCL kernel, one work-group per segment, with custom backward.
- **Control:** microbenchmark vs the current `index_add_` on the real chunk shapes, forward and forward+backward. Record the ratio; N2-b follows N2-a whenever the SYCL gate (S0–S4) has passed. Either is adopted only if correct and a win.
- **Expected gain:** only the `index_add_` share of the 659 ms compute row. Unmeasured.

### N3 [step 29] — Measured edge capacity with margin
- **Nitro idea:** `plan_capacities` / `_measured_caps` with `EDGE_MARGIN` (1.30) instead of a worst-case cap.
- **Do:** export with cap = measured max per-rank shard edges × {1.05, 1.10, 1.30}, rounded up (N3-a/b/c). Nitro's remeasure-and-retry is **not** copied (capacity is baked into the traced model); overflow is a hard error.
- **Control:** the three margins against the current cap 65 536.
- **Expected gain:** small by design. Real edges are about 58–60k against the 65,536 cap, so the padded fraction is under about 12% of edge slots. That is cleanup, not a large lever for NaCl. For ZIF-8 (step 10) the real edges are about 36k per rank, so about 44% of the edge slots are padding; N3 is a larger candidate there (judged per system, H.0).

### N4 [step 30] — MOLE merge at export, keyed by composition
- **Nitro idea:** merge the mixture-of-linear-experts per system (`_composition_key`, `prepare_for_inference`).
- **Do:**
  - N4-a: fold the per-system expert coefficients at export; store composition + charge + spin in metadata and refuse a mismatch.
  - N4-b: fold other composition-constant subgraphs.
- **Control:** unfolded vs folded artifact; bit-level or tolerance equality of PE/F.
- **Expected gain:** about 8 ms (1% of 825 ms), near the 6.27 ms MDE. Optional.

### N5 [steps 19–28] — Fused edge-chain blocks (Nitro kernel set)
Each block: **microbenchmark → one layer → all layers → gate**. Every block is attempted. A block whose FP64 microbenchmark is not detectably faster than the unfused torch sequence (after a tuning round) is closed flat or loss and stays out of the stack.

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
- **Expected gain:** at most the 659 ms compute row. The size is unknown until the microbenchmarks. No threshold applies.

### Order
(N5 execution detail is in §5; F6 is folded into each block stage there.)
N1 → N2 → N5 microbenchmarks (F1, F2, F4, F5, F6, F3) → N5 in-model for blocks that passed → N3 → N4 → stack the passing activities (`nitro/stack/s<k>`) and run the gate.

### Handles
`nitro/n1-a`, `nitro/n1-b`, `nitro/n2-a`, `nitro/n2-b`, `nitro/n3-cap`, `nitro/n4-a`, `nitro/n4-b`, `nitro/n5-f1` … `nitro/n5-f7`, `nitro/stack/s<k>`; tags `nitro/<id>/r<k>` before each `qsub`.

---

## 2. Failure points and robustness (Nitro activities only)

Severity: **S** can ship wrong physics or invalidate results; **H** likely to block or waste effort; **M** reduces confidence.

### Applies to all
| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| X1 | H | The ~390 ms all-reduce timer is mostly the local XPU queue draining (~360 ms). Rank skew is ~20 ms and the transfer is ~12 ms at ~65 GB/s (H5). Nitro has no activity for that local queue. The old "zero compute ⇒ 1.57×" cap counted this drain as communication. | H6 puts the drain, the skew, and the transfer on separate ledger rows |
| X2 | H | Nitro's Triton/CuTe kernels cannot run in the Python-free C++ engine. | Prototype one trivial SYCL op end-to-end (export → load → autograd) first |
| X3 | S | Checkpoint mismatch: Nitro is validated for `uma-s-1p2p1`; local is `uma-s-1p2` (composition embedding differs). | Reuse edge-side designs only; parity vs the pinned fairchem 2.21.0 forward |
| X4 | H | Every compute activity needs export, re-export (27 GB artifacts) and re-validation. | Disk and job budget up front |
| X5 | M | Nitro's speedups are NVIDIA-tuned; PVC FP64 behaviour differs. | Per-block microbenchmark verdict |
| X6 | M | Licence: adapting MIT/BSD code into the GPL-2 tree needs notices; CUTLASS DSL is NVIDIA-licensed. | Clean-room re-implementation |

### N1
| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| N1-1 | S | Edge set differs from the current builder (cutoff inclusivity, 300-neighbor truncation, self-edges, tiny/triclinic cells, atoms outside the box). | Edge-set equality tests across crafted cases |
| N1-2 | S | Non-deterministic or different edge order → irreproducible forces or a different summation order than BASE; breaks N2. | Reproduce the reference order (center, distance, neighbor, ox, oy, oz) |
| N1-3 | H | Overflow cannot be remeasured (baked cap). | Read the flag every step; fail fast |
| N1-4 | M | Ranks could build different edge sets. | Cross-rank checksum on debug runs |
| N1-5 | M | Boundary flips at r ≈ 6.0 (negligible energy effect, not bit-identical). | Tolerance band in the equality test |
| N1-6 | M | Host-build time. | Measured in H6: `ms_graph` 124 ms, of which `ms_vesin` is 118 ms. The N1 control still has to beat that host time |

### N2
| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| N2-1 | M | (Corrected) Pad edges are prepended with the first owned node as center, so the padded shard stays sorted; but this holds only while the graph is center-sorted (true for the CPU builder, **not guaranteed** for a vesin/device builder) and chunk boundaries can still split a segment. | Assert sortedness at runtime in debug; test multi-chunk |
| N2-2 | S | Global cumsum-difference loses FP64 precision; custom kernel needs a custom backward. | Per-segment reduction; FP64 `gradcheck` |
| N2-3 | H | May not be hot at all. | Microbenchmark gate |

### N3
| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| N3-1 | H | Tighter cap turns a density/NPT excursion into a mid-run crash for ≤ ~11% gain. | Opt-in only because of the crash risk; measured regardless |
| N3-2 | M | Cap must cover the max over ranks and is coupled to chunk size. | Existing cross-rank agreement; one export per cap |

### N4
| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| N4-1 | S | Silent wrong energy if the folded artifact is used with a different composition/charge/spin, or composition changes mid-run. | Metadata key; refuse mismatch |
| N4-2 | M | Coefficients are a runtime input in some modules and constant in one; interface changes. | Per-module audit |
| N4-3 | M | ~1% gain will not clear a threshold alone. | Judge in a stack; recorded even if flat |

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
| B4 host build | **Resolved.** CPU neighbor list. H6 measured `ms_vesin` 118 ms and `ms_graph` 124 ms on the BASE step. |
| B5 VCS | **Decided**: commit harness on `uma-nitro`; run outputs/binaries excluded via `.git/info/exclude`; clean check = tracked files only. |
| B2 SYCL op | Open, but now permitted. Prior art found (PyTorch `SyclExtension`) and the engine already compiles `icpx` objects that use torch's XPU queue (`xccl_peer.cpp`). Needs the prototype in §4 before N5. |
| B3 device-build speed | Open. Decided by the N1 measurement. |

### 3.1 Blocking items (resolve before any implementation)

| # | Blocking item | Evidence | Required resolution |
|---|---|---|---|
| B1 | **Architecture match is inferred, and the code contradicts itself.** The traced chunk shows 9×9 Wigner and `[3,4,2]` m-splits (lmax=2), but `export_blocks_xpu.py` docstrings and comments repeatedly say `[Ec,25,25]` wigner (lmax=4). Every N5 kernel shape depends on this. | `export_blocks_xpu.py` lines 20, 145, 336, 385 vs traced `torch.zeros([1,9,9])` | Read `lmax`, `mmax`, `sphere_channels`, `num_layers` from the actual checkpoint config (`backbone.lmax`) on a compute node and write them into this plan. Do not start N5 before this |
| B2 | **No SYCL/C++ custom compute op exists in the engine yet.** Only SYCL use is `xccl_peer.cpp` compiled by `icpx` into an object; the registered ops (`uma_ckpt`, `uma_halo`, `uma_peer`) are checkpoint/communication ops. A compute kernel needs an `icpx` kernel object, a `TORCH_LIBRARY` registration with Autograd, an exporter-side stand-in, and a validated trace. | `CMakeLists.txt` lines 97–156; `TORCH_LIBRARY` greps | Run the X2 prototype (one trivial elementwise SYCL op through export → traced load → autograd → W=12) and record whether it works. N5 is blocked until it does |
| B3 | **Device-build speed is not a given.** The code comment in `mpi_peer_predictor.cpp` reports a vesin rebuild of ~1534 ms/step at N=32 (CUDA vesin), i.e. a device build can be slower than the 118 ms host build. | `mpi_peer_predictor.cpp` ~lines 375–381 | Measure it; adopt N1-a only if correct and a win; do not assume the Nitro-style builder wins |
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
| S0 (step 9) | Toolchain check: build the PyTorch tutorial `mymuladd` in **FP64** with `SyclExtension` into a scratch dir (`TORCH_XPU_ARCH_LIST=pvc`) and run it | compile on login, run on 1 compute node | Compiles with the `fxpu` torch; result equals `a*b+c` in FP64 (rel. error ≤ 1e-15); no ABI or `libsycl` errors |
| S1 (step 12) | Engine-style C++ test (no Python): standalone exe built like the engine links the new `.o`; call the op on XPU FP64 tensors, then `torch::autograd::grad` | 1 compute node | Forward equals the torch reference (≤ 1e-15 rel.); gradient equals the analytic formula and a central finite difference (≤ 1e-9); bitwise reproducible across 10 repeats |
| S2 (step 13) | Export round trip: trace the small module with the Python stand-in, save the `.pt`, load it in a C++ process that links the op, run forward+backward | 1 compute node | Loads without "unknown builtin op"; outputs and gradients equal eager torch (≤ 1e-15 / 1e-9) |
| S3 (step 14) | Link into `uma_engine` and rebuild `lmp`; run the existing NaCl 14³ W=12 NVT+NPT gate with the op linked but **unused** | `debug` job, ABBA vs BASE | Step-0 PE = −74142.191907631001 eV (|ΔE| ≤ 1e-8), identical to BASE; no regression in step time beyond noise; no new static-init or load errors |
| S4 (step 15) | 12-tile run: `mpiexec -n 12` test that calls the op repeatedly interleaved with an `uma_peer` all-reduce | 1 compute node | No hang; results identical on all ranks; op and collective ordering unchanged |
| S5 (step 17) | Cost measurement: launch overhead per call, FP64 bandwidth vs `torch` elementwise on the same tensors at the real chunk shapes | 1 compute node | Report only; this is the baseline for the N5 micro stages |

**S0 result (2026-11-12).** Job 9083937 on `debug-scaling`, node x4009c2s7b0n0, tag `nitro/s0/r2`. Torch 2.13.0+xpu, icpx 2026.1.0. FP64 `y = a*b+c` on 2^20 elements: abs 1.776e-15, rel 1.537e-16, `S0_PASS`, exit 0. The `.so` loaded `libsycl.so.9` from the `hen-xpu` env (no load error). Code is on `nitro/n5-proto`, not on `uma-nitro`.

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

It does not fuse any real UMA block, change the exported model, or alter `max_neighbors`/edge sets. It is a toolchain and integration test only. Real fusions (F1–F6) start only after the gate in §4.3 passes. Each block is then microbenchmarked and judged by the H.0 verdict.

---

## 5. N5 — dedicated step plan (revision 7, 2026-11-12)

This section replaces the one-paragraph N5 in §1 for execution purposes. Each N5.x stage is a numbered step in R.6 (N5.0 = step 19 … N5.9 = step 28; N5.10 = F7 = step 32, deferred). §1 stays as the summary. The stage rules of H.7 apply to every stage here: sequential close, commit and push at the end of each stage, closure reported in every status report, syntax-check subagent after edits and before `qsub`, `debug-scaling` only, FP64 only.

### 5.1 What is being fused (read from the pinned fairchem 2.21.0 code, `uma-s-1p2`)

One layer, one edge chunk (E = 65,536 padded edges, about 58–60k real; C = 128 channels; lmax = mmax = 2, so 9 coefficients). The exporter calls `Edgewise.forward_chunk` once per chunk and each chunk recomputes its own wigner and `x_edge` (`export_blocks_xpu.py`, `_ChunkCore`, line ~322). The stock sequence is:

| # | Operation (stock source) | Tensor shapes (derived; confirmed by the N5.1 shape dump) | Fused by |
|---|---|---|---|
| 1 | `x_full[src]`, `x_full[dst]`, `cat`, `bmm(wigner, ·)` (`node_to_edge_wigner_permute`) | gather to [E,9,128] twice, cat [E,9,256], bmm with [E,9,9] | F4 |
| 2 | Radial MLP `rad_func(x_edge)`: Linear, LayerNorm, SiLU, Linear, LayerNorm, SiLU, Linear (`RadialMLP`) | [E,edge_features] → [E,128] → [E,128] → [E,~1536] | F1 (LN+SiLU), F3 (chain) |
| 3 | conv1: m=0 multiply by radial, `fc_m0` Linear (768→640); m=1,2 multiply by radial, block GEMM (`SO2_m_Conv_Block`, `x_cat @ w_block.T`) | [E,9,256] → [E,9,128] plus gating [E,256] | F4 |
| 4 | Gate (`GateActivation`): sigmoid, `index_select` to 8 components, SiLU on the scalar row, multiply, `cat` | [E,256] and [E,9,128] → [E,9,128] | F2 |
| 5 | conv2: `fc_m0` (384→384), block GEMMs for m=1,2, `cat` | [E,9,128] → [E,9,128] | F5 |
| 6 | `bmm(wigner_inv, ·)`, `zeros`, `index_add_` over `edge_index[1]` (`permute_wigner_inv_edge_to_node`) | [E,9,128] → nodes [N,9,128] | F5 (+ N2) |

Facts that shape the plan:

- **Checkpointing runs each chunk three times** per layer per step: a no-grad forward, a recompute forward with grad, and the backward. A forward fusion pays twice, a backward fusion once.
- **Weights need no gradients** in inference (to be confirmed in N5.1). The hand-written backward only returns gradients for `x_full`, `wigner`, `wigner_inv_envelope` and `x_edge` (these carry forces and virial). That is a smaller job than a training backward.
- **FP64 GEMM floor (paper estimate, not measured).** About 3.7 MFLOP per edge per layer forward, so about 240 GFLOP per chunk-layer, about 0.7 TFLOP over the three passes, about 2.9 TFLOP per step per rank over 4 layers. At a realistic 10–15 TFLOP/s per tile that is roughly 200–300 ms of the 659 ms compute row that no fusion can remove. The memory-bound rest (gather, cat, elementwise, gate, `index_add_`, LayerNorm) is where N5 can act. N5.1 measures the real split.
- **Memory traffic.** One [E,9,256] FP64 tensor is 1.2 GB and [E,~1536] is 0.8 GB. Stock step 1 moves about 90 KB per edge against about 36 KB fused, so F4 has a believable 2×+ traffic cut. F1 touches only [E,128] tensors (67 MB) and is likely too small to matter.
- **`index_add_` is atomic on XPU**, which is one reason E0 showed non-bitwise repeats (max 5.8e-11 eV). A sorted segment reduction in F5 is deterministic. That is a side benefit, not a goal.

### 5.2 Rules specific to N5

- **Stage entry gate (N5.0).** N5 starts only after S0–S4 pass (§4.3, S5 reported) and the N1 and N2 verdicts are recorded (closed means a verdict is recorded, not necessarily a win). If S1 or S2 fail unfixably, N5 is dropped (§4.3).
- **No size filter.** Every block gets its stage, whatever its measured share. The N5.1 profile orders the blocks and tells us how small a gain to expect; it never removes one. A block too small to detect is closed flat with its profile number.
- **Micro verdict.** Fused (forward + backward, all three passes counted) against the unfused torch sequence on the real shapes, median of ≥30 repeats per job, ≥3 jobs on different nodes. Win, flat or loss by the H.0 rule (same sign in all jobs and a median gain above the larger of 3 × pooled within-job MAD and the between-job range). There is no ratio threshold. Only a win moves on to the in-model stages.
- **Tuning round.** Before a block is closed flat or loss, try at least: three work-group shapes, sub-group 16 and 32, with and without local memory, and vendor library against custom code for the GEMM parts. Record each variant's number.
- **Correctness bars.** Per block, FP64: forward relative max error ≤ 1e-12 against stock eager; gradients ≤ 1e-12 relative; `torch.autograd.gradcheck` (FP64, small shapes) passes. In the model the P.1 bars apply unchanged (step-0 |ΔPE| ≤ 1e-8 eV, force 1e-9 eV/Å, virial 1e-7 eV). A miss is investigated, never loosened.
- **Backward in two steps.** (a) Scaffold: the op's backward calls `autograd.grad` on the torch reference. This proves plumbing and checkpoint recompute only, and **no timing is claimed from it** (it does extra work). (b) Hand-written SYCL backward, which is the only timed variant. So F6 is folded into each block stage and is not a separate stage.
- **No architecture change.** Fused ops compute the same math (the §4 and `uma-architecture-immutable` rules). Weights are passed to the ops as tensors taken from the exported module, so MOLE-merged weights are used as they are.
- **One artifact per flag set.** The exporter takes `UMA_NITRO_FUSED=<comma list of f1,f2,f3,f4,f5>` and `UMA_NITRO_FUSED_LAYERS=<list or all>` at export time. Both are written into the manifest and `meta.json`. The engine refuses an artifact whose op list is not linked.
- **Branches and tags.** `nitro/n5-proto` (S0–S5), `nitro/n5-infra`, `nitro/n5-f2`, `nitro/n5-f1`, `nitro/n5-f4`, `nitro/n5-f5`, `nitro/n5-f3`, `nitro/n5-stack`. Tags `nitro/n5-<id>/r<k>` before every `qsub`.

### 5.3 Stages

| Stage | Work | Compute | Exit criteria |
|---|---|---|---|
| **N5.0** (step 19) Entry gate | Checklist only: S0–S4 closed, S5 reported, N1 and N2 decisions recorded, H1–H6 closed. | no | All boxes ticked in this table; status line updated. |
| **N5.1** (step 20) Op-level profile (diagnostic) | Standalone script (not LAMMPS) on one tile: real chunk shapes dumped from one rank of the 14³ W=12 run, stock eager `forward_chunk` in FP64, XPU profiler, three passes per layer. Output the per-op device time, then group into the table in 5.1. Calibrate against the 659 ms row (4 layers, one chunk per rank): the sum should land within ±15%; if not, find the gap before continuing. Confirm the shapes and that weights do not require grad. | 1 node, 1 tile, `debug-scaling` | `pbs/out/nitro_n5/N5_PROFILE.md` with: shares in ms, the measured GEMM floor, and the blocks ordered by share (priority only). Nothing is filtered out. |
| **N5.2** (step 21) Fixtures and reference harness | Save one rank's chunk inputs (for NaCl and for ZIF-8) (`x_full`, `edge_index`, `wigner`, `wigner_inv_envelope`, `x_edge`) and reference outputs and gradients from stock eager FP64, in scratch (not committed). Write `pbs/nitro/n5_bench.py`: runs a block unfused vs fused, reports error, max abs gradient error, time (median, MAD, 30 repeats), traffic estimate, GB/s. | 1 node | Harness reproduces the stock result bitwise against itself; a no-op "fused = stock" arm shows ratio 1.00 ± MDE. |
| **N5.3** (step 22) Infra | Shared header for op schemas; build rule linking the N5 `.o` files into `uma_engine` (as `xccl_peer.o`, from S3); exporter flags with a Python stand-in for every op (`torch.library`) and an eager fallback; manifest and engine load check; measure export time and per-arm job time for the budget. | build, then 1 node ABBA | Export with an empty flag set produces an artifact whose traced graph equals BASE (manifest hash equal); BASE gate unchanged (step-0 PE −74142.191907631001, ABBA vs BASE within MDE); job-time budget recorded (how many arms fit in 01:00:00). |
| **N5.4** (step 23) F2 Gate | Fused sigmoid + expand + SiLU-scalar + multiply + concat. Micro: fwd, backward scaffold, then hand-written backward. In-model arms `L0` (layer 0 only) and `ALL`. | micro: 3 jobs; model: 3 jobs | Correctness bars met; tuning round done; micro verdict recorded. If the micro is a win, `L0` and `ALL` pass step-0 PE and NVT parity and their ABBA verdicts are recorded. Branch merged to `nitro/n5-stack` only for a win. Flat or loss closes the stage with its numbers. |
| **N5.5** (step 24) F4 Wigner + conv1 | Fused gather, concat, rotate, radial multiply, `fc_m0` and block GEMMs (GEMM part may stay on the vendor library; the fusion is the memory-bound prologue and epilogue). Same sub-steps as N5.4. | 3 + 3 jobs | As N5.4, plus force parity ≤ 1e-9 eV/Å. |
| **N5.6** (step 25) F5 Inverse rotation + conv2 + reduce | Includes the sorted segment reduction (N2-b if it passed) and its backward. Needs the center-sorted assertion (N2-1) and a test across a chunk boundary. | 3 + 3 jobs | As N5.5, plus NPT 10-step parity (virial ≤ 1e-7 eV). |
| **N5.7** (step 26) F1 LN+SiLU | Expected to be small (tensors are only [E,128]); attempted anyway. | 3 + 3 jobs | As N5.4. |
| **N5.8** (step 27) F3 Radial chain | Merge the radial MLPs across layers (one batched first Linear for 4 layers), fused LN+SiLU between Linears. | 3 + 3 jobs | As N5.4. |
| **N5.9** (step 28) Stack gate | All passing blocks together on `nitro/n5-stack`; ABBA vs BASE, ≥3 jobs on different nodes; NVT and NPT 10-step parity; if W=12 fails, first run the same artifact at W=1 to decide kernel or multi-tile cause. | 3 jobs + diagnostics | Verdict per the replication rule; step time and the BASE / stack ratio reported as measurements; leave-one-out and add-one-in arms for the N5 blocks; result handed to the §1 "stack" step. |
| **N5.10** (step 32) F7 side queues | Deferred. Opens only if N5.9 is closed and the user asks for it. | n/a | n/a |

Default block order is N5.4 → N5.5 → N5.6 → N5.7 → N5.8. N5.1 may reorder it by measured share; the reorder is recorded in the N5.1 closure. A flat or loss block still closes its stage (with the measured numbers), so the next stage can start.

### 5.4 Failure points specific to this plan

| # | Sev | Failure point | Mitigation |
|---|---|---|---|
| N5.a | M | FP64 GEMMs are about a third to a half of the compute row and fusion cannot remove them, so the N5 gain is bounded by the rest. | N5.1 measures the split; reported, not a gate |
| N5.b | S | Backward misses a path (gradient through `wigner`, `wigner_inv_envelope` or `x_edge` carries forces and virial). | `gradcheck`, FD spot checks, NPT parity, scaffold backward as a cross-check |
| N5.c | S | Checkpoint recompute: the op must have the Autograd key and give identical outputs in the no-grad and the grad pass. | S1 test through a checkpoint-style no-grad forward then recompute (P-6) |
| N5.d | H | Atomics or sorted reduce change the summation order, so results are not bitwise equal to BASE. | Tolerance bars above; document, do not claim bit-identity (P-7) |
| N5.e | H | Tiny 9×9 batched matrices run poorly on the vendor batched GEMM; a custom kernel needs a good work-group and sub-group shape and FP64 register budget on PVC. | Micro stage decides; bandwidth reported against the S5 FP64 baseline |
| N5.f | M | Exporter drift: the exporter reimplements `forward_chunk`; a fairchem change silently breaks parity. | Parity vs stock eager per block, pinned fairchem 2.21.0, `meta.json` records the version |
| N5.g | M | Job-time budget: each arm needs an export (27 GB artifact) plus a run, and walltime is 01:00:00. | N5.3 measures it; arms per job limited accordingly; `KEEP_ARTIFACT` only where reused |
| N5.h | M | A weight passed as a tensor argument gets copied at trace time or goes stale after an N4 fold. | Pass module parameters by reference; N4 runs before or is flagged in the manifest |
| N5.i | M | A fused op hides a bad kernel launch count or a first-call JIT (P-4). | AOT PVC; discard the first call; report launches per chunk |

### 5.5 Decisions that belong to the user

1. **Start N5 before N1 and N2?** Plan order says N1, N2, then N5. Default: keep the order (nothing opens early).
2. **Block order** is default F2, F4, F5, F1, F3 until N5.1 measures shares. Say if you want a different order.
3. (Removed in revision 8: there is no ceiling or stop rule by speed.)
