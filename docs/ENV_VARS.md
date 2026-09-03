# UMA LAMMPS pair style — environment variables

This is the authoritative catalog of every `UMA_*` (and related) environment
variable the ML-UMA package reads, at **runtime** (the C++ `pair_style uma` engine)
and at **export/CI** time (the Python exporter and gate scripts). Maintained as
part of P4.1/P6.1 (`docs/CODE_QUALITY.md`).

Status (audit 2026-08-31, E3): there is **no `UmaConfig` struct** — env vars are
read at their point of use via `getenv`/`uma_env_bool`; the `PairUMA` ctor echoes a
few config flags in one startup log line (`init_style`), but this is not a full
consolidation. A full `struct UmaConfig` is tracked as a residual (D.7 / P4.1).

A Tier-0 CI guard (`ci/tier0_guards.sh`, HARD check "env vars documented") greps
the compiled source for `getenv("UMA_*")` / `environ["UMA_*"]` and fails if any name
is **not** listed in this file. Keep this catalog in sync when adding a var.

## 1. Runtime — core config (C++ engine, read once at init)

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `UMA_CHECKPOINT` | path | (none) | Path to the scientific UMA checkpoint (`uma-s-1p2.pt`); required by the eager/GP worker and by provenance sha256. |
| `UMA_ENGINE_BUILD_GRAPH` | 0/1 | 0 | Build/trace the graph in-engine instead of loading a prebuilt artifact. |
| `UMA_AC` | off/chunk/block/full | **off** | **A10 (owner directive §G.25): activation checkpointing master, DEFAULT OFF.** `off` = no recompute anywhere (fully differentiable, NPT/virial works, lower capacity); `chunk` = per-chunk recompute (restores most capacity); `block` = chunk+block; `full` = chunk+block+edge-degree. Enabling AC trades the virial capability for memory. Subsumes the four `UMA_NO_RECOMPUTE*` (now deprecated aliases). |
| `UMA_CKPT` | 0/1 | **0** | Whole-module C++ activation checkpointing. **A10: default OFF on ALL builds** (was ON for XPU). On production artifacts this branch is dead (per-chunk ops short-circuit); use `UMA_AC` for the effective control. |
| `UMA_NO_RECOMPUTE`, `_BLOCK`, `_CHUNK`, `_EDEG` | 0/1 | 0 | **Deprecated (A10):** legacy per-level "retain activations" switches. Redundant now that recompute defaults OFF; kept for back-compat. Prefer `UMA_AC`. |
| `UMA_MN_CKPT` | 0/1 | 0 | Whole-module checkpointing on the multi-node (GP) legacy monolithic-shard path. |
| `UMA_EAGER_CKPT` | 0/1 | 0 | Eager (non-traced) checkpoint path selection. |
| `UMA_CHUNK_RETAIN_K` | int | (adaptive) | Number of edge-chunk activations to retain (memory/speed knob); the harness sweeps 3→2→1→0. |
| `UMA_SKIP_MAXNBR_CAP` | 0/1 | 0 | Skip the max-neighbors cap (fast artifacts). |
| `UMA_ALLOW_LEGACY_METADATA` | 0/1 | 0 | **P4'.1**: accept a `metadata.json` without `metadata_version>=2` (pre-Sprint-5 artifacts). Off ⇒ such artifacts are rejected. |
| `UMA_ALLOW_FAIRCHEM_MISMATCH` | 0/1 | 0 | **P5'.1**: allow export with a fairchem/torch version different from the pinned one (`trace_patch`). |
| `UMA_COMPUTE_VIRIAL` | 0/1 | **auto** | **P0'.1 step 2 / A10**: single-tile stress/virial (pos+cell autograd) → NPT. **Now AUTO-ENABLED** when a barostat (`fix npt/nph/…`) is present on a single tile with AC off (the new default), so NPT works out of the box. Set to `0` to force off; refused on the GP/DD path and when AC is explicitly on. |
| ⚠ `UMA_SKIP_FORCE_GP_REDUCE` | 0/1 | 0 | **CHANGES NUMERICS.** Skip the cross-rank force all-reduce on the GP path (`kokkos_gp_runtime.py`, debug/perf only) → forces are per-shard, NOT the full system. Do not use for production. |

## 2. Runtime — performance / debug (may be read per-step; informational)

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `UMA_MP_PERF` | 0/1 | 0 | Print per-step GP timing (graph/fwd/bwd/allreduce ms). |
| `UMA_MP_LOG_DIR` | path | (none) | Directory for per-rank GP logs. |
| `UMA_MP_NATOMS` | int | (none) | Hint for the n-specific monolithic shard filename. |
| `UMA_SKIP_PRE_BWD_BARRIER` | 0/1 | 0 | Skip the deterministic pre-backward barrier (perf experiment; may desync collectives). |
| `UMA_ALLREDUCE_WITH_GRAD_BWD` | 0/1 | 0 | Fuse the force all-reduce into the backward. |
| `UMA_CUDA_GRAPH` | 0/1 | 0 | CUDA-graph capture (CUDA backend). |
| `UMA_CUDA_LAUNCH_BLOCKING` | 0/1 | 0 | Force synchronous CUDA launches (debug). |
| `UMA_NL_ALLPAIRS` | 0/1 | 0 | Force the O(N²) all-pairs neighbor list (A/B validation vs the cell list). |
| `UMA_DEBUG_PARTITION_CHECK` | 0/1 | 0 | Extra GP node-partition coverage assertions. |
| `UMA_PEER_TRANSPORT` | string | (auto) | Force a peer transport (xccl/nccl/shm/…); normally auto-detected. |
| `UMA_MP_VERBOSE` | 0/1 | 0 | Per-step predict tracing on the Python-worker GP path (`graph_parallel.cpp`). |
| `UMA_MP_PAYLOAD_SHM` | path | (auto) | GP worker payload shared-memory path (set internally by `libtorch_mp.cpp` for the child). |
| `UMA_MP_PAYLOAD_BYTES` | int | (auto) | GP worker payload buffer size in bytes (set internally for the child). |
| `UMA_STRUCTURE_NATOMS` | int | (none) | Expected system atom count hint (`libtorch_mp.cpp` diagnostic). |
| `UMA_CACHE_DIR` | path | (derived) | Directory of the flat `*.pt` checkpoints (`checkpoints.py`); else derived from `UMA_CHECKPOINT`/repo-sibling. |
| `PYTHON` | path | (auto) | Fallback Python interpreter for the GP worker if `UMA_PYTHON` is unset (`graph_parallel.cpp`). |
| `CUDA_LAUNCH_BLOCKING` / `CUDA_VISIBLE_DEVICES` | — | — | Standard CUDA env (read only on the CUDA backend for diagnostics / device binding). |

**Test-worker only** (`tests/uma_libtorch_mp_worker.cpp`, not in the shipped
library; ⚠ = changes numerics): `UMA_GRAD_ENERGY_SCALE` (energy scale before the
backward), `UMA_EDGE_PAD_E` (override traced edge-pad cap), `UMA_CUDA_GRAPH_WARMUP`
(CUDA-graph warmup iters).

## 3. Runtime — worker / backend selection (C++)

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `UMA_GPUS_PER_NODE` | int | 4 | GPUs per node for local-rank device binding (GP). |
| `UMA_HEN_ROOT` | path | (repo `../hen`) | **G15/S7**: root of the `hen` shim/patches checkout used by the export/worker layer. Resolved via `uma_hen.py` (env → `hen` sibling of the repo → loud `FileNotFoundError`); replaced 4 hardcoded absolute-path sites. |
| `UMA_PYTHON` / `UMA_PYTHON_GP_WORKER` | path | (auto) | Python interpreter / GP worker script for the eager FairChem devices>1 path. |
| `UMA_GP_WORKER` / `UMA_LIBTORCH_MP_WORKER` | path | (auto) | Worker binary/script overrides. |
| `UMA_ALLOW_RAY_GP` / `UMA_FORBID_RAY_GP` | 0/1 | 0 | Ray-based GP fallback opt-in/opt-out (**scheduled for removal, P3.1**). |

## 4. Runtime — domain decomposition (DD / Phase 3, **deferred**)

These belong to the deferred multi-node DD path. Listed for completeness; not used
by the validated GP-over-MPI production path.

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `UMA_DD` | 0/1 | 0 | Enable spatial domain decomposition (k=4 halo). |
| `UMA_DD_HALO` | 0/1 | 0 | Export/expect the k=4 per-layer halo artifact (top module returns per-atom node energy). |
| `UMA_DD_EDGE_CAP` | int | 0 | Override the fixed traced edge cap (must match the artifact; a mismatch errors). |
| `UMA_DD_DEBUG` / `UMA_DD_NO_HALO` / `UMA_DD_HALO_TEST` | 0/1 | 0 | DD debug/self-test toggles. |

## 5. Promoted to `pair_style` keywords (prefer the keyword)

Per P4.1, the following are (being) promoted to `pair_style uma` / `pair_coeff`
keywords; the env var remains as an override:

- `precision {mixed|double}` — was implicit; now a `pair_style` keyword.
- `devices N` — GP device count.
- (planned) `UMA_DD`, `UMA_ENGINE_BUILD_GRAPH`, `UMA_DD_EDGE_CAP`,
  `UMA_GPUS_PER_NODE` as keywords.

## 6. Export-time (Python exporter, `export_blocks_xpu.py` et al.)

| Variable | Default | Meaning |
|---|---|---|
| `UMA_CHECKPOINT` | (req) | Source checkpoint to export/trace. |
| `UMA_TASK` | omat | UMA task head to export/trace (e.g. omat, omol, …). |
| `EDGE_AC_CHUNK` | 16384 | Edge activation-checkpoint chunk size (baked into the trace). |
| `RECONSTRUCT` | 1 | Run the traced==monolithic reconstruct check (**P1.4** now gates the exit code). |
| `SHAPE_GENERIC` | 0 | Symbolic-dim Wigner trace. |
| `WIGNER_CHUNK` / `WIGNER_CHUNK_OFF` | — | Wigner chunking (the N≥10 FP64 fix). |
| `EXPORT_WORLD` / `EXPORT_RANK` / `EXPORT_ONLY_RANK` | 1/0 | GP shard export coordinates. |
| `MERGE_MOLE` | 0 | Merge the MoLE expert weights at export. |
| `TRACE_DEV` / `TRACE_N` / `MOVE_TRACED_TO_XPU` | — | Trace device / example size / post-trace device move. |
| `KEEP_TRACE_DIR` / `OUT` | — | Output artifact directory. |
| `UMA_EDGE_PAD` / `UMA_EXPORT_CELL_LIST` / `UMA_NO_FREEZE` | — | Padding / cell-list / freeze toggles. |
| `UMA_ALLOW_FAIRCHEM_MISMATCH` | 0 | Bypass the P5'.1 version assertion. |
| `UMA_ALLOW_MISSING_PATCHES` | 0 | **P5'.5**: export even if a correctness-critical monkeypatch (wigner-chunk N≥10 fix, xpu-device) fails to apply. NOT recommended — the artifact may be numerically wrong. |
| `UMA_COMPUTE_VIRIAL` | 0 | **P0'.1 step 2**: enable the single-tile strain-autograd virial/stress. Refused on XPU (segfaults); works on CUDA. |

## 7. CI / gate scripts

| Variable | Default | Meaning |
|---|---|---|
| `E_TOL_PER_ATOM_MEV` | 1e-3 | Per-atom energy gate (meV/atom) — **use `scripts/uma_gates.py`**. |
| `F_TOL` | 1e-5 | Per-atom force gate (eV/Å). |
| `AG_FD_TOL` | 1e-5 | AG=FD force gate (eV/Å). |
| `MIN_SAMPLE` | 100 | Minimum samples before a PASS is trustworthy. |
| `FD_EPS` | 1e-4 | Central-difference step (Å). |
| `ALLOW_SKIP` | 0 | Gate-1: explicitly permit skipping the ASE oracle cross-check. |
| `UMA_TIER0_STRICT` | 0 | Treat Tier-0 REPORT findings (cleanup debt) as hard failures. |

All gate tolerances are defined once in `scripts/uma_gates.py` (P1.5). Do not
hard-code copies in comparators or `.pbs` scripts.

## 8. Known limitations (not env vars)

- **✅ `MPI_COMM_WORLD` hardcoding FIXED** (P7.1, audit rev 26 §G.18.6 / A8): the
  XCCL KVS rendezvous now bootstraps on the LAMMPS communicator. `pair_uma.cpp`
  passes `MPI_Comm_c2f(world)` to `MpiPeerPredictor::create`, which threads it
  (`comm_f`) through `init_xccl_external` to `XcclPeer::create`, where
  `MPI_Comm_f2c(comm_f)` recovers the real comm for the address `MPI_Bcast`
  (`comm_f=0` still means `MPI_COMM_WORLD` for backward compatibility). The
  handle is passed as an `int` (Fortran MPI handle) so the GCC-safe engine
  headers stay MPI-type-free. The multi-node GP path now bootstraps on the same
  communicator LAMMPS gathers on, so `-partition` / library sub-communicator /
  MDI use the correct comm instead of silently gathering across the wrong world.
- **✅ `atom->natoms` int32 overflow now fails loudly** (P7.2, audit rev 26 §G.18.6
  / A8): the GP gather still uses int counts (full 64-bit GP is deferred), but
  `pair_uma.cpp` now guards `atom->natoms > INT_MAX` with an explicit
  `error->all` naming the limit, instead of silently narrowing >2^31 atoms into a
  garbage int. Converts silent-wrong-physics into a clear, actionable abort.
- **NPT/stress now works OUT OF THE BOX on a single tile** (A10, §G.25): activation
  checkpointing defaults OFF (`UMA_AC=off`), so the fully-differentiable virial path
  is the default; a barostat on a single tile auto-enables `UMA_COMPUTE_VIRIAL`. The
  GP/DD paths still refuse a barostat (no virial there), and enabling AC
  (`UMA_AC=chunk/…`) makes the virial unavailable (strain grad can't thread the
  checkpoint) — refused with a clear message. Capacity is lower by default; the A11
  pre-flight check warns and names `UMA_AC=chunk` when a run may OOM.
- **DD multi-species energies are approximate** (P0'.2): the traced MoLE mixture uses
  each rank's local composition; a one-time warning is emitted (R4). DD is deferred.
