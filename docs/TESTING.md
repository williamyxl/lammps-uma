# Testing the UMA LAMMPS pair style

Test pyramid for the ML-UMA package (see `docs/CODE_QUALITY.md` PART C.3 / D.5).
Runs bottom-up; each tier is cheaper and faster than the one above it.

## Tier 0 — static guards (login node, no deps, ~5 s)

```bash
bash ci/tier0_guards.sh          # HARD checks + Sprint-6 cleanup REPORT
UMA_TIER0_STRICT=1 bash ci/tier0_guards.sh   # treat REPORT debt as failures
```
HARD: every tracked Python file parses; `uma_gates.py` imports and exposes the gate
table; the mandatory gate `n16_ase_parity.pbs` has `set -euo pipefail`.
REPORT: counts remaining cleanup debt (foreign machine paths; `.pbs` missing the
`set -euo pipefail` preamble; library modules importing `spike_*`/attic). These flip
to HARD under `UMA_TIER0_STRICT=1` once Sprint 6 cleanup finishes.

## Tier 1 — hermetic unit tests (login node, python3+numpy, ~40 s)

```bash
bash ci/ci_local.sh              # Tier 0 + all Tier 1 (plain python3 runners)
bash ci/ci_local.sh --pytest     # also run via pytest if available
python3 ci/tests/test_gate_arithmetic.py     # a single file
```
No torch/fairchem/XPU required. Coverage:
- `test_metadata_contract.py` — metadata.json contract; real-JSON-parser behavior
  (P4′.2) vs the old substring scanner; version-gate / legacy policy (P4′.1).
- `test_edge_padding_partition.py` — `edge_pad_cap` and `node_partition` invariants
  (P2.1 / P0′.3 / P5′.4-GP contracts).
- `test_neighbor_image_repeats.py` — interplanar-spacing image bound (P0.5).
- `test_gate_arithmetic.py` — `uma_gates` + fail-closed parity decision (P1.1/2/5).

Tests needing torch/fairchem are marked `needs_torch` / `needs_fairchem` and
auto-skip on the base env (run them under the fxpu conda env).

## Tier 1 (C++) — CTest (built with the standalone engine)

```bash
# in the engine build dir:
ctest --output-on-failure
```
`graph_shard_smoke` (GP node-partition coverage) runs on every backend; the NCCL
smoke tests register on the non-XPU build. Registered via `enable_testing()` /
`add_test()` in `uma-engine/CMakeLists.txt` (P1.6). Note the engine target is
`EXCLUDE_FROM_ALL` inside the LAMMPS build, so build it explicitly for CTest.

## Tier 2 — CPU engine + toy artifact (follow-on T2)

Planned: a committed toy artifact + CPU trace path gating opt2/opt4/retain-K/padding
equality, and `pair-uma` in upstream `unittest/force-styles/test_pair_style.cpp`
(`mol-pair-uma.yaml`) for a CPU energy/force/**virial** regression. Blocked on a CPU
LibTorch toolchain. Until then these claims are gated by the XPU G4 suite (below).

## Tier 3 — XPU parity + performance (PBS allocation)

The mandatory per-round gate and the full per-sprint regression (goal G4).

```bash
qsub scripts/n16_ase_parity.pbs      # tripwire: N=16 W=1 + N=32 W=12 (~10 min)
qsub scripts/final_perf_parity.pbs   # full G4 matrix: N=16 W=1,2,4,6,8,12 + N=32 W=12
qsub scripts/phase1_xpu_force_agfd.pbs   # AG=FD (autograd vs finite difference)
```
PBS policy: submit to `debug` or `debug-scaling`, always `walltime=01:00:00`.
Pass criteria (from `scripts/uma_gates.py`): step-0 PE bit-identical to the
validated record, per-atom max|dF| ≤ 1e-5 eV/Å (FP64 floor ~1e-13), cos = 1.0,
AG=FD ≤ 1e-5. Results are appended to `REPORT_2path_nvt_comparison.md` §14 each
sprint.

## What must stay green before any `src/ML-UMA/` change lands

1. `bash ci/ci_local.sh` (Tier 0 + Tier 1).
2. The mandatory ASE parity tripwire (Tier 3) at sprint close.
3. The full G4 suite at sprint close, with step-0 PE bit-identical (goal G3).
