# TEST_PROTOCOL — multi-GPU uma/kk tests

**Stamp:** 2026-08-08 ~20:25 CDT · Branch `uma-kokkos-mlip`  
**Product path:** `pair_style uma/kk precision double devices N` with
`lmp -k on g N -sf kk`, **1 MPI rank**, FP64 only.

## Systems under test

| ID | Geometry | Role |
|----|----------|------|
| **NaCl6** | `structures/nacl6_rattle_fixed.extxyz` (1728 atoms, rattled, frozen) + NVT data `structures/nacl6_nvt_300K_atomic_metal.data` (same positions + MB velocities @ 300 K, seed=0) | Parity + **NVT 300 K, 10 steps** on ASE / FC / uma (default future gates) |
| **H2O888** | `water888/water_nvt_300K_atomic_metal.data` (648 atoms, Velocities included; type1=O, type2=H) | **NVT 300 K**, campaign often **100 steps**, smoke **10**; flexible water |

This file is the **single source of truth** for gates and timing. Prefer it over
scattered notes in `README.md`, `write_progress.md`, or historical `perf_*.slurm`
comments when they disagree.

---

## 1. Workflow (order)

### 1.1 NaCl6 — default path gates (SP E+F + NVT @ 300 K / 10 steps)

Same layout as water888 `run_path_*` (first-frame E+F, then timed NVT):

```text
scripts:  multi_gpu_nacl6/run_path_{ase,fc,uma}.slurm
data:     structures/nacl6_nvt_300K_atomic_metal.data
default:  NSTEPS=10  T=300 K  dt=0.001  tdamp=0.1
metrics:  ASE/FC → nvt_ms_per_step; uma → nvt_pair_ms_per_step (+ first-frame E+F)
```

```bash
cd .../examples/multi_gpu_nacl6
sbatch --gpus-per-node=2 --export=ALL,NGPUS=2,FAIRCHEM_WORKERS=2,NSTEPS=10 run_path_ase.slurm
sbatch --gpus-per-node=2 --export=ALL,NGPUS=2,FAIRCHEM_WORKERS=2,NSTEPS=10 run_path_fc.slurm
sbatch --gpus-per-node=2 --export=ALL,NGPUS=2,UMA_DEVICES=2,NSTEPS=10,RECOMPILE=1 run_path_uma.slurm
```

Campaign submitters (`submit_settings_matrix.sh`, `submit_matching_ase_fc_bars.sh`)
**default to these path scripts with `NSTEPS=10`**. Do not use SP-only
`run_ngpuN_*.slurm` for new speed bars unless explicitly debugging SP.

Optional smoke input: [`in.nacl6_nvt`](in.nacl6_nvt) (`run 10`). Regenerate velocities only
via `python prep_nacl6_nvt_data.py --force` (never re-rattle positions).

### 1.2 NaCl6 — legacy SP / devices=1 parity (optional)

```text
0. Export MP artifacts          export_mp_w{2,4}*.slurm
1. Engine smoke E+F             smoke_mp_w2.slurm → smoke_mp_w4.slurm
2. LAMMPS product E+F           lammps_smoke.slurm / run_ngpuN_uma_double.slurm (SP + NVE)
```

### 1.3 H2O888 NVT (flexible water)

```text
untimed:  stage `water_nvt_300K_atomic_metal.data` + `in.h2o888_nvt`
timed:    run_path_{ase,fc,uma}.slurm  (campaign NSTEPS=100; smoke run 10)
```

Script: [`water888/run_h2o888_nvt.slurm`](../water888/run_h2o888_nvt.slurm)  
Input: [`water888/in.h2o888_nvt`](../water888/in.h2o888_nvt) — NVT 300 K, `run 10`, **no** `fix shake` / rigid / bond constraints.

```bash
cd .../examples/water888
sbatch --export=ALL,RECOMPILE=1,NGPUS=1,UMA_DEVICES=1 run_h2o888_nvt.slurm
# multi-GPU (needs MP shards for n=648 if devices>1):
sbatch --gres=gpu:4 --export=ALL,NGPUS=4,UMA_DEVICES=4,RECOMPILE=1 run_h2o888_nvt.slurm
```

Reported metric for H2O888 path gates: **`nvt_pair_ms_per_step`** (uma) /
`nvt_ms_per_step` (ASE/FC). Smoke launch-wall still uses `slurm_wall_s`.

Constraints (both systems):

- FP64 only (`uma-s-1p2-omat-f64` / `*-f64-fast`); mixed disabled.
- No Ray / Python GP as product (`UMA_FORBID_RAY_GP=1`).
- `RECOMPILE=1` on engine/LAMMPS jobs unless binary already matches source.
- NaCl6: land **devices=2** E+F green before opening **devices=4**.
- Partition `gpuA100x4`, `--ntasks=1`, `--gpus-per-node=N` matching `NGPUS`.
- H2O888: **flexible** molecules only — never add `fix shake`, `fix rigid`, or
  similar constraint fixes.
- **Both systems:** future ASE/FC/uma speed tests include **NVT @ 300 K** (NaCl
  default **10** steps; water campaign often **100**).

---

## 2. Accuracy gates (hard)

Primary oracle for abort gates: **`results/ngpu1` uma_double** (devices=1).

| Metric | Threshold | Code |
|--------|-----------|------|
| \|ΔE\| vs d1 | ≤ **1×10⁻⁸** eV | `parity_gates.py` / SLURM gate Python |
| max\|ΔF\| vs d1 (component) | ≤ **1×10⁻⁶** eV/Å | same |
| force cosine vs d1 | ≥ **1 − 1×10⁻¹²** | `parity_gates.py` |

ASE FP64@1 (`gp_round/oracle_ase_fp64_w1.json`) is the **scientific oracle**
(report band: \|ΔE\| ~1×10⁻¹⁰, max\|ΔF\| ~5×10⁻⁷). Product SLURM jobs abort on
**d1** thresholds above, not on ASE.

Engine smoke may use a slightly tighter E tol (`ENERGY_TOL=1e-9`); product gate stays 1e-8.

---

## 3. Performance gates

### 3.1 Self-scale (hard)

```text
ms(devices=2) < ms(devices=1)
ms(devices=4) < ms(devices=2)
```

### 3.2 Campaign hard bar (hard)

Reported ms/eval **≤ ASE FairChem FP64 and ≤ FairChem FC LAMMPS** at the same
GPU count (NaCl6 1728, FP64):

| Ref | @1 | @2 | @4 |
|-----|----:|----:|----:|
| ASE | 396.5 | 193.9 | 115.2 |
| FC  | 345.5 | 193.2 | 118.0 |

Soft targets (≤200 @2, ≤150 @4, etc.) do **not** close the campaign alone.

### 3.3 Beat-prior (perf A/B)

Each `perf_p*.slurm` may require beating the prior phase @2/@4. Exit codes
(typical): E+F fail→2, self-scale fail→3, beat-prior fail→4.

---

## 4. Timing measurement — **SLURM wraps the launch command only**

### Policy (authoritative)

Reported `ms_per_eval` comes from a **bash wall around the single command that
actually launches** the engine — not around `python run_multigpu.py`, rebuild,
input writing, force parsing, or gate Python.

```text
ms_per_eval = 1000 * slurm_wall_s / N_TIMING
```

| Path | Timed command (one line) | Not timed |
|------|--------------------------|-----------|
| **uma/kk** | `"${LMP_UMA}" -k on g ${NGPUS} -sf kk -in in.nve -log log.nve` | module/conda, rebuild, write `in.*`/`data.lmp`, SP dump run, E/F parse, worker log tails, gates |
| **ASE FairChem** | one-line wrapper / process that only runs the ASE/FairChem eval for `N_TIMING` | `load_predict_unit` prep if split out, report writers |
| **FC LAMMPS** | one-line wrapper that **only** launches FairChem LAMMPS on the prepared input (see §4.2) | write `data.lmp`/`in.fc`, `load_predict_unit(...)`, force extract / Ray teardown, gates |

Stamp with `stamp_slurm_timing.py` (`timing_source: slurm_wall`).  
Pair / `uma64` / `PERF_*` stay **debug only**.

Default `N_TIMING`: **5** (perf) · **3** (smoke).

**Historical:** P3c PASS `20940474` used pair/`uma64` ms. Rebench under this
launch-only wall before updating RESULTS canonical numbers.

### 4.1 Target SLURM shape (uma — NaCl6 perf / H2O888 NVT)

```bash
# untimed prep: modules, env, write data + input (or prep_data.py)
t0=$(date +%s.%N)
"${LMP_UMA}" -k on g "${NGPUS}" -sf kk -var UMA_DEVICES "${UMA_DEVICES}" \
  -in in.h2o888_nvt -log log.nvt          # H2O888 NVT example
# or: ... -in in.nve -log log.nve         # NaCl6 timed NVE / eval deck
t1=$(date +%s.%N)
wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.9f", b-a }')
# NaCl6 parity: ms_per_eval = 1000 * wall / N_TIMING
# H2O888 NVT:   report wall_s (total) and ms_per_step = 1000 * wall / 10
```

### 4.2 Target SLURM shape (FC LAMMPS)

Same rule: **time only the FC LAMMPS launch**, not predictor load or the parity harness.

Today the launch is buried in Python as:

```python
lmp = run_lammps_with_fairchem(predictor, str(inp), "omat")
# fairchem.lammps.lammps_fc — constructs lammps(...) and runs in.fc
```

Protocol target (SLURM owns the clock):

```bash
# untimed: write data.lmp + in.fc; load_predict_unit / export predictor handle if needed
t0=$(date +%s.%N)
# one-liner that ONLY starts FC LAMMPS on that input, e.g.:
#   python -m fairchem.lammps.lammps_fc …   OR a thin wrap_fc_lammps.sh
./wrap_fc_lammps.sh "${IN_FC}"   # must exec/run only run_lammps_with_fairchem (or equiv CLI)
t1=$(date +%s.%N)
wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.9f", b-a }')
# ms_per_eval = 1000 * wall / N_TIMING
```

`in.fc` (or the timed wrapper’s NVE script) must contain the `N_TIMING` evals
(`run 0` repeated or equivalent). Do **not** include `load_predict_unit` inside
the timed region. Do **not** time the whole `run_multigpu.py` / `run_fairchem_lammps()`
function (that currently folds load + launch + extract + extra `run 0` loops).

ASE: same idea — `t0`/`t1` only around the ASE launch line / one-line wrapper.

### 4.3 Gap vs today’s harness

`lammps_smoke.slurm` / `perf_p3c.slurm` / `_run_common.sh` currently time
**all of** `python run_multigpu.py`. Inside that:

| Path | Real launch (not SLURM-visible today) |
|------|----------------------------------------|
| uma | `_run_lmp([lmp, "-k", "on", "g", N, "-sf", "kk", "-in", "in.nve", …])` |
| FC  | `run_lammps_with_fairchem(predictor, in.fc, "omat")` |

**Implication:** split prep from launch so the `.slurm` can wrap a **one-line**
uma `lmp …` and a **one-line** FC launch wrapper. Do not keep timing the full
Python harness.

---

## 5. Reference: where the launch lives today (not yet SLURM-timed)

### 5.1 uma — real command (inside Python)

```text
lmp -k on g ${NGPUS} -sf kk -in in.nve -log log.nve
```

Built by `uma_kk_argv` / run via `_run_lmp` in `run_multigpu.py` (`run_uma_kk`).
`in.nve` ends with `run 0` then `run ${N_TIMING}`.

### 5.2 FC LAMMPS — real launch (inside Python)

```python
# run_multigpu.run_fairchem_lammps — AFTER load_predict_unit (untimed under this protocol)
lmp = run_lammps_with_fairchem(predictor, str(inp), "omat")
```

That call is the FC LAMMPS launch (`fairchem.lammps.lammps_fc`: `lammps(...)` +
input/`fix external` + run cmds). Extra `lmp.command("run 0")` loops in
`run_fairchem_lammps` are **not** a separate shell command today; fold `N_TIMING`
into the timed input/wrapper so SLURM still wraps **one** launch line.

FairChem also exposes a Hydra CLI entry (`python -m fairchem.lammps.lammps_fc` /
`lammps_fc.main`) — a thin `wrap_fc_lammps.sh` around that (or around
`run_lammps_with_fairchem` only) is the preferred SLURM-visible one-liner.

### 5.3 What SLURM times today (wrong for this protocol)

```bash
# _run_common.sh / lammps_smoke / perf_p3c — TOO WIDE
t0=$(date +%s.%N)
python run_multigpu.py    # includes prep + SP + NVE + teardown
t1=$(date +%s.%N)
```

### 5.4 Desired SLURM timing (protocol)

```bash
# uma NaCl6
t0=$(date +%s.%N)
"${LMP_UMA}" -k on g "${NGPUS}" -sf kk -in in.nve -log log.nve
t1=$(date +%s.%N)

# uma H2O888 NVT (already implemented in water888/run_h2o888_nvt.slurm)
t0=$(date +%s.%N)
"${LMP_UMA}" -k on g "${NGPUS}" -sf kk -var UMA_DEVICES "${UMA_DEVICES}" \
  -in in.h2o888_nvt -log log.nvt
t1=$(date +%s.%N)

# FC — only the FC LAMMPS launch wrapper (predictor already loaded / passed in)
t0=$(date +%s.%N)
./wrap_fc_lammps.sh "${IN_FC}"
t1=$(date +%s.%N)
```

### 5.5 H2O888 reference paths

| Path | Role |
|------|------|
| `examples/water888/water_nvt_300K_atomic_metal.data` | Structure + Velocities (648 atoms, O/H) |
| `examples/water888/prep_data.py` | Optional legacy extxyz→data (NVT job uses `.data` directly) |
| `examples/water888/in.h2o888_nvt` | NVT 300 K, 10 steps, flexible |
| `examples/water888/run_h2o888_nvt.slurm` | Thin SLURM; times `lmp` only |
| `examples/water888/work_nvt_<job>/timing_slurm.json` | `slurm_wall_s`, `ms_total`, `ms_per_step` |

---

## 6. Artifacts

| Path | Role |
|------|------|
| `results/ngpu{1,2,4}/parity.json` | Energies + stamped `ms_per_eval` |
| `results/.../timing_slurm.json` | Wall stamp |
| `results/ngpu1/` + `forces.npz` | devices=1 baseline for E+F |
| `results/gp_round/oracle_ase_fp64_w1.json` | ASE FP64 oracle |
| `agent_stamps/cpp_libtorch/perf/gate_*.json` | Per-devices gate rows |
| `agent_stamps/cpp_libtorch/perf/summary_*.json` | Job summary |
| `results/RESULTS.md` | Canonical human report |

---

## 7. Checklist

### NaCl6 (parity / perf)

1. Rebuild if needed (`RECOMPILE=1`).
2. Run SLURM script; confirm launch-only timing stamp.
3. E+F green vs `ngpu1` (thresholds §2).
4. Record `ms_per_eval` from SLURM stamp (not `uma64`).
5. For full campaign: devices 1, 2, 4 → self-scale + ≤ASE/FC.
6. Refresh RESULTS/SUMMARY only after gates pass under this timing policy.

### H2O888 NVT

1. Rebuild if needed (`RECOMPILE=1`).
2. `sbatch …/water888/run_h2o888_nvt.slurm` (default `NGPUS=1`).
3. Confirm `timing_source=slurm_lmp_launch` and `fix_shake=false` in `timing_slurm.json`.
4. Job completes `run 10` without constraint fixes; record `slurm_wall_s` / `ms_total`.
