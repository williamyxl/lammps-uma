# CODE REVIEW — multi_gpu_nacl6

**Reviewer:** REVIEW agent (Cursor Grok 4.5 High)  
**Reviewed at:** 2026-08-06T18:10:00Z (approx.)  
**Write stamp:** `.write_agent_done.json` present (`completed_at=2026-08-06T18:05:20Z`)  
**C++ changed:** no (`rebuild_required=false`)

## Verdict: **approve-with-nits**

Core recipes match the campaign contract: frozen NaCl6 geometry, ASE/FC `workers=NGPUS` with FP64 `InferenceSettings` (not turbo), uma/kk **single MPI rank + Kokkos `-k on g N`** (no `mpirun` / domain decomp), honest single-device LibTorch caveat, outputs wired for all 4 paths × ngpu. One **major** gap: Ray/ParallelMLIP teardown is missing for `workers>1`, so checklist VRAM cleanup is incomplete for the multi-GPU case. Fix before relying on NGPUS=2/4 sequential ASE→FC→uma/kk runs.

---

## Checklist

| # | Item | Status |
|---|------|--------|
| 1 | Fixed geometry only (`nacl6_rattle_fixed.extxyz` / npz fallback); no re-rattle | **Pass** |
| 2 | ASE: `workers=NGPUS` / ParallelMLIPPredictUnit; FP64 settings (not turbo) | **Pass** |
| 3 | FairChem FC: same `workers=N`; FP64 policy + cell FP32 caveat noted | **Pass** |
| 4 | uma/kk: Kokkos same-node `-k on g N`, single MPI rank; no mpirun; correctness vs 1-GPU documented | **Pass** |
| 5 | VRAM cleanup between paths | **Partial** (see Major #1) |
| 6 | Outputs: timing, energy, per-atom forces for 4 paths × ngpu | **Pass** (schema ready) |
| 7 | C++ changes / rebuild notes | **Pass** (none; docs correct) |
| 8 | Security / secrets | **Pass** |
| 9 | Style: minimal, matches `delta_parity` | **Pass** |

Clarification applied: uma/kk multi-GPU must be Kokkos same-node `-k on g N` with `--ntasks=1`. Write agent did **not** use `mpirun` / domain decomp.

---

## Findings

### Major

#### M1. No Ray / ParallelMLIPPredictUnit teardown between paths (`workers>1`)

**Where:** `run_multigpu.py` `run_ase` / `run_fairchem_lammps` / `release_cuda`

**Issue:** For `NGPUS∈{2,4}`, `load_predict_unit(..., workers=NGPUS)` builds `ParallelMLIPPredictUnit`, which `ray.init`s, creates GPU placement groups, and holds remote workers. FairChem’s class has **no** `shutdown`/`__del__`. Suite only does `del predictor` + `torch.cuda.empty_cache()`, which does **not** reliably free Ray actor VRAM or remove placement groups.

Sequential ASE → FC (second ParallelMLIP) → uma/kk subprocess can therefore OOM, hang on PG scheduling, or leave GPUs occupied after ASE.

`delta_parity` is safe because it uses `workers=1` (no Ray). This suite needs explicit teardown for the multi-GPU path.

**Suggested patch** (after ASE and after FC, before next path):

```python
# run_multigpu.py — add helper near release_cuda

def teardown_predict_unit(predictor, tag: str = "") -> None:
    """Drop predictor and shut down Ray so the next path can claim GPUs."""
    try:
        del predictor
    except Exception:
        pass
    gc.collect()
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception as exc:
        print(f"ray.shutdown warning{f' ({tag})' if tag else ''}: {exc}", flush=True)
    release_cuda(tag)
```

In `run_ase` / `run_fairchem_lammps`, replace bare `del …; release_cuda(...)` with `teardown_predict_unit(predictor, "ASE"|"FC")` (and drop redundant `del calc` ordering carefully so calculator releases the predictor ref first).

Also call `teardown_predict_unit` / `ray.shutdown()` once more before `run_uma_kk` if any FairChem path ran with `workers>1`.

---

### Nits

#### N1. `UMA_KK_LAUNCH` is documentation-only

**Where:** `_run_common.sh:37`, `run_multigpu.py:558–559`

SLURM exports `UMA_KK_LAUNCH=lmp -k on g ${NGPUS} -sf kk`, but `run_multigpu.py` only **logs** it and always launches via `uma_kk_argv()` / direct `subprocess.run`. Harmless (recipe matches), but the env var is a false control surface.

**Suggestion:** Either parse/honor `UMA_KK_LAUNCH` for the uma/kk argv, or drop the export and say launch is hard-coded to `uma_kk_argv` / `launch_uma_kk.sh`.

#### N2. `launch_uma_kk.sh` unused by the harness

**Where:** `launch_uma_kk.sh` vs `run_multigpu.py:uma_kk_argv`

Helper matches the correct recipe and is documented in README; `run_multigpu.py` duplicates the argv. Fine for a standalone recipe script; optional nit to `exec` the helper for one source of truth.

#### N3. `collect_results.py` meta key mismatch

**Where:** `collect_results.py:121` (`parity.get("gpu")`) vs `run_multigpu.py:698` (`"gpu_name"`)

SUMMARY meta will show `gpu: null`. Cosmetic.

**Suggestion:**

```python
"gpu": parity.get("gpu_name") or parity.get("gpu"),
```

#### N4. No guard that visible GPUs ≥ `NGPUS`

**Where:** `run_multigpu.py:main` after device_count log

If SLURM/`CUDA_VISIBLE_DEVICES` is wrong, ASE/FC Ray PG creation fails late. Early fail would be clearer:

```python
if n_visible < ngpus:
    raise SystemExit(
        f"torch.cuda.device_count()={n_visible} < NGPUS={ngpus}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
    )
```

#### N5. uma/kk LibTorch remains single-device (documented — tracking, not a code defect)

Write agent correctly keeps single-rank guard and documents that `-k on g N` does not shard the traced UMA forward. Energy/force parity vs 1-GPU is the right correctness bar; do not expect MLIP wall-time scaling until engine multi-device work. No change required for this campaign beyond keeping the honesty in reports.

---

## What looks good

- **Geometry:** `load_geometry.py` loads `delta_parity/structures/nacl6_rattle_fixed.extxyz` (npz fallback), asserts 1728 atoms, never rattles.
- **ASE/FC FP64:** `inference_settings_with_dtype("float64")` via engine `common.py` (export/default lineage, **not** turbo string); FC notes cell FP32.
- **uma/kk launch:** `lmp -k on g N -sf kk`, SLURM `--ntasks=1`, explicit “no mpirun” in README / API notes / slurm comments. Matches user clarification.
- **Outputs:** `results/ngpu{N}/parity.json` + `forces.npz` (energies + per-atom forces) + `run.log`; collectors merge path × ngpu.
- **C++:** untouched; rebuild notes accurate.
- **Style:** mirrors `delta_parity/run_parity.py` closely; minimal surface.

---

## Security

No secrets, tokens, or credentials. Paths are lab-local (`xyan11`, conda env, uma-cache checkpoint) consistent with sibling suites.

---

## Rebuild

None required for this deliverable. Existing `build-uma/lmp` is appropriate for the single-rank Kokkos recipe (`BUILD_MPI=OFF` as documented).
