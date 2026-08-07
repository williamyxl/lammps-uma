# Multi-GPU API notes (UMA / FairChem)

Short reference for the three stacks used by `run_multigpu.py`.

**VRAM isolation:** submit **one** `ONLY_PATHS` value per SLURM job
(`./submit_path_jobs.sh`). Multi-path in one allocation is refused unless
`ALLOW_MULTI_PATH=1`.

**Timing:** `_run_common.sh` measures wall time of `run_multigpu.py` and
`stamp_slurm_timing.py` writes that as the sole `ms_per_eval`
(`1000 * wall_s / N_TIMING`). Python/Pair internal timers are not the
reported result.

## FairChem ASE — `workers=`

Installed signature (`fairchem.core.units.mlip_unit.load_predict_unit` in `uma312`):

```
load_predict_unit(
    path,
    inference_settings='default',
    overrides=None,
    device=None,
    atom_refs=None,
    form_elem_refs=None,
    workers: int = 1,
    seed: int = 41,
)
```

- `workers == 1` → `MLIPPredictUnit` on one device.
- `workers > 1` → `ParallelMLIPPredictUnit` (Ray placement groups, 1 GPU per
  worker, NCCL + FairChem `gp_utils` graph-parallel groups).
- Also available via `pretrained_mlip.get_predict_unit(..., workers=N)`.
- Needs `fairchem-core[extras]` / Ray.
- For FP64 parity: pass custom `InferenceSettings` with float64 (see
  `fp64_settings` in `run_multigpu.py` / `delta_parity/run_parity.py`).
  **Not** turbo for energy/force parity.

## FairChem LAMMPS fix-external

```python
predictor = load_predict_unit(..., workers=N)
run_lammps_with_fairchem(predictor, inp, "omat")
```

Multi-GPU is entirely inside the predictor (Ray workers). The conda `lmp`
Python bridge stays single-process.

## uma/kk — Kokkos `g N` + `devices N`, single MPI rank

```bash
pair_style uma/kk precision double devices ${UMA_DEVICES}
lmp -k on g ${NGPUS} -sf kk -in ...
# helper: ./launch_uma_kk.sh
```

- **Preferred** same-node multi-GPU launch. MPI / `mpirun -np N` is **not** used.
- `pair_uma.cpp` still errors if `comm->nprocs > 1` (full-graph MLIP correctness).
- Local binary is built with `BUILD_MPI=OFF` (`scripts/build_lammps_uma.sh`).
- **`devices=1`:** traced LibTorch `model_traced.pt` on one CUDA device.
- **`devices>1`:** FairChem eager graph-parallel via `GraphParallelRuntime` /
  `uma_gp_worker.py` (`load_predict_unit(..., workers=N)`). If `devices` is
  omitted and Kokkos `ngpus>1`, pair auto-sets `devices=ngpus`.
- `UMA_DEVICES` env (default `NGPUS`) is written into `pair_style` by
  `run_multigpu.py`.
- Rebuild required after engine/pair GP wiring (`rebuild_needed=true` in WRITE stamp).

## Delta binding

`gpuA100x4`: request `--gpus-per-node=NGPUS`. Trust SLURM
`CUDA_VISIBLE_DEVICES`. Kokkos `-k on g N` counts visible devices.
