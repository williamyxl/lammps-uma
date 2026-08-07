# Multi-GPU API notes (UMA / FairChem)

Short reference for stacks used by `run_multigpu.py`.

**Native plan (active):** [`uma-engine/docs/native_kokkos_libtorch_gp.md`](../../uma-engine/docs/native_kokkos_libtorch_gp.md)  
Target: `uma/kk` + Kokkos `-k on g N` + vesin NL + LibTorch shards — **no Ray**.  
Land **devices=2** on `gpuA100x4` before **devices=4**.

**VRAM isolation:** one `ONLY_PATHS` per SLURM job (`./submit_path_jobs.sh`).

**Timing:** Prefer honest `uma64 E=… XXX ms` / pair timers for scaling reports.
`stamp_slurm_timing.py` wall/`N_TIMING` is provenance only (often contaminated).

## FairChem ASE — `workers=` (external oracle / baseline only)

```
load_predict_unit(..., workers=N)  # workers>1 → Ray ParallelMLIPPredictUnit
```

Not the uma/kk product backend.

## FairChem LAMMPS fix-external (baseline only)

```python
predictor = load_predict_unit(..., workers=N)
run_lammps_with_fairchem(predictor, inp, "omat")
```

## uma/kk — product path

```bash
# Target / devices=1 today:
pair_style uma/kk precision double devices ${UMA_DEVICES}
lmp -k on g ${NGPUS} -sf kk -in ...
```

| Mode | Behavior |
|------|----------|
| `devices=1` | Traced LibTorch + vesin CUDA NL |
| `devices>1` **legacy** | FairChem Ray worker; `run_multigpu.py` may use plain `uma` without Kokkos |
| `devices>1` **target** | Vesin full graph → `graph_shard.h` → LibTorch + Kokkos peer; keep `uma/kk` + `-k on g N` |

Development: `UMA_FORBID_RAY_GP=1` refuses legacy Ray fork.

## Delta binding

- Partition: **`gpuA100x4`**, account `bbpl-delta-gpu`
- `--gpus-per-node=NGPUS`, `--ntasks=1`
- Trust SLURM `CUDA_VISIBLE_DEVICES`
