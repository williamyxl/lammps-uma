# Multi-GPU API notes (UMA / FairChem)

Short reference for stacks used by `run_multigpu.py`.

**Native product (landed):** [`uma-engine/docs/native_kokkos_libtorch_gp.md`](../../uma-engine/docs/native_kokkos_libtorch_gp.md)  
`uma/kk` + Kokkos `-k on g N` + vesin NL + LibTorch MP shards + **`uma_peer`** — **no Ray**.  
Canonical numbers: [`results/RESULTS.md`](results/RESULTS.md) — **321.04 / 183.30 / 112.04 ms** @1/2/4 (NCCL, job `20940474`), E+F green, campaign **PASS** ≤ASE/FC.

**VRAM isolation:** one `ONLY_PATHS` per SLURM job (`./submit_path_jobs.sh`).

**Timing:** Prefer honest `uma64 E=… XXX ms` / pair timers for scaling reports.
`stamp_slurm_timing.py` wall/`N_TIMING` is provenance only (often contaminated).  
Peer transport: **`nccl`** (code/product default) · `cuda_ipc` / `shm` (fallback via `UMA_PEER_TRANSPORT`).

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
pair_style uma/kk precision double devices ${UMA_DEVICES}
lmp -k on g ${NGPUS} -sf kk -in ...
```

| Mode | Behavior |
|------|----------|
| `devices=1` | Traced LibTorch + vesin CUDA NL |
| `devices>1` **default** | C++ `LibtorchMpRuntime` + `model_mp_w*_n*_r*.pt` + `uma_peer` **CUDA IPC** |
| `devices>1` **PASS path** | `UMA_PEER_TRANSPORT=nccl` → NCCL collectives (campaign numbers above) |
| `devices>1` **opt-in** | `UMA_PYTHON_GP_WORKER=1` → Python process-GP (lab only) |
| `devices>1` **legacy** | `UMA_ALLOW_RAY_GP=1` → FairChem Ray (`uma_gp_worker.py`) — not product |

`UMA_FORBID_RAY_GP=1` rejects Ray. Current pair ms: **321.04 / 183.30 / 112.04** @1/2/4 (job `20940474`, NCCL).

## Delta binding

- Partition: **`gpuA100x4`**, account `bbpl-delta-gpu`
- `--gpus-per-node=NGPUS`, `--ntasks=1`
- Trust SLURM `CUDA_VISIBLE_DEVICES`
