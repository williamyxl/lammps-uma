# Activation checkpointing for UMA (capacity lever)

**What it is:** during the forward pass, do NOT keep the intermediate activations
of each message-passing block for the backward pass; instead recompute them in
backward. Trades extra compute for a large reduction in retained activation
memory. This is the single highest-leverage, exact, low-effort way to run bigger
boxes on fixed GPU memory (A100-40GB).

FairChem exposes it as `InferenceSettings.activation_checkpointing`; internally
each of UMA-S's 4 message-passing blocks is wrapped in `torch.utils.checkpoint`.

## Why it helps so much for UMA

Activations dominate UMA's GPU memory. Measured on one A100 (NaCl, FP64):

| N atoms | ckpt OFF peak | ckpt ON peak | activation share |
|--------:|--------------:|-------------:|-----------------:|
| 1728 | 30.6 GiB | 10.1 GiB | ~20 GiB (66%) removed |

The node-embedding tensors of each block scale ~linearly with N and are the
binding constraint (not weights, not edges). Recomputing them frees ~2/3 of the
footprint.

## Measured effect (single A100-40GB, NaCl, all bit-exact)

**Capacity — max atoms that fit a single-point E+F (autograd):**

| config | max atoms | vs baseline |
|--------|----------:|------------:|
| baseline (no checkpointing) | 1,728 | 1.0× |
| **activation checkpointing** | **21,952** | **12.7×** |
| checkpointing + save_on_cpu (offload) | 32,768 | 19× |

**Throughput + correctness (N=1728):**

| config | ms/step | vs baseline | mem saving | \|ΔE\| | max\|ΔF\| |
|--------|--------:|------------:|-----------:|-------:|---------:|
| baseline | 418.6 | 1.00× | 1.0× | 0 | 0 |
| **checkpointing** | 557.1 | **1.33×** | **3.0×** | **0.0** | **4.5e-16** |

**Checkpointing is exactly correct** — energy is bit-identical (ΔE = 0) and forces
match to the FP64 machine floor (~1e-16). It costs **1.33× step time for 3×
memory / 12.7× atoms**. Excellent trade for the memory-bound regime.

## How to use it

### Export (FairChem / eager settings)
`InferenceSettings.activation_checkpointing = True`. The exporter now takes a
flag:

```
python uma-engine/python/export_artifact.py \
    --checkpoint $UMA_CHECKPOINT --dtype float64 --task omat \
    --output <dir> --activation-checkpointing
```

### Eager inference (works)
Set the flag in `InferenceSettings` and run the FairChem calculator / predictor
directly (this is what the capacity sweeps used). Stacks with
`torch.autograd.graph.save_on_cpu` for even more capacity (see
`cpu_offload_plan.md`).

## CRITICAL limitation: does NOT survive TorchScript tracing

`torch.jit.trace` **cannot** serialize `torch.utils.checkpoint`. Attempting to
export a checkpointed model fails at trace time:

```
Could not export Python function call '_NoopSaveInputs'. Remove calls to Python
functions before export.
  torch/utils/checkpoint.py: _checkpoint_without_reentrant_generator
```

Consequence: the LAMMPS C++ engine loads a **traced** `model_traced.pt`, so it
**cannot** get checkpointing through the normal export. Checkpointing is an
**eager-only** capability today.

### Paths to use checkpointing from LAMMPS
1. **Eager Python-worker path** (recommended): run the eager checkpointed model
   in-process via the engine's `UMA_PYTHON_GP_WORKER` hook. Gets the full 12.7×
   (or 19× with save_on_cpu) at the cost of Python in the loop.
2. **`torch.jit.script` the checkpoint region** instead of `trace`: scripting can
   represent the control flow, but the UMA model has many tracer-only patterns —
   high effort.
3. **Per-block traced + C++ recompute**: trace the 4 blocks separately and have
   the engine recompute activations between blocks (manual checkpointing in
   C++). Medium-high effort, keeps the no-Python engine.

## Recommendation

For the memory-bound goal, checkpointing is the first lever to pull (exact,
12.7×, only 1.33× slower). Combine with `save_on_cpu` (→19×) and graph-parallel
across GPUs/nodes for aggregate capacity. Use it via the eager path until/unless
a per-block C++ recompute is implemented for the traced engine.

## Reproduce

- Capacity sweep: `polaris/pbs/cap_sweep_1gpu.pbs`
  (`max_atoms_sweep.py --checkpointing 0|1`).
- Parity + timing: `polaris/pbs/ckpt_parity.pbs`
  (`ckpt_parity_timing.py`, compares baseline / ckpt / save_on_cpu / soc+ckpt).
- Trace test (shows the export failure): `polaris/pbs/export_ckpt_test.pbs`.
