#!/bin/bash
#SBATCH --job-name=uma-gp-f32-diag
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=gp_round/%x-%j.out
#SBATCH --error=gp_round/%x-%j.out

set -euo pipefail
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312
export WANDB_MODE=disabled WANDB_DISABLED=true

python3 - <<'PY'
import json, os
from pathlib import Path
import numpy as np
from ase.io import read
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import inference_settings_default
import torch

struct = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz")
ckpt = "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"
atoms0 = read(str(struct))
print("natoms", len(atoms0), "cuda", torch.cuda.device_count())

def run(dtype: str, workers: int):
    settings = inference_settings_default()
    settings.base_precision_dtype = getattr(torch, dtype)
    settings.external_graph_gen = workers <= 1
    settings.activation_checkpointing = False
    settings.execution_mode = "general"
    pred = load_predict_unit(ckpt, device="cuda", inference_settings=settings, workers=workers)
    calc = FAIRChemCalculator(pred, task_name="omat")
    atoms = atoms0.copy()
    atoms.calc = calc
    e = float(atoms.get_potential_energy())
    f = np.asarray(atoms.get_forces(), dtype=np.float64)
    print(f"RESULT dtype={dtype} workers={workers} E={e:.12f} fmax={np.abs(f).max():.6e} frms={np.sqrt((f*f).mean()):.6e}", flush=True)
    return e, f

results = {}
for dtype in ("float64", "float32"):
    for workers in (1, 2):
        key = f"{dtype}_w{workers}"
        e, f = run(dtype, workers)
        results[key] = {"E": e, "fmax": float(np.abs(f).max())}

# comparisons
pairs = [
    ("float64_w1", "float64_w2"),
    ("float32_w1", "float32_w2"),
    ("float32_w1", "float64_w1"),
    ("float32_w2", "float64_w2"),
]
print("=== deltas ===")
for a,b in pairs:
    dE = abs(results[a]["E"] - results[b]["E"])
    print(f"|E({a})-E({b})| = {dE:.6e}")

out = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/gp_round/f32_diag.json")
out.write_text(json.dumps(results, indent=2) + "\n")
print("wrote", out)
PY
