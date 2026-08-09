#!/bin/bash
#SBATCH --job-name=ase-merge-or
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/ase_merge_oracle-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/ase_merge_oracle-%j.out

# ASE FP64@1 oracle with merge_mole(+optional umas_fast) vs uma merge energy.
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
EX=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6
ENG=${ROOT}/src/ML-UMA/uma-engine
CAMP=${EX}/agent_stamps/cpp_libtorch/perf_campaign
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

export FAIRCHEM_WORKERS=1
export PYTHONPATH="${ENG}/python:${EX}:${PYTHONPATH:-}"

python - <<'PY'
import json, time, os
from pathlib import Path
import numpy as np
import torch
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from load_geometry import load_nacl6_fixed
from common import inference_settings_with_dtype

camp = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign")
ckpt = os.environ.get("UMA_CHECKPOINT", "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt")
atoms0 = load_nacl6_fixed()
uma_merge_E = -5830.9237413382  # from gate 20983451
general_ase_E = -5830.92372016672

variants = [
    ("general_merge", "general", True),
    ("umas_fast_merge", "umas_fast_pytorch", True),
]
rows = []
for name, mode, merge in variants:
    settings = inference_settings_with_dtype("float64")
    settings.external_graph_gen = True
    settings.activation_checkpointing = False
    settings.execution_mode = mode
    settings.merge_mole = merge
    print(f"=== ASE oracle {name} mode={mode} merge_mole={merge} ===", flush=True)
    t0 = time.perf_counter()
    pred = load_predict_unit(ckpt, device="cuda", inference_settings=settings, workers=1)
    calc = FAIRChemCalculator(pred, task_name="omat")
    a = atoms0.copy()
    a.calc = calc
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(), dtype=np.float64)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # timed repeat
    times = []
    for _ in range(3):
        if hasattr(a.calc, "results"):
            a.calc.results.clear()
        a.positions = a.positions.copy()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        e = float(a.get_potential_energy())
        f = np.asarray(a.get_forces(), dtype=np.float64)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t1)
    ms = float(np.mean(times) * 1e3)
    row = {
        "name": name,
        "execution_mode": mode,
        "merge_mole": merge,
        "energy_eV": e,
        "ms_per_eval": ms,
        "dE_vs_general_ase": abs(e - general_ase_E),
        "dE_vs_uma_merge": abs(e - uma_merge_E),
        "force_rms": float(np.sqrt(np.mean(f**2))),
    }
    rows.append(row)
    np.savez(camp / f"oracle_ase_{name}.npz", energy_eV=e, forces=f)
    print(json.dumps(row, indent=2), flush=True)
    a.calc = None
    del calc, a, pred
    torch.cuda.empty_cache()

out = {
    "stamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "job": os.environ.get("SLURM_JOB_ID"),
    "uma_merge_E": uma_merge_E,
    "general_ase_E": general_ase_E,
    "rows": rows,
    "verdict": {
        "merge_matches_uma": any(r["dE_vs_uma_merge"] <= 1e-8 for r in rows),
        "merge_matches_general_ase": any(r["dE_vs_general_ase"] <= 1e-8 for r in rows),
    },
}
(camp / "oracle_ase_merge_mole.json").write_text(json.dumps(out, indent=2) + "\n")
print("WROTE", camp / "oracle_ase_merge_mole.json")
print("VERDICT", out["verdict"])
PY
echo ASE_MERGE_ORACLE_OK job=${SLURM_JOB_ID}
