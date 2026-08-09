#!/bin/bash
#SBATCH --job-name=ase-merge-h2o
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/ase_merge_oracle_water-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/ase_merge_oracle_water-%j.out
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
WEX=${ROOT}/src/ML-UMA/examples/water888
CAMP=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312
export FAIRCHEM_WORKERS=1
export PYTHONPATH="${ENG}/python:${PYTHONPATH:-}"
python - <<'PY'
import json, time, os
from pathlib import Path
import numpy as np
import torch
from ase.io import read
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from common import inference_settings_with_dtype

camp = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign")
wex = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/water888")
ckpt = os.environ.get("UMA_CHECKPOINT", "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt")
atoms0 = read(str(wex / "water_nvt_300K_atomic_metal.data"), format="lammps-data", atom_style="atomic", units="metal")
atoms0.pbc = True
general_E = -3143.3893561182
# Tier1b water energies
uma_jobs = {
  2: json.loads((wex/"results/uma_ngpu2_20982812/timing.json").read_text()),
  4: json.loads((wex/"results/uma_ngpu4_20982814/timing.json").read_text()),
}
settings = inference_settings_with_dtype("float64")
settings.external_graph_gen = True
settings.activation_checkpointing = False
settings.execution_mode = "general"
settings.merge_mole = True
print("=== ASE water merge_mole FP64@1 ===", flush=True)
pred = load_predict_unit(ckpt, device="cuda", inference_settings=settings, workers=1)
calc = FAIRChemCalculator(pred, task_name="omat")
a = atoms0.copy(); a.calc = calc
e = float(a.get_potential_energy())
f = np.asarray(a.get_forces(), dtype=np.float64)
torch.cuda.synchronize()
np.savez(camp/"oracle_ase_water_merge.npz", energy_eV=e, forces=f)
rows=[]
for ng,t in uma_jobs.items():
    uf=np.load(wex/f"results/uma_ngpu{ng}_{t['jobid']}/forces.npz")["forces"].reshape(-1,3)
    df=uf-f.reshape(-1,3)
    row=dict(
      ngpu=ng, job=t["jobid"], uma_E=t["energy_eV"], nvt_ms=t["nvt_ms_per_step"],
      dE_vs_merge_ase=abs(t["energy_eV"]-e),
      dE_vs_general_ase=abs(t["energy_eV"]-general_E),
      force_max=float(np.max(np.abs(df))),
      force_mae=float(np.mean(np.abs(df))),
      cosine=float(np.dot(uf.ravel(),f.ravel())/(np.linalg.norm(uf)*np.linalg.norm(f))),
    )
    row["ef_pass"]= row["dE_vs_merge_ase"]<=1e-8 and row["force_max"]<=1e-5  # water hist ~5e-6
    rows.append(row)
    print(json.dumps(row,indent=2), flush=True)
out=dict(stamp=time.strftime("%Y-%m-%dT%H:%M:%S"), job=os.environ.get("SLURM_JOB_ID"),
         ase_merge_E=e, general_ase_E=general_E, rows=rows,
         all_ef_pass=all(r["ef_pass"] for r in rows))
(camp/"oracle_ase_water_merge_mole.json").write_text(json.dumps(out,indent=2)+"\n")
print("VERDICT", out["all_ef_pass"], "ASE_E", e)
PY
echo ASE_WATER_MERGE_ORACLE_OK job=${SLURM_JOB_ID}
