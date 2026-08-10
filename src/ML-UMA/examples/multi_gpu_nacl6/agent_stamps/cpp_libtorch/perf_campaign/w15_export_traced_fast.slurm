#!/bin/bash
#SBATCH --job-name=w15-traced
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/w15_export-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/w15_export-%j.out

# W15: re-export devices=1 model_traced.pt into *-f64-fast with
# umas_fast_pytorch + merge_mole. Prior Tier1 export only wrote MP shards;
# model_traced.pt was still the general f64 copy (E~general, ~533 ms).
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART=${ENG}/artifacts/uma-s-1p2-omat-f64-fast
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID:-none} W15 devices=1 traced fast export"
mkdir -p "${ART}"
if [[ -f "${ART}/model_traced.pt" ]]; then
  ts=$(date +%Y%m%dT%H%M%S)
  mv -v "${ART}/model_traced.pt" "${ART}/model_traced.pt.general_copy_bak_${ts}"
fi

PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" python "${ENG}/python/export_artifact.py" \
  --checkpoint "${CKPT}" \
  --output "${ART}" \
  --dtype float64 \
  --task omat \
  --device cuda \
  --execution-mode umas_fast_pytorch \
  --merge-mole

python - <<PY
import hashlib, json
from pathlib import Path
art = Path("${ART}")
tr = art / "model_traced.pt"
assert tr.is_file(), tr
h = hashlib.md5()
with open(tr, "rb") as f:
    h.update(f.read(1024 * 1024))
meta = json.loads((art / "metadata.json").read_text())
inf = meta.get("inference_settings") or {}
print("traced_size", tr.stat().st_size, "md5_1MB", h.hexdigest())
print("execution_mode", inf.get("execution_mode"), "merge_mole", inf.get("merge_mole"))
assert inf.get("execution_mode") == "umas_fast_pytorch", inf
assert inf.get("merge_mole") is True, inf
print("W15_EXPORT_OK job=${SLURM_JOB_ID} art=${ART}")
PY
