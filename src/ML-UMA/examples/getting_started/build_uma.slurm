#!/bin/bash
#SBATCH --job-name=build-lammps-uma
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Export TorchScript FP64 artifacts (optional; getting_started uses omat-f64).
#   sbatch .../getting_started/export_all_tasks_fp64.slurm
# → artifacts/uma-s-1p2-{oc20,oc22,oc25,omat,odac,omc,omol}-f64/

set -euo pipefail

EX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${EX}/../../../.." && pwd)"
ENG="${ROOT}/src/ML-UMA/uma-engine"
CKPT="${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}"
ART_F64="${ENG}/artifacts/uma-s-1p2-omat-f64"

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

cd "${ROOT}"
bash "${ENG}/scripts/vendor_vesin_for_arch.sh"

if [[ ! -f "${ART_F64}/model_traced.pt" ]]; then
  PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" python "${ENG}/python/export_artifact.py" \
    --checkpoint "${CKPT}" --dtype float64 --device cuda --output "${ART_F64}"
fi

bash "${ROOT}/scripts/build_lammps_uma.sh"
echo "Built: ${ROOT}/build-uma/lmp"
