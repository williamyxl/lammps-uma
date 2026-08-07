#!/bin/bash
#SBATCH --job-name=uma-parity-nacl5
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Single-system probe: NaCl 5x5x5 (1000 atoms) × 4 paths on A100 40GB

set -euo pipefail

ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
EX=${ROOT}/src/ML-UMA/examples/delta_parity
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
CONDA_SH=/u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
LMP_FC=${LMP_FC:-/u/xyan11/miniforge3-x86_64/envs/uma312/bin/lmp}

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8

source "${CONDA_SH}"
conda activate uma312

export UMA_CHECKPOINT="${CKPT}"
export LMP_FC
export LMP_UMA="${LMP_UMA:-${ROOT}/build-uma/lmp}"
export PYTHONUNBUFFERED=1
export ONLY_SYSTEM=nacl5
export N_TIMING="${N_TIMING:-5}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT}"
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true
nvidia-smi --query-gpu=memory.total,memory.free --format=csv || true

test -x "${LMP_UMA}" || { echo "ERROR: missing ${LMP_UMA}"; exit 1; }
test -f "${ENG}/artifacts/uma-s-1p2-omat/model_traced.pt"
test -f "${ENG}/artifacts/uma-s-1p2-omat-f64/model_traced.pt"

python "${EX}/freeze_structures.py" nacl5
export SKIP_BUILD=1
python "${EX}/run_parity.py"
python "${EX}/write_reports.py"

echo "DONE"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
