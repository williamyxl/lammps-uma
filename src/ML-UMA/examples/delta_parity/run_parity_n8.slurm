#!/bin/bash
#SBATCH --job-name=uma-parity-n8
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# N=8 supercell 4-path parity: NaCl / Al FCC / Si diamond (4096 / 2048 / 4096 atoms)
# Paths: ASE FairChem FP64, FairChem fix-external, uma/kk double, uma/kk mixed
#
#   sbatch --export=ALL,SKIP_EXPORT=1,SKIP_BUILD=1 \
#     src/ML-UMA/examples/delta_parity/run_parity_n8.slurm

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
export ONLY_SYSTEM="${ONLY_SYSTEM:-n8}"
export N_TIMING="${N_TIMING:-5}"

cd "${ROOT}"
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true
nvidia-smi --query-gpu=memory.total,memory.free --format=csv || true
echo "python=$(which python)  torch=$(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "LMP_FC=${LMP_FC}  LMP_UMA=${LMP_UMA}  ONLY_SYSTEM=${ONLY_SYSTEM}"

# Artifacts / binary should already exist from prior parity job
if [[ ! -f "${ENG}/artifacts/uma-s-1p2-omat/model_traced.pt" || ! -f "${ENG}/artifacts/uma-s-1p2-omat-f64/model_traced.pt" ]]; then
  echo "ERROR: missing TorchScript artifacts; run full run_parity.slurm first" >&2
  exit 1
fi
test -x "${LMP_UMA}" || { echo "ERROR: missing ${LMP_UMA}"; exit 1; }

# Freeze N=8 structures only
python "${EX}/freeze_structures.py" nacl8 al8 si8

export SKIP_BUILD=1
python "${EX}/run_parity.py"
python "${EX}/write_reports.py"

echo "DONE. Results under ${EX}/results/"
ls -la "${EX}/results/" || true
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
