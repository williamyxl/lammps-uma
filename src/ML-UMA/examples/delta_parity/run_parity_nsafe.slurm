#!/bin/bash
#SBATCH --job-name=uma-parity-nsafe
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# ASE FP64-safe large cells on A100 40GB (~500 atoms; 7x7x7 NaCl OOMed):
#   NaCl 4x4x4 (512), Al 5x5x5 (500), Si 4x4x4 (512)
# Paths: ASE FairChem FP64, FairChem fix-external, uma/kk double, uma/kk mixed

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
export ONLY_SYSTEM="${ONLY_SYSTEM:-nsafe}"
export N_TIMING="${N_TIMING:-5}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT}"
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true
echo "LMP_FC=${LMP_FC}  LMP_UMA=${LMP_UMA}  ONLY_SYSTEM=${ONLY_SYSTEM}"

if [[ ! -f "${ENG}/artifacts/uma-s-1p2-omat/model_traced.pt" || ! -f "${ENG}/artifacts/uma-s-1p2-omat-f64/model_traced.pt" ]]; then
  echo "ERROR: missing TorchScript artifacts" >&2
  exit 1
fi
test -x "${LMP_UMA}" || { echo "ERROR: missing ${LMP_UMA}"; exit 1; }

python "${EX}/freeze_structures.py" nacl4 al5 si4

export SKIP_BUILD=1
python "${EX}/run_parity.py"
python "${EX}/write_reports.py"

echo "DONE. Results under ${EX}/results/"
ls -la "${EX}/results/" || true
