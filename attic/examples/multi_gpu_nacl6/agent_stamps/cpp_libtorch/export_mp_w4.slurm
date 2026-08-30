#!/bin/bash
#SBATCH --job-name=uma-cpp-exp-w4
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=03:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/export_w4-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/export_w4-%j.out

# Export world=4 MP TorchScript for n=64 (legacy name) and n=1728 (n-tagged).
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART=${ENG}/artifacts/uma-s-1p2-omat-f64
EX=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
ATOMS1728=${EX}/structures/nacl6_rattle_fixed.extxyz
EXPORT_N64="${EXPORT_N64:-1}"
EXPORT_N1728="${EXPORT_N1728:-1}"

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=$SLURM_JOB_ID host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

export_one() {
  local R=$1
  shift
  echo "=== export world=4 rank=$R $* ==="
  CUDA_VISIBLE_DEVICES=$R PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" \
    python "${ENG}/python/export_mp_artifact.py" \
      --checkpoint "${CKPT}" \
      --output "${ART}" \
      --world 4 \
      --ranks "$R" \
      --dtype float64 \
      --task omat \
      --device cuda \
      --state-dict "${ART}/model_state.pt" \
      "$@"
}

if [[ "${EXPORT_N64}" == "1" ]]; then
  # Default NaCl64 → model_mp_w4_r{0..3}.pt
  for R in 0 1 2 3; do export_one "$R"; done
  ls -lh "${ART}"/model_mp_w4_r*.pt
fi

if [[ "${EXPORT_N1728}" == "1" ]]; then
  for R in 0 1 2 3; do export_one "$R" --atoms "${ATOMS1728}"; done
  ls -lh "${ART}"/model_mp_w4_n1728_r*.pt
fi

echo "EXPORT_W4_OK job=$SLURM_JOB_ID"
