#!/bin/bash
#SBATCH --job-name=uma-cpp-exp1728
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/export_nacl6-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/export_nacl6-%j.out

# Export MP TorchScript traced on 1728-atom NaCl6 (baked partition offsets).
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART=${ENG}/artifacts/uma-s-1p2-omat-f64
EX=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
STAMP=${EX}/agent_stamps/cpp_libtorch
ATOMS=${EX}/structures/nacl6_rattle_fixed.extxyz

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=$SLURM_JOB_ID host=$(hostname) ATOMS=$ATOMS"
ls -lh "$ATOMS"

echo "=== export_mp_artifact world=2 n=1728 (per-rank CVD) ==="
for R in 0 1; do
  CUDA_VISIBLE_DEVICES=$R PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" \
    python "${ENG}/python/export_mp_artifact.py" \
      --checkpoint "${CKPT}" \
      --output "${ART}" \
      --world 2 \
      --ranks "$R" \
      --dtype float64 \
      --task omat \
      --device cuda \
      --state-dict "${ART}/model_state.pt" \
      --atoms "${ATOMS}"
done

ls -lh "${ART}"/model_mp_w2_n1728_r*.pt
echo "EXPORT_NACL6_OK job=$SLURM_JOB_ID"
