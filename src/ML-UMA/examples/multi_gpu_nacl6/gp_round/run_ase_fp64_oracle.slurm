#!/bin/bash
#SBATCH --job-name=uma-ase-fp64-oracle
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=gp_round/%x-%j.out
#SBATCH --error=gp_round/%x-%j.out

# One-shot ASE FairChem FP64 workers=1 ground truth (no ParallelMLIPPredictUnit).
set -euo pipefail
EX=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6
cd "$EX"

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

export UMA_CHECKPOINT="${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}"
export FAIRCHEM_WORKERS=1
export N_TIMING="${N_TIMING:-3}"

echo "=== ASE FP64 oracle (workers=1) ==="
nvidia-smi -L || true
python3 record_ase_fp64_oracle.py --force
echo DONE_ASE_FP64_ORACLE
