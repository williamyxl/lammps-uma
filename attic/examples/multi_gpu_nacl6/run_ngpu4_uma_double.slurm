#!/bin/bash
#SBATCH --job-name=uma-mgpu-n4-d64
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=224G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# VRAM isolation: ONE path per job (uma_double @ 4 GPU)
set -euo pipefail
export NGPUS=4
export UMA_DEVICES=4
export ONLY_PATHS=uma_double
export FAIRCHEM_WORKERS=4
export RECOMPILE="${RECOMPILE:-0}"
export MERGE_RESULTS=1
export SKIP_DIST_DESTROY=1
source /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/_run_common.sh
