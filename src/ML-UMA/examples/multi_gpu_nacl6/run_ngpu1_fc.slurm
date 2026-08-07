#!/bin/bash
#SBATCH --job-name=uma-mgpu-n1-fc
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

# VRAM isolation: ONE path per job (fc @ 1 GPU)
set -euo pipefail
export NGPUS=1
export UMA_DEVICES=1
export ONLY_PATHS=fc
export FAIRCHEM_WORKERS=1
export RECOMPILE="${RECOMPILE:-0}"
export MERGE_RESULTS=1
export SKIP_DIST_DESTROY=1
export HARD_EXIT_AFTER_FC=1
source /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/_run_common.sh
