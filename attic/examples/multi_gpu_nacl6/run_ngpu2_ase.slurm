#!/bin/bash
#SBATCH --job-name=uma-mgpu-n2-ase
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# VRAM isolation: ONE path per job (ase @ 2 GPU)
set -euo pipefail
export NGPUS=2
export UMA_DEVICES=2
export ONLY_PATHS=ase
export FAIRCHEM_WORKERS=2
export RECOMPILE="${RECOMPILE:-0}"
export MERGE_RESULTS=1
export SKIP_DIST_DESTROY=1
source /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/_run_common.sh
