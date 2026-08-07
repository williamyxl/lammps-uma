#!/bin/bash
#SBATCH --job-name=uma-gp-nacl6-g4
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=224G
#SBATCH --time=02:00:00
#SBATCH --output=gp_round/%x-%j.out
#SBATCH --error=gp_round/%x-%j.out

# Graph-parallel round — devices=4 (submit only after ngpu2 gates green)
export NGPUS=4
source /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/gp_round/_run_gp_common.sh
