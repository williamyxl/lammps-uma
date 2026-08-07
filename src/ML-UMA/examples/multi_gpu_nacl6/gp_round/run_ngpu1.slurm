#!/bin/bash
#SBATCH --job-name=uma-gp-nacl6-g1
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=02:00:00
#SBATCH --output=gp_round/%x-%j.out
#SBATCH --error=gp_round/%x-%j.out

# Graph-parallel round — devices=1 baseline (uma_double + uma_mixed only)
export NGPUS=1
source /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/gp_round/_run_gp_common.sh
