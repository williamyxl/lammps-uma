#!/bin/bash
#SBATCH --job-name=uma-gp-nacl6-g2
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=gp_round/%x-%j.out
#SBATCH --error=gp_round/%x-%j.out

# Graph-parallel round — devices=2 parity gate (requires ngpu1 baseline)
export NGPUS=2
source /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/gp_round/_run_gp_common.sh
