#!/bin/bash
#SBATCH --job-name=uma-mgpu-nacl6-g4
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=224G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# NaCl 6x6x6 multi-GPU parity — 4 GPUs (Kokkos -k on g 4; FairChem workers=4)
# Absolute source path: SLURM spool breaks BASH_SOURCE-relative paths.

export NGPUS=4
source /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/_run_common.sh
