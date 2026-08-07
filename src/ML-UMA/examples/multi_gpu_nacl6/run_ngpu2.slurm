#!/bin/bash
#SBATCH --job-name=uma-mgpu-nacl6-g2
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# NaCl 6x6x6 multi-GPU parity — 2 GPUs (Kokkos -k on g 2; FairChem workers=2)
# Absolute source path: SLURM spool breaks BASH_SOURCE-relative paths.

export NGPUS=2
source /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/_run_common.sh
