#!/bin/bash
#SBATCH --job-name=uma-mgpu-nacl6-g1
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# NaCl 6x6x6 (1728 atoms) multi-GPU parity — 1 GPU
# uma/kk: Kokkos same-node, 1 MPI task: lmp -k on g 1 -sf kk
# ASE/FC: FairChem workers=1 in one process
# NOTE: use absolute path — SLURM copies this script to /var/spool/... so BASH_SOURCE fails.

export NGPUS=1
source /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/_run_common.sh
