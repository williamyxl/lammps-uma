#!/bin/bash
#SBATCH --job-name=uma-cpp-nacl6
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/smoke_nacl6-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/smoke_nacl6-%j.out

# devices=2 E+F gate on 1728-atom NaCl6 vs devices=1 (ASE oracle −5830.92…).
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
export STRUCT="${ENG}/artifacts/nacl6_1728.txt"
export FORCE_TOL="${FORCE_TOL:-1e-6}"
export ENERGY_TOL="${ENERGY_TOL:-1e-8}"
export RECOMPILE="${RECOMPILE:-1}"
# MP TS traced on n=1728 (baked partition offsets).
export UMA_MP_NATOMS=1728
STAMP=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch
bash "${STAMP}/smoke_mp_w2.slurm"
