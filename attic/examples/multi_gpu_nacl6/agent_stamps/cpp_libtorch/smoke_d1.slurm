#!/bin/bash
#SBATCH --job-name=uma-cpp-d1
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/smoke_d1-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/smoke_d1-%j.out
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART=${ENG}/artifacts/uma-s-1p2-omat-f64
module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312
export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${ENG}/third_party/vesin/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ENG}/build-cpp-mp:${PATH}"
STRUCT="${ENG}/artifacts/nacl64.txt"
echo "=== devices=1 ==="
"${ENG}/build-cpp-mp/uma_parity_cli" "${ART}" "$STRUCT" --devices 1
echo "D1_OK"
