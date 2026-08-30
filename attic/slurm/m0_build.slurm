#!/bin/bash
#SBATCH --job-name=m0-mn-build
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=01:30:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/uma-engine/tests/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/uma-engine/tests/%x-%j.out
set -uo pipefail
module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312
# Isolated dirs only; the script itself refuses protected trees and verifies
# all four protected binaries are byte-identical afterwards.
bash /work/nvme/bfzx/xyan11/workdir/lammps-uma/scripts/build_lammps_uma_mn.sh
echo "M0_BUILD_RC=$?"
