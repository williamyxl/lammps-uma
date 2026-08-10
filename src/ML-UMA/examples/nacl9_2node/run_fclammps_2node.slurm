#!/bin/bash
#SBATCH --job-name=n9-fclmp-2n
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=00:45:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/logs/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/logs/%x-%j.out

# FC LAMMPS across 2 nodes. Two independent blockers are expected:
#   1. build-uma/lmp is built -DBUILD_MPI=OFF ("LAMMPS MPI STUBS"), so a single
#      lmp process cannot span nodes at all.
#   2. FC + FP64 + merge_mole crashes in merge_MOLE (Float/Double), the same
#      failure that blocked the matching-settings FC bar (jobs 20989342+).
# Run it anyway to capture both diagnostics as recorded evidence rather than
# assertion, and stamp the outcome as data.
set -uo pipefail
EX=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node
module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

cd "${EX}"
python build_nacl9_replicated.py

echo "=== BLOCKER 1: does lmp have real MPI? ==="
/work/nvme/bfzx/xyan11/workdir/lammps-uma/build-uma/lmp -h 2>/dev/null \
  | grep -iE 'MPI v|MPI STUBS' | head -3 | tee /tmp/fclmp_mpi_check.txt

echo "=== BLOCKER 2: FC FP64 + merge_mole on this box ==="
python run_fclammps_nacl9.py --workers "${WORKERS:-8}"
echo "FCLMP_2NODE_RC=$?"
