#!/bin/bash
#SBATCH --job-name=n9-oracle
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/logs/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/logs/%x-%j.out

# Dedicated ASE FP64 merge oracle for the 5832-atom perturb-then-replicate box.
# Single GPU by design: this is the reference every path is gated against.
set -uo pipefail
EX=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node
module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID} nacl9rep ASE FP64 oracle"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "${EX}"
python build_nacl9_replicated.py    # idempotent; no-op if present
python make_oracle.py
echo "ORACLE_RC=$?"
