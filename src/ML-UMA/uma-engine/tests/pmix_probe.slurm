#!/bin/bash
#SBATCH --job-name=pmix-probe
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/uma-engine/tests/pmix-probe-%j.out
set -uo pipefail
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312
echo "=== WITHOUT workaround ==="
srun --mpi=pmix --ntasks=2 hostname 2>&1 | grep -iE 'PMIX_ERR|error|^gpu' | head -4 || true
echo "rc_before=$?"
echo "=== WITH workaround ==="
export TMPDIR=/tmp/pmix_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"
export PMIX_MCA_gds=hash
srun --mpi=pmix --ntasks=2 hostname 2>&1 | grep -iE 'PMIX_ERR|error' | head -4
echo "rc_after=$?"
