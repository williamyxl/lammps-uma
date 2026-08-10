#!/bin/bash
#SBATCH --job-name=h2o-ase
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/water888/logs/path_ase-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/water888/logs/path_ase-%j.out

# Override GPUs: sbatch --gpus-per-node=N --export=ALL,NGPUS=N,FAIRCHEM_WORKERS=N,NSTEPS=100 ...

set -euo pipefail
# Always resolve to this script's directory (ignore SLURM_SUBMIT_DIR / --chdir).
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${EX}/logs"
export NGPUS="${NGPUS:-${SLURM_GPUS_ON_NODE:-1}}"
export FAIRCHEM_WORKERS="${FAIRCHEM_WORKERS:-${NGPUS}}"
export NSTEPS="${NSTEPS:-100}"
export UMA_CHECKPOINT="${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}"

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID:-none} path=ase NGPUS=${NGPUS} WORKERS=${FAIRCHEM_WORKERS} NSTEPS=${NSTEPS}"
cd "${EX}"
python run_path_ase.py --data "${EX}/water_nvt_300K_atomic_metal.data" \
  --ckpt "${UMA_CHECKPOINT}" --workers "${FAIRCHEM_WORKERS}" --nsteps "${NSTEPS}"
echo "RESULT=${EX}/results/ase_ngpu${NGPUS}_${SLURM_JOB_ID}/timing.json"
