#!/bin/bash
#SBATCH --job-name=w15-h2o-tr
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/w15_export_water-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/w15_export_water-%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART=${ENG}/artifacts/uma-s-1p2-omat-f64-fast-water648
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
WATER_XYZ=${ROOT}/src/ML-UMA/examples/water888/water_nvt_300K.extxyz
module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312
echo "JOBID=${SLURM_JOB_ID:-none} W15 water@1 traced"
mkdir -p "${ART}"
PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" python "${ENG}/python/w15_export_traced_fast.py" \
  --checkpoint "${CKPT}" --output "${ART}" --atoms "${WATER_XYZ}" \
  --dtype float64 --task omat --device cuda --out-name model_traced.pt --backup-existing
echo "W15_WATER_EXPORT_OK job=${SLURM_JOB_ID} art=${ART}"
