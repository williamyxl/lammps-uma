#!/bin/bash
#SBATCH --job-name=h2o-uma
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/water888/logs/path_uma-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/water888/logs/path_uma-%j.out

# Override GPUs: sbatch --gpus-per-node=N --export=ALL,NGPUS=N,UMA_DEVICES=N,NSTEPS=100 ...
# Multi-GPU needs model_mp_wN_n648_r*.pt from export_mp_water888.slurm

set -euo pipefail
EX="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT="$(cd "${EX}/../../../.." && pwd)"
ENG="${ROOT}/src/ML-UMA/uma-engine"
mkdir -p "${EX}/logs"

export NGPUS="${NGPUS:-${SLURM_GPUS_ON_NODE:-1}}"
export UMA_DEVICES="${UMA_DEVICES:-${NGPUS}}"
export NSTEPS="${NSTEPS:-100}"
export RECOMPILE="${RECOMPILE:-0}"
export UMA_FORBID_RAY_GP=1
export UMA_MP_NATOMS="${UMA_MP_NATOMS:-648}"
export UMA_STRUCTURE_NATOMS="${UMA_STRUCTURE_NATOMS:-648}"
export UMA_PEER_TRANSPORT="${UMA_PEER_TRANSPORT:-nccl}"
export UMA_LIBTORCH_MP_WORKER="${UMA_LIBTORCH_MP_WORKER:-${ENG}/build-cpp-mp/uma_libtorch_mp_worker}"
unset UMA_ALLOW_RAY_GP || true
unset UMA_PYTHON_GP_WORKER || true

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID:-none} path=uma NGPUS=${NGPUS} UMA_DEVICES=${UMA_DEVICES} NSTEPS=${NSTEPS}"

if [[ "${RECOMPILE}" == "1" ]]; then
  bash "${ROOT}/scripts/build_lammps_uma.sh"
fi

ART="${UMA_ARTIFACT_DIR:-${ENG}/artifacts/uma-s-1p2-omat-f64}"
test -x "${LMP_UMA:-${ROOT}/build-uma/lmp}"
test -f "${ART}/model_traced.pt"
if [[ "${UMA_DEVICES}" -gt 1 ]]; then
  test -x "${UMA_LIBTORCH_MP_WORKER}"
  test -f "${ART}/model_mp_w${UMA_DEVICES}_n648_r0.pt"
fi
echo "UMA_ARTIFACT_DIR=${ART}"

VESIN="${ENG}/third_party/vesin/lib"
TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${VESIN}:${TORCH_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

cd "${EX}"
python run_path_uma.py --data "${EX}/water_nvt_300K_atomic_metal.data" \
  --ngpus "${NGPUS}" --uma-devices "${UMA_DEVICES}" --nsteps "${NSTEPS}"
echo "RESULT=${EX}/results/uma_ngpu${NGPUS}_${SLURM_JOB_ID}/timing.json"
