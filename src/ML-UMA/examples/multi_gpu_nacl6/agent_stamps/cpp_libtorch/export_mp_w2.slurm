#!/bin/bash
#SBATCH --job-name=uma-cpp-mp-export
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/export_mp-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/export_mp-%j.out

# Export per-rank MP TorchScript + rebuild engine + peer device smoke.
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART=${ENG}/artifacts/uma-s-1p2-omat-f64
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
STAMP=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=$SLURM_JOB_ID host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

# Export each rank with CUDA_VISIBLE_DEVICES=<rank> so TS bakes cuda:0.
echo "=== export_mp_artifact world=2 (per-rank CVD) ==="
for R in 0 1; do
  CUDA_VISIBLE_DEVICES=$R PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" \
    python "${ENG}/python/export_mp_artifact.py" \
      --checkpoint "${CKPT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}" \
      --output "${ART}" \
      --world 2 \
      --ranks "$R" \
      --dtype float64 \
      --task omat \
      --device cuda \
      --state-dict "${ART}/model_state.pt"
done

ls -lh "${ART}"/model_mp_w2_r*.pt "${ART}/model_mp_export.json"

echo "=== rebuild uma-engine (RECOMPILE) ==="
mkdir -p "${ENG}/build-cpp-mp"
cd "${ENG}/build-cpp-mp"
cmake .. -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" -DUMA_ENGINE_USE_CUDA=ON
cmake --build . -j "${SLURM_CPUS_PER_TASK:-8}"

export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${ENG}/third_party/vesin/lib:${LD_LIBRARY_PATH:-}"
export UMA_LIBTORCH_MP_WORKER="${ENG}/build-cpp-mp/uma_libtorch_mp_worker"
export PATH="${ENG}/build-cpp-mp:${PATH}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TORCH_SHOW_CPP_STACKTRACES=1

echo "=== kokkos_peer_device_smoke ==="
"${ENG}/build-cpp-mp/kokkos_peer_device_smoke"

echo "=== uma_parity_cli devices=2 (C++ MP path) ==="
# Small structure if available
STRUCT="${ENG}/artifacts/nacl64.txt"
if [[ -f "$STRUCT" ]]; then
  "${ENG}/build-cpp-mp/uma_parity_cli" "${ART}" "$STRUCT" --devices 2 || {
    echo "parity_cli devices=2 failed (expected until MP export/runtime green)"
    exit 1
  }
else
  echo "WARN: no structure for parity_cli"
fi

echo "EXPORT_AND_SMOKE_OK job=$SLURM_JOB_ID"
