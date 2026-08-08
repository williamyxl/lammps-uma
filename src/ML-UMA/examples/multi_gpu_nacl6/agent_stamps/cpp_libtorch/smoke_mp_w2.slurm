#!/bin/bash
#SBATCH --job-name=uma-cpp-mp-smoke
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/smoke_mp-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/smoke_mp-%j.out

# Rebuild C++ MP runtime + parity_cli --devices 2 (artifacts already exported).
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART=${ENG}/artifacts/uma-s-1p2-omat-f64
STAMP=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=$SLURM_JOB_ID host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
ls -lh "${ART}"/model_mp_w2_r*.pt

export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export RECOMPILE=1
export UMA_MP_LOG_DIR="${STAMP}/worker_logs_${SLURM_JOB_ID:-manual}"
mkdir -p "${UMA_MP_LOG_DIR}"
echo "UMA_MP_LOG_DIR=${UMA_MP_LOG_DIR}"

echo "=== rebuild uma-engine ==="
mkdir -p "${ENG}/build-cpp-mp"
cd "${ENG}/build-cpp-mp"
cmake .. -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" -DUMA_ENGINE_USE_CUDA=ON
cmake --build . -j "${SLURM_CPUS_PER_TASK:-8}"

export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${ENG}/third_party/vesin/lib:${LD_LIBRARY_PATH:-}"
export UMA_LIBTORCH_MP_WORKER="${ENG}/build-cpp-mp/uma_libtorch_mp_worker"
export PATH="${ENG}/build-cpp-mp:${PATH}"

echo "=== inspect uma_peer ops in traced module ==="
PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" python - <<PY || echo "WARN: inspect failed (non-fatal)"
import torch
from pathlib import Path
from uma_peer_ops import install_export_ops
install_export_ops()
p = Path("${ART}") / "model_mp_w2_r0.pt"
m = torch.jit.load(str(p), map_location="cpu")
g = m.inlined_graph
ops = sorted({n.kind() for n in g.nodes()})
peer = [k for k in ops if "uma_peer" in k]
print("uma_peer ops in graph:", peer)
print("n nodes", len(list(g.nodes())))
for n in g.nodes():
    if "uma_peer" in n.kind():
        print(" ", n.kind())
PY

echo "=== kokkos_peer_device_smoke ==="
"${ENG}/build-cpp-mp/kokkos_peer_device_smoke"

echo "=== uma_parity_cli devices=2 ==="
STRUCT="${ENG}/artifacts/nacl64.txt"
set +e
"${ENG}/build-cpp-mp/uma_parity_cli" "${ART}" "$STRUCT" --devices 2
RC=$?
set -e
echo "=== worker logs (${UMA_MP_LOG_DIR}) ==="
ls -la "${UMA_MP_LOG_DIR}" || true
for f in "${UMA_MP_LOG_DIR}"/worker_r*.log; do
  [[ -f "$f" ]] || continue
  echo "----- $f -----"
  tail -80 "$f" || true
done
if [[ $RC -ne 0 ]]; then
  echo "parity_cli failed rc=$RC"
  exit "$RC"
fi

echo "SMOKE_OK job=$SLURM_JOB_ID"
