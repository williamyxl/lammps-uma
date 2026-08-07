#!/usr/bin/env bash
# uma/kk multi-GPU launch recipe (single MPI rank + Kokkos GPUs on one node).
#
# Usage:
#   NGPUS=4 UMA_DEVICES=4 ./launch_uma_kk.sh -in in.sp -log log.sp
#   ./launch_uma_kk.sh 2 -in in.sp          # NGPUS as first arg
#
# pair_style in the input deck should include devices=N, e.g.:
#   pair_style uma/kk precision double devices ${UMA_DEVICES}
#
# On Delta gpuA100x4, request matching GPUs in SLURM:
#   #SBATCH --gpus-per-node=4
#   #SBATCH --partition=gpuA100x4
# CUDA_VISIBLE_DEVICES is set by the allocator; do not remap unless debugging.
#
# Graph-parallel UMA: devices=N shards inference across GPUs (engine GP runtime).
# Keep --ntasks=1; do NOT use mpirun / domain decomposition for full-graph MLIP.

set -euo pipefail

NGPUS="${NGPUS:-1}"
UMA_DEVICES="${UMA_DEVICES:-${NGPUS}}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NGPUS="$1"
  UMA_DEVICES="${UMA_DEVICES:-${NGPUS}}"
  shift
fi

if [[ ! "$NGPUS" =~ ^(1|2|4)$ ]]; then
  echo "ERROR: NGPUS must be 1, 2, or 4 (got '$NGPUS')" >&2
  exit 2
fi
if [[ ! "$UMA_DEVICES" =~ ^(1|2|4)$ ]]; then
  echo "ERROR: UMA_DEVICES must be 1, 2, or 4 (got '$UMA_DEVICES')" >&2
  exit 2
fi

export NGPUS UMA_DEVICES

# .../src/ML-UMA/examples/multi_gpu_nacl6 -> lammps-uma root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LMP="${LMP_UMA:-${ROOT}/build-uma/lmp}"
ENG="${ROOT}/src/ML-UMA/uma-engine"
VESIN="${ENG}/third_party/vesin/lib"
TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${VESIN}:${TORCH_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "launch_uma_kk: NGPUS=${NGPUS} UMA_DEVICES=${UMA_DEVICES} LMP=${LMP}"
if [[ "${UMA_DEVICES}" -gt 1 ]]; then
  echo "  argv: ${LMP} $*  (no Kokkos; devices>1 → FairChem Ray owns GPUs)"
  echo "  pair_style hint: uma precision <double|mixed> devices ${UMA_DEVICES}"
  exec "${LMP}" "$@"
fi
echo "  argv: ${LMP} -k on g ${NGPUS} -sf kk $*"
echo "  pair_style hint: uma/kk precision <double|mixed> devices ${UMA_DEVICES}"
exec "${LMP}" -k on g "${NGPUS}" -sf kk "$@"
