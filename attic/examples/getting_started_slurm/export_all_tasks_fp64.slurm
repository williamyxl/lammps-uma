#!/bin/bash
#SBATCH --job-name=export-uma-all-f64
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Export TorchScript FP64 artifacts for every UMA-S energy task head.
#   oc20 oc22 oc25 omat odac omc omol
# → artifacts/uma-s-1p2-<task>-f64/

set -euo pipefail

EX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${EX}/../../../.." && pwd)"
ENG="${ROOT}/src/ML-UMA/uma-engine"
CKPT="${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}"
ART_ROOT="${ENG}/artifacts"

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true
cd "${ROOT}"

PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" python "${ENG}/python/export_artifact.py" \
  --checkpoint "${CKPT}" \
  --dtype float64 \
  --device cuda \
  --all-tasks \
  --skip-existing \
  --artifacts-root "${ART_ROOT}"

echo "=== artifacts ==="
ls -lh "${ART_ROOT}"/*/model_traced.pt 2>/dev/null || true
du -sh "${ART_ROOT}"/uma-s-1p2-*-f64 2>/dev/null || true
test -f "${ART_ROOT}/export_all_float64_summary.json" && \
  python -c "import json; d=json.load(open('${ART_ROOT}/export_all_float64_summary.json')); print([(x['task_name'], x.get('export_format'), (x.get('nacl_parity') or {}).get('passed')) for x in d])"
