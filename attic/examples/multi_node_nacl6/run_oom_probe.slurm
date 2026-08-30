#!/bin/bash
#SBATCH --job-name=uma-oom-probe
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=224G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Phase G1 OOM probe: one path @ 4 GPUs on NaCl N×N×N (ideal cell).
# Writes ONLY to RESULTS_DIR (independent; no shared parity merge).
# Required exports: NACL_N, ONLY_PATHS (single), RESULTS_DIR
set -euo pipefail

: "${NACL_N:?}"
: "${ONLY_PATHS:?}"
: "${RESULTS_DIR:?}"

ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
EX_MN=${ROOT}/src/ML-UMA/examples/multi_node_nacl6
EX_MG=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

export NGPUS=4
export UMA_DEVICES=4
export FAIRCHEM_WORKERS=4
export N_TIMING="${N_TIMING:-1}"
export RECOMPILE="${RECOMPILE:-0}"
export MERGE_RESULTS=0
export ALLOW_MULTI_PATH=0
export SKIP_DIST_DESTROY=1
export USE_SLURM_TIMING=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export UMA_CHECKPOINT="${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}"
export LMP_UMA="${LMP_UMA:-${ROOT}/build-uma/lmp}"
export LMP_FC="${LMP_FC:-/u/xyan11/miniforge3-x86_64/envs/uma312/bin/lmp}"

mkdir -p "${RESULTS_DIR}"
STRUCT="${RESULTS_DIR}/structure_ideal.extxyz"
python "${EX_MN}/build_nacl_n.py" --n "${NACL_N}" --ideal --out "${STRUCT}"
export FIXED_STRUCTURE="${STRUCT}"
export EXPECTED_NATOMS=$((8 * NACL_N * NACL_N * NACL_N))
if [[ "${ONLY_PATHS}" == "fc" ]]; then
  export HARD_EXIT_AFTER_FC=1
fi

# Soften ONLY_PATHS single-path check via multi_gpu _run pieces without merge.
cd "${EX_MG}"
echo "=== OOM probe N=${NACL_N} path=${ONLY_PATHS} RESULTS_DIR=${RESULTS_DIR} ==="
nvidia-smi -L || true
_t0=$(date +%s.%N)
set +e
python "${EX_MG}/run_multigpu.py"
_rc=$?
set -e
_t1=$(date +%s.%N)
_wall=$(awk -v a="${_t0}" -v b="${_t1}" 'BEGIN { printf "%.6f", b-a }')

STATUS=PASS
if [[ "${_rc}" -ne 0 ]]; then
  STATUS=FAIL
  # Heuristic OOM tag from log if present
  if grep -qiE 'out of memory|cuda.*oom|cublas.*alloc' "${RESULTS_DIR}"/*.log "${SLURM_SUBMIT_DIR}"/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out 2>/dev/null; then
    STATUS=OOM
  fi
fi

python - <<PY
import json, os
from pathlib import Path
out = Path(os.environ["RESULTS_DIR"])
status = {
  "n": int(os.environ["NACL_N"]),
  "natoms": 8 * int(os.environ["NACL_N"]) ** 3,
  "path": os.environ["ONLY_PATHS"],
  "ngpus": 4,
  "status": "${STATUS}",
  "returncode": int("${_rc}"),
  "slurm_wall_s": float("${_wall}"),
  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
  "results_dir": str(out),
}
(out / "probe_status.json").write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status, indent=2))
PY

# Stamp SLURM timing into parity if present (PASS only)
if [[ "${STATUS}" == "PASS" && -f "${RESULTS_DIR}/parity.json" ]]; then
  python "${EX_MG}/stamp_slurm_timing.py" \
    --results-dir "${RESULTS_DIR}" \
    --wall-s "${_wall}" \
    --n-timing "${N_TIMING}" \
    --paths "${ONLY_PATHS}" || true
fi

nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
echo "DONE probe status=${STATUS} rc=${_rc}"
# Always exit 0 so parallel sweep siblings are not cancelled; status is in probe_status.json
exit 0
