#!/bin/bash
# Shared body for run_ngpu{1,2,4}.slurm
# Expects NGPUS already exported by the caller SBATCH script.
#
# Multi-GPU contract (same-node, no MPI across GPUs):
#   uma/kk : lmp -k on g ${NGPUS} -sf kk   with --ntasks=1
#   ASE/FC : FairChem workers=${NGPUS} in one process
# Never: srun -n ${NGPUS} / mpirun for uma/kk.

set -euo pipefail

: "${NGPUS:?NGPUS must be set (1|2|4)}"

ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
EX=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6
ENG=${ROOT}/src/ML-UMA/uma-engine
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
CONDA_SH=/u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
LMP_FC=${LMP_FC:-/u/xyan11/miniforge3-x86_64/envs/uma312/bin/lmp}
FIXED_XYZ=${ROOT}/src/ML-UMA/examples/delta_parity/structures/nacl6_rattle_fixed.extxyz

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8

source "${CONDA_SH}"
conda activate uma312

export UMA_CHECKPOINT="${CKPT}"
export LMP_FC
export LMP_UMA="${LMP_UMA:-${ROOT}/build-uma/lmp}"
export NGPUS
export UMA_DEVICES="${UMA_DEVICES:-${NGPUS}}"
export N_TIMING="${N_TIMING:-5}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Hint for run_multigpu.py
export UMA_KK_LAUNCH="${UMA_KK_LAUNCH:-lmp -k on g ${NGPUS} -sf kk; pair_style uma/kk ... devices ${UMA_DEVICES}}"
export FAIRCHEM_WORKERS="${FAIRCHEM_WORKERS:-${NGPUS}}"
export FIXED_STRUCTURE="${FIXED_STRUCTURE:-${FIXED_XYZ}}"
export RESULTS_DIR="${RESULTS_DIR:-${EX}/results/ngpu${NGPUS}}"

cd "${EX}"
echo "host=$(hostname)  NGPUS=${NGPUS}  UMA_DEVICES=${UMA_DEVICES}  ntasks=${SLURM_NTASKS:-?}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "UMA_KK_LAUNCH=${UMA_KK_LAUNCH}"
echo "FAIRCHEM_WORKERS=${FAIRCHEM_WORKERS}"
echo "FIXED_STRUCTURE=${FIXED_STRUCTURE}"
echo "RESULTS_DIR=${RESULTS_DIR}"
nvidia-smi -L || true

# Rebuild before jobs when WRITE lands C++ changes (gp_round stamp preferred).
# RECOMPILE=0 forces skip (use a prebuilt LMP_UMA); RECOMPILE=1 always rebuilds.
WRITE_STAMP="${EX}/gp_round/.write_agent_done.json"
if [[ ! -f "${WRITE_STAMP}" ]]; then
  WRITE_STAMP="${EX}/.write_agent_done.json"
fi
if [[ "${RECOMPILE:-0}" == "0" ]]; then
  echo "RECOMPILE=0 → skipping rebuild (using ${LMP_UMA})"
elif [[ -f "${WRITE_STAMP}" ]] && python -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if (d.get("rebuild_needed") or d.get("rebuild_required")) else 1)' \
      "${WRITE_STAMP}" 2>/dev/null; then
  echo "rebuild_needed=1 → building LAMMPS via scripts/build_lammps_uma.sh (${WRITE_STAMP})"
  bash "${ROOT}/scripts/build_lammps_uma.sh"
elif [[ "${RECOMPILE:-0}" == "1" ]]; then
  echo "RECOMPILE=1 → building LAMMPS via scripts/build_lammps_uma.sh"
  bash "${ROOT}/scripts/build_lammps_uma.sh"
fi

test -x "${LMP_UMA}" || { echo "ERROR: missing ${LMP_UMA}"; exit 1; }
test -f "${FIXED_XYZ}" || { echo "ERROR: missing fixed geometry ${FIXED_XYZ}"; exit 1; }
test -f "${ENG}/artifacts/uma-s-1p2-omat/model_traced.pt"
test -f "${ENG}/artifacts/uma-s-1p2-omat-f64/model_traced.pt"

mkdir -p "${RESULTS_DIR}"

if [[ ! -f "${EX}/run_multigpu.py" ]]; then
  echo "ERROR: run_multigpu.py not present yet (write agent unfinished)."
  echo "See ${EX}/.test_prep_pending_write"
  exit 2
fi

python "${EX}/run_multigpu.py"

# Optional post-job merge if all ngpu results exist (idempotent)
export RESULTS_PARENT="${RESULTS_PARENT:-${EX}/results}"
if [[ -f "${EX}/collect_results.py" ]]; then
  python "${EX}/collect_results.py" || true
fi
if [[ -f "${EX}/write_multigpu_reports.py" ]]; then
  python "${EX}/write_multigpu_reports.py" || true
fi

echo "DONE ngpu=${NGPUS}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
