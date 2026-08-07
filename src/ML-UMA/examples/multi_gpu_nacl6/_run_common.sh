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

# ---------------------------------------------------------------------------
# VRAM isolation policy: exactly ONE path per SLURM job.
# Multi-path ONLY_PATHS in one allocation contaminates GPU memory across
# ASE / FairChem Ray / uma Kokkos / LibTorch. Use run_ngpuN_<path>.slurm or
# ALLOW_MULTI_PATH=1 for rare login-node debugging only.
# ---------------------------------------------------------------------------
_only="${ONLY_PATHS:-}"
if [[ -z "${_only}" ]]; then
  echo "ERROR: ONLY_PATHS must be set to exactly one of: ase|fc|uma_double|uma_mixed" >&2
  echo "  (VRAM isolation — see README; ./submit_path_jobs.sh)" >&2
  exit 2
fi
# strip spaces; count comma-separated entries
_only_compact="${_only// /}"
if [[ "${_only_compact}" == *","* ]] && [[ "${ALLOW_MULTI_PATH:-0}" != "1" ]]; then
  echo "ERROR: ONLY_PATHS='${_only}' has multiple paths — refuse (VRAM isolation)." >&2
  echo "  Submit separate jobs (run_ngpu${NGPUS}_<path>.slurm) or ALLOW_MULTI_PATH=1." >&2
  exit 2
fi
if [[ "${ALLOW_MULTI_PATH:-0}" == "1" ]]; then
  echo "WARNING: ALLOW_MULTI_PATH=1 — multi-path in one job (VRAM contamination risk)"
fi
export MERGE_RESULTS="${MERGE_RESULTS:-1}"

ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
EX=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6
ENG=${ROOT}/src/ML-UMA/uma-engine
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
CONDA_SH=/u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
LMP_FC=${LMP_FC:-/u/xyan11/miniforge3-x86_64/envs/uma312/bin/lmp}
FIXED_XYZ=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz

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

# ---------------------------------------------------------------------------
# Sole timing source: SLURM wall of run_multigpu.py (not Python/Pair timers).
# Rebuild / collect_results / report writers stay outside the timed region.
# ---------------------------------------------------------------------------
export USE_SLURM_TIMING=1
echo "=== SLURM timing start (USE_SLURM_TIMING=1, N_TIMING=${N_TIMING}) ==="
_t0=$(date +%s.%N)
python "${EX}/run_multigpu.py"
_rc=$?
_t1=$(date +%s.%N)
if [[ "${_rc}" -ne 0 ]]; then
  echo "ERROR: run_multigpu.py failed rc=${_rc}" >&2
  exit "${_rc}"
fi
_wall=$(awk -v a="${_t0}" -v b="${_t1}" 'BEGIN { printf "%.9f", b - a }')
echo "=== SLURM timing end: wall_s=${_wall} ==="
python "${EX}/stamp_slurm_timing.py" \
  --results-dir "${RESULTS_DIR}" \
  --wall-s "${_wall}" \
  --n-timing "${N_TIMING}" \
  --paths "${ONLY_PATHS}"

# Optional post-job merge if all ngpu results exist (idempotent) — untimed
export RESULTS_PARENT="${RESULTS_PARENT:-${EX}/results}"
if [[ -f "${EX}/collect_results.py" ]]; then
  python "${EX}/collect_results.py" || true
fi
if [[ -f "${EX}/write_multigpu_reports.py" ]]; then
  python "${EX}/write_multigpu_reports.py" || true
fi

echo "DONE ngpu=${NGPUS} ONLY_PATHS=${ONLY_PATHS} slurm_wall_s=${_wall}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
