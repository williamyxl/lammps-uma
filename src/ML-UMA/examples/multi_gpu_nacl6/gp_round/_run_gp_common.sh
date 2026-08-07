#!/bin/bash
# gp_round shared body — graph-parallel uma/kk devices=N campaign.
# Sources parent _run_common.sh with gp_round defaults.

set -euo pipefail

GP_ROUND="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EX="$(cd "${GP_ROUND}/.." && pwd)"
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma

export RECOMPILE="${RECOMPILE:-1}"
export ONLY_PATHS="${ONLY_PATHS:?ONLY_PATHS required: uma_double or uma_mixed (one path)}"
export UMA_DEVICES="${UMA_DEVICES:-${NGPUS}}"
export RESULTS_DIR="${RESULTS_DIR:-${EX}/results/gp_round/ngpu${NGPUS}}"
export RESULTS_PARENT="${RESULTS_PARENT:-${EX}/results/gp_round}"
export MERGE_RESULTS="${MERGE_RESULTS:-1}"

# Enforce single-path (same policy as parent _run_common.sh).
_only_compact="${ONLY_PATHS// /}"
if [[ "${_only_compact}" == *","* ]] && [[ "${ALLOW_MULTI_PATH:-0}" != "1" ]]; then
  echo "ERROR: gp_round ONLY_PATHS='${ONLY_PATHS}' must be a single path (VRAM isolation)." >&2
  echo "  Use gp_round/run_ngpu${NGPUS}_uma_double.slurm or ..._uma_mixed.slurm" >&2
  exit 2
fi

echo "gp_round: ONLY_PATHS=${ONLY_PATHS} RECOMPILE=${RECOMPILE}"
echo "gp_round: RESULTS_DIR=${RESULTS_DIR}  RESULTS_PARENT=${RESULTS_PARENT}"

source "${EX}/_run_common.sh"
