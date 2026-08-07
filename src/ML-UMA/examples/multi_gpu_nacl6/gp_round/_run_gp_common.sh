#!/bin/bash
# gp_round shared body — graph-parallel uma/kk devices=N campaign.
# Sources parent _run_common.sh with gp_round defaults.

set -euo pipefail

GP_ROUND="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EX="$(cd "${GP_ROUND}/.." && pwd)"
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma

export RECOMPILE="${RECOMPILE:-1}"
export ONLY_PATHS="${ONLY_PATHS:-uma_double,uma_mixed}"
export UMA_DEVICES="${UMA_DEVICES:-${NGPUS}}"
export RESULTS_DIR="${RESULTS_DIR:-${EX}/results/gp_round/ngpu${NGPUS}}"
export RESULTS_PARENT="${RESULTS_PARENT:-${EX}/results/gp_round}"

echo "gp_round: ONLY_PATHS=${ONLY_PATHS} RECOMPILE=${RECOMPILE}"
echo "gp_round: RESULTS_DIR=${RESULTS_DIR}  RESULTS_PARENT=${RESULTS_PARENT}"

source "${EX}/_run_common.sh"
