#!/bin/bash
# Extract a LAMMPS-loadable UMA artifact from a FairChem checkpoint.
#
# Requires:
#   UMA_CHECKPOINT  path to uma-s-1p2.pt (only extra-repo input)
# Optional:
#   N               NaCl NxNxN trace size (default 6)
#   ARTIFACT        output directory
set -euo pipefail

LU="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ENG="${LU}/src/ML-UMA/uma-engine"

if [ -z "${UMA_CHECKPOINT:-${UMA_CKPT:-}}" ]; then
  echo "ERROR: set UMA_CHECKPOINT to the FairChem UMA .pt file" >&2
  exit 1
fi
export UMA_CHECKPOINT="${UMA_CHECKPOINT:-$UMA_CKPT}"
if [ ! -f "${UMA_CHECKPOINT}" ]; then
  echo "ERROR: checkpoint not found: ${UMA_CHECKPOINT}" >&2
  exit 1
fi

N=${N:-6}
ARTIFACT=${ARTIFACT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/artifact_n${N}}

export PYTHONPATH="${ENG}/python:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export FAIRCHEM_OFFLINE="${FAIRCHEM_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export UMA_TASK="${UMA_TASK:-omat}"
export OUT="${ARTIFACT}"
export N_LIST="${N}"

python "${ENG}/python/export_blocks_xpu.py"
echo "Wrote artifact: ${ARTIFACT}"
