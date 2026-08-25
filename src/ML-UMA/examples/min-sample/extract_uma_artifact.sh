#!/bin/bash -l
# ---------------------------------------------------------------------------
# Extract a LAMMPS-loadable UMA artifact from the FairChem checkpoint.
#
# Produces a directory (ARTIFACT) containing:
#   model_traced.pt          TorchScript top module (energy; differentiable)
#   model_block_{i}.pt       per-message-passing-block sub-modules
#   model_chunk_{i}.pt       per-block edge-chunk sub-modules (activation ckpt)
#   model_edgedeg_chunk.pt   prologue edge-degree chunk module
#   metadata.json            cutoff, normalizer, element refs, num_blocks, ...
# pair_style uma points its pair_coeff at this directory.
#
# The artifact is TRACED AT A TARGET CELL SIZE N (chunk count is baked at N).
# Export one artifact per N you intend to run. Single-tile uses W=1; a 12-tile
# graph-parallel run needs per-rank artifacts under <ARTIFACT>/w{W}/r{R}/ (set
# EXPORT_WORLD=W and export each EXPORT_RANK).
#
# Runs on ONE XPU tile (tracing device). FP64.
#
# Usage:   N=6 bash extract_uma_artifact.sh                 # single-tile, N=6
#          N=32 EXPORT_WORLD=12 EXPORT_RANK=0 bash extract_uma_artifact.sh  # GP rank 0
# ---------------------------------------------------------------------------
set -e

LU=${LU:-/lus/flare/projects/MatSciAI/xiaoliyan/workdir/lammps-uma}
HEN=${HEN:-/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen}
ACTIVATE=${ACTIVATE:-${HEN}/scripts/activate_fxpu.sh}
ENG=${LU}/src/ML-UMA/uma-engine
UMA_CKPT=${UMA_CKPT:-${HEN}/uma-cache/uma-s-1p2.pt}   # FairChem UMA-s-1p2 checkpoint

N=${N:-6}                                             # NaCl NxNxN trace size (8*N^3 atoms)
ARTIFACT=${ARTIFACT:-${LU}/src/ML-UMA/examples/min-sample/artifact_n${N}}

source "${ACTIVATE}"
export PYTHONPATH="${ENG}/python:${HEN}/shim:${HEN}/patches:${HEN}:${PYTHONPATH:-}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export HF_HUB_OFFLINE=1 FAIRCHEM_OFFLINE=1 PYTHONUNBUFFERED=1
export UMA_CKPT UMA_TASK=${UMA_TASK:-omat}
# activation-checkpoint + neighbor knobs (defaults are the validated ones):
export FXPU_WIGNER_PREP_CHUNK=65536 FXPU_WIGNER_PREP_CHUNK_MODE=both
export EDGE_AC_CHUNK=${EDGE_AC_CHUNK:-16384}          # edge-chunk size (activation ckpt)
export UMA_EXPORT_CELL_LIST=1                         # O(N) neighbor build for the trace

mkdir -p "${ARTIFACT}"
echo "Exporting UMA artifact: N=${N}  world=${EXPORT_WORLD:-1} rank=${EXPORT_RANK:-0}"
echo "  ckpt=${UMA_CKPT}"
echo "  -> ${ARTIFACT}"
OUT="${ARTIFACT}" N_LIST=${N} TRACE_DEV=xpu RECONSTRUCT=${RECONSTRUCT:-1} \
  EXPORT_WORLD=${EXPORT_WORLD:-1} EXPORT_RANK=${EXPORT_RANK:-0} \
  python -u "${ENG}/python/export_blocks_xpu.py"

# For multi-tile GP, metadata must also sit at the artifact root (pair_uma reads it):
if [ "${EXPORT_WORLD:-1}" -gt 1 ] && [ -f "${ARTIFACT}/w${EXPORT_WORLD}/r0/metadata.json" ]; then
  cp -f "${ARTIFACT}/w${EXPORT_WORLD}/r0/metadata.json" "${ARTIFACT}/metadata.json"
fi
echo "Artifact ready:"; ls -1 "${ARTIFACT}" | sed 's/^/  /'
