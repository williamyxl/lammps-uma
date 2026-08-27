#!/bin/bash -l
# Export a k=4 DD single-tile UMA artifact for spatial domain decomposition.
#
# Differs from a normal single-tile export:
#   UMA_DD_HALO=1        inject uma_halo::exchange before each block + trace the
#                        NodeEnergyExportWrapper (top returns (node_energy, total))
#   UMA_DD_EDGE_CAP=C    pad the trace edge list to C so the baked chunk count is
#                        rank-invariant; the RUNTIME must set the SAME cap.
#
# One artifact serves every DD rank (single-tile, world=1). Trace at any N whose
# real edge count <= cap (the pad fills the rest); N need not equal the run N.
#
#   UMA_DD_EDGE_CAP=917504 N=8 bash export_dd_artifact.sh
# ---------------------------------------------------------------------------
set -e
LU=${LU:-/lus/flare/projects/MatSciAI/xiaoliyan/workdir/lammps-uma}
HEN=${HEN:-/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen}
ACTIVATE=${ACTIVATE:-${HEN}/scripts/activate_fxpu.sh}
ENG=${LU}/src/ML-UMA/uma-engine
UMA_CKPT=${UMA_CKPT:-${HEN}/uma-cache/uma-s-1p2.pt}

N=${N:-8}                                        # trace NaCl size (edges must be <= cap)
EDGE_AC_CHUNK=${EDGE_AC_CHUNK:-65536}
UMA_DD_EDGE_CAP=${UMA_DD_EDGE_CAP:-917504}       # 14 x 65536; covers 2-node N=32 worst rank
ARTIFACT=${ARTIFACT:-${LU}/scripts/out/dd/n32_k4_cap${UMA_DD_EDGE_CAP}}

source "${ACTIVATE}"
export PYTHONPATH="${ENG}/python:${HEN}/shim:${HEN}/patches:${HEN}:${PYTHONPATH:-}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export HF_HUB_OFFLINE=1 FAIRCHEM_OFFLINE=1 PYTHONUNBUFFERED=1
export UMA_CKPT UMA_TASK=${UMA_TASK:-omat}
export FXPU_WIGNER_PREP_CHUNK=65536 FXPU_WIGNER_PREP_CHUNK_MODE=both
export EDGE_AC_CHUNK
export UMA_EXPORT_CELL_LIST=1
export UMA_DD_HALO=1                              # <-- k=4 halo + node-energy wrapper
export UMA_DD_EDGE_CAP                            # <-- fixed trace edge/chunk cap

mkdir -p "${ARTIFACT}"
echo "Exporting DD k=4 artifact: trace N=${N}  cap=${UMA_DD_EDGE_CAP}  chunk=${EDGE_AC_CHUNK}"
echo "  -> ${ARTIFACT}"
# RECONSTRUCT off: the reconstruct check assumes the scalar-energy wrapper; the
# DD node-energy tuple output needs a DD-aware check (TODO). Validate via the
# 2-node parity run instead.
OUT="${ARTIFACT}" N_LIST=${N} TRACE_DEV=xpu RECONSTRUCT=0 \
  EXPORT_WORLD=1 EXPORT_RANK=0 \
  python -u "${ENG}/python/export_blocks_xpu.py"

echo "Artifact ready:"; ls -1 "${ARTIFACT}" | sed 's/^/  /'
echo "metadata dd_halo:"; grep -E '"dd_halo"|"dd_k"|"returns_node_energy"|"edge_ac_chunk"' "${ARTIFACT}/metadata.json" || true
