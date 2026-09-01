#!/bin/bash
# Per-rank wrapper for 12-tile data-parallel LAMMPS UMA on Aurora.
# Pins this rank to one XPU tile (from the PALS local rank) and runs an
# independent NVT in a per-rank directory. Args: <lmp> <in_file> <base_rundir>
set -e
LMP="$1"; INFILE="$2"; BASE="$3"

# Local rank id (PALS on Aurora; fall back to MPI/PMI vars).
LR="${PALS_LOCAL_RANKID:-${PMI_LOCAL_RANK:-${MPI_LOCALRANKID:-0}}}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK="${LR}"        # one flat tile per rank (0..11)
export UMA_EAGER_CKPT=1

RUNDIR="${BASE}/rank${LR}"
mkdir -p "${RUNDIR}"
cp -f "${BASE}/data.nacl" "${RUNDIR}/" 2>/dev/null || true
cp -f "${INFILE}" "${RUNDIR}/in.nvt"
cd "${RUNDIR}"
echo "rank ${LR}: tile ${ZE_AFFINITY_MASK} rundir ${RUNDIR}" >&2
exec "${LMP}" -in in.nvt
