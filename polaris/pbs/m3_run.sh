#!/usr/bin/env bash
# Shared M3 run body: launch build-uma-mn/lmp under mpiexec with one rank per GPU
# on NaCl 8x8x8 (4096 atoms), single point (E + per-atom F) + NVT-300K timing.
#
# Env in:
#   NRANKS   total MPI ranks (== total GPUs). MUST be >= 4 (4096 OOMs on <4 GPUs).
#   PPN      ranks per node (Polaris: 4)
#   TAG      output tag (r4 | r8)
#   RESULTS  results dir (shared across the 4- and 8-GPU jobs so parity can read both)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/polaris/env_polaris.sh"

: "${NRANKS:?}"; : "${PPN:?}"; : "${TAG:?}"; : "${RESULTS:?}"
NSTEPS="${M3_NVT_STEPS:-10}"
SYSTEM="${M3_SYSTEM:-nacl4096}"
LMP="${ROOT}/build-uma-mn/lmp"
M3="${ROOT}/src/ML-UMA/examples/polaris_m3"
export PYTHONPATH="${ROOT}/src/ML-UMA/examples/polaris_p0:${ENG}/python:${PYTHONPATH:-}"

if (( NRANKS < 4 )); then
  echo "M3_FAIL refusing NRANKS=${NRANKS}: NaCl 8x8x8 (4096) OOMs on <4 GPUs"; exit 1
fi
test -x "${LMP}" || { echo "M3_FAIL missing ${LMP} (run MG4)"; exit 1; }
test -f "${UMA_ARTIFACT_DIR}/model_traced.pt" || { echo "M3_FAIL missing artifact"; exit 1; }

mkdir -p "${RESULTS}"
WORK="${RESULTS}/work_${TAG}"
mkdir -p "${WORK}"

# Build the deck (single rank does this; deck references absolute paths).
python "${M3}/m3_lammps_mn.py" "${SYSTEM}" --work "${WORK}" --nsteps "${NSTEPS}" --build-input

# NCCL-over-MPI bootstrap hints (used by the engine's cross-node reductions).
export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$((29700 + RANDOM % 300))"
export UMA_FORBID_RAY_GP=1
# vesin + torch already on LD_LIBRARY_PATH via env_polaris.sh.

echo "=== mpiexec: ${NRANKS} ranks, ${PPN}/node, 1 GPU/rank, tag=${TAG} ==="
# Cray PALS: --ppn ranks/node; affinity wrapper pins PMI_LOCAL_RANK -> one GPU.
mpiexec -n "${NRANKS}" --ppn "${PPN}" \
  "${ROOT}/polaris/gpu_affinity_polaris.sh" \
  "${LMP}" -in "${WORK}/in.m3" -log "${WORK}/log.m3" > "${WORK}/out.m3" 2>&1
rc=$?
echo "mpiexec rc=${rc}"
tail -5 "${WORK}/out.m3" || true

# Parse (rank-agnostic; reads the single shared dump/log).
python "${M3}/m3_lammps_mn.py" "${SYSTEM}" --work "${WORK}" --nsteps "${NSTEPS}" --parse --tag "${TAG}"
# Copy the per-tag json/npz up to the shared results dir for the parity step.
cp "${WORK}/m3_${TAG}.json" "${RESULTS}/m3_${TAG}.json"
cp "${WORK}/m3_${TAG}.npz"  "${RESULTS}/m3_${TAG}.npz"
echo "M3_RUN_OK tag=${TAG} ranks=${NRANKS}"
