#!/usr/bin/env bash
# Multi-node (MPI) LAMMPS + ML-UMA build — ISOLATED OUTPUT DIRS.
#
# Build isolation is a hard requirement: queued and running jobs exec the
# existing binaries, and rebuilding in place under them silently changes what
# they run. Every artifact this script produces lives in dedicated -mn dirs:
#
#     build-uma-mn/lmp                              (this build)
#     uma-engine/build-cpp-mp-mn/uma_libtorch_mp_worker
#
# Never written by this script:
#     build-uma/          frozen v6 5d50357634, used by nsw-* and n9-* jobs
#     build-uma-v7/       V7 campaign
#     uma-engine/build-cpp-mp{,-v7}/
#
# The script asserts those stay byte-identical and aborts if they move.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENG="${ROOT}/src/ML-UMA/uma-engine"
export BUILD_DIR="${BUILD_DIR:-${ROOT}/build-uma-mn}"
export MP_BUILD_DIR="${MP_BUILD_DIR:-${ENG}/build-cpp-mp-mn}"
JOBS="${JOBS:-$(nproc)}"

# --- refuse to clobber protected trees ------------------------------------
for protected in "${ROOT}/build-uma" "${ROOT}/build-uma-v7" \
                 "${ENG}/build-cpp-mp" "${ENG}/build-cpp-mp-v7"; do
  if [[ "${BUILD_DIR}" == "${protected}" || "${MP_BUILD_DIR}" == "${protected}" ]]; then
    echo "REFUSING: build dir '${protected}' is in use by other jobs" >&2
    exit 1
  fi
done

# --- fingerprint protected binaries before ---------------------------------
declare -A BEFORE
for b in "${ROOT}/build-uma/lmp" "${ROOT}/build-uma-v7/lmp" \
         "${ENG}/build-cpp-mp/uma_libtorch_mp_worker" \
         "${ENG}/build-cpp-mp-v7/uma_libtorch_mp_worker"; do
  [[ -f "$b" ]] && BEFORE["$b"]="$(md5sum "$b" | cut -d' ' -f1)"
done

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: activate a conda env with torch (uma312)" >&2
  exit 1
fi
TORCH_CMAKE="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
echo "Torch cmake prefix: ${TORCH_CMAKE}"
echo "LAMMPS build dir  : ${BUILD_DIR}   (BUILD_MPI=ON)"
echo "MP worker dir     : ${MP_BUILD_DIR}"

bash "${ENG}/scripts/vendor_vesin_for_arch.sh"
mkdir -p "${BUILD_DIR}/CMakeFiles/lammps.dir${ROOT}/src/ML-UMA" \
         "${BUILD_DIR}/CMakeFiles/lammps.dir${ROOT}/src/KOKKOS" \
         "${MP_BUILD_DIR}"

cmake -S "${ROOT}/cmake" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE}" \
  -DPKG_KOKKOS=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DPKG_ML-UMA=ON \
  -DUMA_ENGINE_ROOT="${ENG}" \
  -DBUILD_MPI=ON \
  -DBUILD_OMP=ON
cmake --build "${BUILD_DIR}" -j"${JOBS}"

cmake -S "${ENG}" -B "${MP_BUILD_DIR}" \
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE}" \
  -DUMA_ENGINE_USE_CUDA=ON
cmake --build "${MP_BUILD_DIR}" -j"${JOBS}" --target uma_libtorch_mp_worker

# --- verify protected binaries did not move --------------------------------
rc=0
for b in "${!BEFORE[@]}"; do
  now="$(md5sum "$b" | cut -d' ' -f1)"
  if [[ "$now" != "${BEFORE[$b]}" ]]; then
    echo "CONTAMINATION: $b changed (${BEFORE[$b]} -> $now)" >&2
    rc=1
  else
    echo "  intact: $b"
  fi
done
[[ $rc -eq 0 ]] || { echo "MN_BUILD_FAIL protected binaries modified" >&2; exit 1; }

test -x "${BUILD_DIR}/lmp"
"${BUILD_DIR}/lmp" -h 2>&1 | grep -iE 'MPI v|MPI STUBS' | head -2
echo "MN_BUILD_OK $(md5sum "${BUILD_DIR}/lmp" | cut -d' ' -f1)"
