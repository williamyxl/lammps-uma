#!/usr/bin/env bash
# Build LAMMPS with Kokkos CUDA + ML-UMA (LibTorch) on Delta.
# Prerequisites: conda env uma312 active; modules cuda/12.8 cmake loaded;
#                vesin vendored; TorchScript artifacts exported (or export later).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENG="${ROOT}/src/ML-UMA/uma-engine"
# NOTE: do not name this BUILD — module load cuda/conda may set BUILD=x86_64-conda-linux-gnu.
LAMMPS_BUILD="${BUILD_DIR:-${ROOT}/build-uma}"
JOBS="${JOBS:-$(nproc)}"

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: activate conda env with torch (uma312)" >&2
  exit 1
fi

TORCH_CMAKE="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
echo "Torch cmake prefix: ${TORCH_CMAKE}"
echo "Build dir: ${LAMMPS_BUILD}"

# Vendor vesin for this arch if missing
bash "${ENG}/scripts/vendor_vesin_for_arch.sh"

# Ensure deep object dirs exist (absolute-source cmake layout)
mkdir -p "${LAMMPS_BUILD}/CMakeFiles/lammps.dir${ROOT}/src/ML-UMA" \
         "${LAMMPS_BUILD}/CMakeFiles/lammps.dir${ROOT}/src/KOKKOS"

cmake -S "${ROOT}/cmake" -B "${LAMMPS_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE}" \
  -DPKG_KOKKOS=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DPKG_ML-UMA=ON \
  -DUMA_ENGINE_ROOT="${ENG}" \
  -DBUILD_MPI=OFF \
  -DBUILD_OMP=ON

cmake --build "${LAMMPS_BUILD}" -j"${JOBS}"

# Process-per-rank MP worker (EXCLUDE_FROM_ALL under LAMMPS add_subdirectory).
# Prefer building into uma-engine/build-cpp-mp so UMA_LIBTORCH_MP_WORKER is stable.
# MP_BUILD_DIR overrides it so an experimental build (V7) cannot clobber the
# worker binary that already-queued jobs are going to exec.
ENG_MP_BUILD="${MP_BUILD_DIR:-${ENG}/build-cpp-mp}"
mkdir -p "${ENG_MP_BUILD}"
cmake -S "${ENG}" -B "${ENG_MP_BUILD}" \
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE}" \
  -DUMA_ENGINE_USE_CUDA=ON
cmake --build "${ENG_MP_BUILD}" -j"${JOBS}" --target uma_libtorch_mp_worker
# Also try building the worker in the LAMMPS tree (best-effort).
cmake --build "${LAMMPS_BUILD}" -j"${JOBS}" --target uma_libtorch_mp_worker 2>/dev/null || true
echo "MP worker: ${ENG_MP_BUILD}/uma_libtorch_mp_worker"
ls -la "${ENG_MP_BUILD}/uma_libtorch_mp_worker"

echo "Built: ${LAMMPS_BUILD}/lmp"
# Sanity: PairUMA must be in the archive (avoids silent incomplete ar races).
# Use grep -c (reads all input) — grep -q + pipefail falsely fails on SIGPIPE when
# nm is closed early after the first match.
PAIR_UMA_SYMS="$(nm -C "${LAMMPS_BUILD}/liblammps.a" 2>/dev/null | grep -c 'T LAMMPS_NS::PairUMA::PairUMA' || true)"
if [[ "${PAIR_UMA_SYMS}" -lt 1 ]]; then
  echo "ERROR: PairUMA symbols missing from liblammps.a — incomplete link" >&2
  exit 1
fi
"${LAMMPS_BUILD}/lmp" -h 2>&1 | head -5 || true
