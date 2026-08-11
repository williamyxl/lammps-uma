#!/usr/bin/env bash
# Build LAMMPS with Kokkos CUDA + ML-UMA + MPI (multi-node / multi-rank path).
# Output: build-uma-mpi/lmp  (do NOT use for same-node Ray GP — that is build-uma/)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BUILD_DIR="${BUILD_DIR:-${ROOT}/build-uma-mpi}"
# Reuse the serial build script's cmake template but force MPI=ON via sed override:
# Prefer a dedicated cmake invocation (keeps build-uma pristine).

ENG="${ROOT}/src/ML-UMA/uma-engine"
LAMMPS_BUILD="${BUILD_DIR}"
JOBS="${JOBS:-$(nproc)}"

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: activate conda env with torch (uma312)" >&2
  exit 1
fi

TORCH_CMAKE="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
echo "Torch cmake prefix: ${TORCH_CMAKE}"
echo "Build dir: ${LAMMPS_BUILD}  (BUILD_MPI=ON)"

bash "${ENG}/scripts/vendor_vesin_for_arch.sh"
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
  -DBUILD_MPI=ON \
  -DBUILD_OMP=ON

cmake --build "${LAMMPS_BUILD}" -j"${JOBS}"

echo "Built: ${LAMMPS_BUILD}/lmp"
PAIR_UMA_SYMS="$(nm -C "${LAMMPS_BUILD}/liblammps.a" 2>/dev/null | grep -c 'T LAMMPS_NS::PairUMA::PairUMA' || true)"
if [[ "${PAIR_UMA_SYMS}" -lt 1 ]]; then
  echo "ERROR: PairUMA symbols missing from liblammps.a" >&2
  exit 1
fi
"${LAMMPS_BUILD}/lmp" -h 2>&1 | head -8 || true
