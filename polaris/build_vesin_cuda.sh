#!/usr/bin/env bash
# Build vesin-torch (libvesin_torch.so) FROM SOURCE with CUDA enabled, then
# vendor it into src/ML-UMA/uma-engine/third_party/vesin/{lib,include}.
#
# Vesin's CUDA kernels are compiled at runtime via NVRTC (see upstream setup.py:
# "it does not matter which architecture we put here ... just linking against
# cudart"), so the build is GPU-arch independent and needs no -gencode. gpulite
# is bundled in the sdist (lib/vesin/external/gpulite.tar.gz) so no network is
# required. Run on a node that can import the uma312 torch.
#
# Usage:
#   VESIN_SRC=/path/to/vesin_torch-0.5.8 bash polaris/build_vesin_cuda.sh
# If VESIN_SRC is unset it downloads the sdist (needs network / login node).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/polaris/env_polaris.sh"

VESIN_VERSION="${VESIN_VERSION:-0.5.8}"
DEST="${ENG}/third_party/vesin"
WORK="${VESIN_WORK:-/tmp/vesin_build_$USER}"
mkdir -p "${WORK}"

# --- obtain source --------------------------------------------------------
SRC="${VESIN_SRC:-}"
if [[ -z "${SRC}" ]]; then
  echo ">>> download vesin-torch==${VESIN_VERSION} sdist"
  python -m pip download --no-deps --no-build-isolation --no-binary :all: \
    --no-input --progress-bar off -d "${WORK}/dl" "vesin-torch==${VESIN_VERSION}"
  tar xzf "${WORK}/dl"/vesin_torch-*.tar.gz -C "${WORK}"
  SRC="$(echo "${WORK}"/vesin_torch-*/)"
fi
SRC="$(cd "${SRC}" && pwd)"
echo "vesin-torch source: ${SRC}"
test -f "${SRC}/lib/CMakeLists.txt" || { echo "ERROR: bad SRC (no lib/CMakeLists.txt)"; exit 1; }
test -f "${SRC}/lib/vesin/external/gpulite.tar.gz" || echo "WARN: gpulite not bundled; build may need network"

# --- CUDA toolkit ---------------------------------------------------------
export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-/usr/local/cuda}}"
echo "CUDA_HOME=${CUDA_HOME}"
TORCH_CMAKE="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"

# --- configure + build (CUDA on) ------------------------------------------
BUILD="${WORK}/cmake-build"
INSTALL="${WORK}/install"
rm -rf "${BUILD}" "${INSTALL}"
mkdir -p "${BUILD}" "${INSTALL}"

cmake -S "${SRC}/lib" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC:-gcc-13}" \
  -DCMAKE_CXX_COMPILER="${CXX:-g++-13}" \
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL}" \
  -DBUILD_VESIN_FOR_PYTHON=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DVESIN_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT="${CUDA_HOME}" \
  -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DTORCH_CUDA_ARCH_LIST=8.0 \
  -DVESIN_INSTALL=ON
cmake --build "${BUILD}" -j"$(nproc)" --target install

# --- locate the built .so + header ---------------------------------------
SO="$(find "${INSTALL}" "${BUILD}" -name 'libvesin_torch.so' -type f | head -1)"
HDR="$(find "${SRC}" "${INSTALL}" -name 'vesin_torch.hpp' -type f | head -1)"
test -f "${SO}" || { echo "ERROR: libvesin_torch.so not produced"; exit 1; }
test -f "${HDR}" || { echo "ERROR: vesin_torch.hpp not found"; exit 1; }

mkdir -p "${DEST}/lib" "${DEST}/include"
cp -f "${SO}" "${DEST}/lib/libvesin_torch.so"
cp -f "${HDR}" "${DEST}/include/vesin_torch.hpp"
echo "${VESIN_VERSION}" > "${DEST}/VERSION"

echo "=== vendored ==="
file "${DEST}/lib/libvesin_torch.so"
echo "VESIN_CUDA_BUILD_OK ${DEST}/lib/libvesin_torch.so"
