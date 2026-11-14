#!/bin/bash
# Login-node compile of the step 13 (S2) loader. Does not run it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL="$(cd "${HERE}/../../.." && pwd)"
GCC_PREFIX="${GCC_PREFIX:-/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/gcc-13.4.0-22p27zg}"
export CC="${GCC_PREFIX}/bin/gcc"
export CXX="${GCC_PREFIX}/bin/g++"
ICPX="${ICPX:-/opt/aurora/26.181.0/oneapi/compiler/latest/bin/icpx}"
ICPX_LIB="$(dirname "$(dirname "${ICPX}")")/lib"
FXPU_PREFIX="${FXPU_PREFIX:-/lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/fxpu}"
TORCH_CMAKE="${FXPU_PREFIX}/lib/python3.13/site-packages/torch/share/cmake"
BLD="${REL}/pbs/out/s2_build"
mkdir -p "${BLD}"
cmake -S "${HERE}" -B "${BLD}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_C_COMPILER="${CC}" \
  -D CMAKE_CXX_COMPILER="${CXX}" \
  -D CMAKE_PREFIX_PATH="${TORCH_CMAKE}" \
  -D UMA_ICPX="${ICPX}" \
  -D UMA_ICPX_LIBDIR="${ICPX_LIB}" \
  -D CONDA_PREFIX="${FXPU_PREFIX}"
cmake --build "${BLD}" -j 8
[[ -x "${BLD}/s2_test" ]]
echo "BUILD_OK ${BLD}/s2_test"
