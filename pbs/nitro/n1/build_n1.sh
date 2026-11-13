#!/bin/bash
# Login-node compile of the step 11 (N1-a) selftest. Does not run it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL="$(cd "${HERE}/../../.." && pwd)"
GCC_PREFIX="${GCC_PREFIX:-/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/gcc-13.4.0-22p27zg}"
export CC="${GCC_PREFIX}/bin/gcc"
export CXX="${GCC_PREFIX}/bin/g++"
FXPU_PREFIX="${FXPU_PREFIX:-/lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/fxpu}"
PY="${FXPU_PREFIX}/bin/python"
TORCH_CMAKE="$("${PY}" -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share','cmake'))")"
BLD="${REL}/pbs/out/n1_build"
mkdir -p "${BLD}"
cmake -S "${HERE}" -B "${BLD}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_C_COMPILER="${CC}" \
  -D CMAKE_CXX_COMPILER="${CXX}" \
  -D CMAKE_PREFIX_PATH="${TORCH_CMAKE}" \
  -D CMAKE_CXX_FLAGS="-O2"
cmake --build "${BLD}" -j 16
[[ -x "${BLD}/n1_selftest" ]]
echo "BUILD_OK ${BLD}/n1_selftest"
