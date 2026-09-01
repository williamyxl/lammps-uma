#!/bin/bash -l
# Build LAMMPS (no Kokkos) with ML-UMA on Intel XPU + native XCCL graph-parallel.
# pair_style uma (non-kk) + uma-engine XPU build with oneCCL (icpx TU). GCC 13.4
# + Cray MPICH + torch+xpu libtorch, forcing conda libsycl.so.9.
set -e

HEN_ROOT=/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen
LU=/lus/flare/projects/MatSciAI/xiaoliyan/workdir/lammps-uma
ENG=${LU}/src/ML-UMA/uma-engine
LMP_BUILD=${LU}/build-lmp-xccl

source "${HEN_ROOT}/scripts/activate_fxpu.sh"
module load cmake 2>/dev/null || true

GCC=/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/gcc-13.4.0-hgnyg4p
MPICH=/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/mpich-5.0.0.aurora_test.3c70a61-hlkigtk
export CC=$GCC/bin/gcc CXX=$GCC/bin/g++

TORCH_CMAKE=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share','cmake'))")
TORCH_LIB=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
SYCL_LIB_DIR=$CONDA_PREFIX/lib
SYCL_INC_DIR=$CONDA_PREFIX/include

echo "TORCH_CMAKE=$TORCH_CMAKE  MPICH=$MPICH"
rm -rf "${LMP_BUILD}" && mkdir -p "${LMP_BUILD}"

cmake -S "${LU}/cmake" -B "${LMP_BUILD}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=ON -D BUILD_OMP=ON \
  -D CMAKE_C_COMPILER=$GCC/bin/gcc \
  -D CMAKE_CXX_COMPILER=$GCC/bin/g++ \
  -D CMAKE_CXX_STANDARD=17 \
  -D MPI_CXX_COMPILER=$MPICH/bin/mpicxx \
  -D MPI_C_COMPILER=$MPICH/bin/mpicc \
  -D CMAKE_PREFIX_PATH="$TORCH_CMAKE" \
  -D PKG_ML-UMA=ON \
  -D UMA_ENGINE_USE_XPU=ON \
  -D UMA_ENGINE_USE_XCCL=ON \
  -D UMA_ICPX="$(which icpx)" \
  -D ONECCL_ROOT="$CONDA_PREFIX" \
  -D UMA_ENGINE_ROOT="${ENG}" \
  -D SYCL_LIBRARY=$SYCL_LIB_DIR/libsycl.so \
  -D SYCL_LIBRARY_DIR=$SYCL_LIB_DIR \
  -D SYCL_INCLUDE_DIR=$SYCL_INC_DIR \
  -D SYCL_INCLUDE_SYCL_DIR=$SYCL_INC_DIR/sycl \
  -D CMAKE_CXX_FLAGS="-O2 -Wno-tautological-compare" \
  -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$SYCL_LIB_DIR -Wl,-rpath,$TORCH_LIB -Wl,-rpath,$MPICH/lib" \
  2>&1 | tail -15

cmake --build "${LMP_BUILD}" -j 32 2>&1 | tail -15
if [ -x "${LMP_BUILD}/lmp" ]; then
  echo "LMP BUILD OK"
  nm "${LMP_BUILD}/lmp" 2>/dev/null | grep -c PairUMA && echo "PairUMA present"
  ldd "${LMP_BUILD}/lmp" 2>/dev/null | grep -iE "sycl|torch_xpu" | head
else
  echo "LMP BUILD FAILED"; exit 1
fi
