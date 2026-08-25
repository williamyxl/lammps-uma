#!/bin/bash -l
# ---------------------------------------------------------------------------
# Minimal build of LAMMPS with the ML-UMA pair_style on Intel XPU (Aurora),
# native XCCL graph-parallel, FP64. Produces: ${LMP_BUILD}/lmp
#
# Toolchain: GCC 13.4 + Cray MPICH + torch 2.13.0+xpu (LibTorch) + oneCCL (icpx TU),
# forcing the conda libsycl.so.9 over the system libsycl.8.
#
# Usage:  bash build_lammps_uma.sh
# Edit LU / ACTIVATE / GCC / MPICH below for your checkout & environment.
# ---------------------------------------------------------------------------
set -e

# --- paths you may need to edit -------------------------------------------
LU=${LU:-/lus/flare/projects/MatSciAI/xiaoliyan/workdir/lammps-uma}   # lammps-uma checkout
ACTIVATE=${ACTIVATE:-/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/scripts/activate_fxpu.sh}  # activates torch+xpu conda env (fxpu/hen-xpu)
GCC=${GCC:-/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/gcc-13.4.0-hgnyg4p}
MPICH=${MPICH:-/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/mpich-5.0.0.aurora_test.3c70a61-hlkigtk}
# --------------------------------------------------------------------------

ENG=${LU}/src/ML-UMA/uma-engine
LMP_BUILD=${LMP_BUILD:-${LU}/build-lmp-xccl}

source "${ACTIVATE}"
module load cmake 2>/dev/null || true
export CC=$GCC/bin/gcc CXX=$GCC/bin/g++

TORCH_CMAKE=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share','cmake'))")
TORCH_LIB=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
SYCL_LIB_DIR=$CONDA_PREFIX/lib
SYCL_INC_DIR=$CONDA_PREFIX/include
echo "TORCH_CMAKE=$TORCH_CMAKE"
echo "ONECCL_ROOT=$CONDA_PREFIX  ICPX=$(which icpx)"

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
  2>&1 | tail -12

cmake --build "${LMP_BUILD}" -j 32 2>&1 | tail -12
if [ -x "${LMP_BUILD}/lmp" ]; then
  echo "LMP BUILD OK -> ${LMP_BUILD}/lmp"
  echo -n "PairUMA symbols: "; nm "${LMP_BUILD}/lmp" 2>/dev/null | grep -c PairUMA
  ldd "${LMP_BUILD}/lmp" 2>/dev/null | grep -iE "sycl|torch_xpu|libccl" | head
else
  echo "LMP BUILD FAILED"; exit 1
fi

# For a CUDA/NVIDIA build instead: drop UMA_ENGINE_USE_XPU/XCCL/SYCL_* and set
# -D UMA_ENGINE_USE_CUDA=ON with a CUDA LibTorch (NCCL is used for multi-GPU).
