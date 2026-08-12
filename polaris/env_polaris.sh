#!/usr/bin/env bash
# Shared Polaris environment for the LAMMPS + LibTorch-UMA campaign.
#
# Source this from every PBS job:  source "${ROOT}/polaris/env_polaris.sh"
#
# It sets: repo ROOT/ENG, conda uma312, cuda module, LD_LIBRARY_PATH (vesin +
# torch/lib + cuda), UMA_CHECKPOINT, UMA_ARTIFACT_DIR. Everything is overridable
# from the environment so the same file works on a login node and a compute node.
#
# Polaris facts (ALCF): PBS Pro + Cray PALS `mpiexec`; 4x A100-40GB/node;
# per-rank env is PMI_RANK / PMI_LOCAL_RANK (NOT SLURM_*). No PMIx flag needed.

# --- repo layout ----------------------------------------------------------
# Resolve ROOT to the lammps-uma checkout regardless of where we're sourced.
if [[ -z "${ROOT:-}" ]]; then
  _env_self="${BASH_SOURCE[0]}"
  ROOT="$(cd "$(dirname "${_env_self}")/.." && pwd)"
fi
export ROOT
export ENG="${ROOT}/src/ML-UMA/uma-engine"

# --- conda (uma312) -------------------------------------------------------
export UMA_CONDA_SH="${UMA_CONDA_SH:-/lus/eagle/projects/RAPINS/xiaoliyan/polaris/software/miniforge3/etc/profile.d/conda.sh}"
export UMA_CONDA_ENV="${UMA_CONDA_ENV:-/lus/grand/projects/RAPINS/xiaoliyan/polaris/software/conda/envs/uma312}"
# shellcheck disable=SC1090
source "${UMA_CONDA_SH}"
conda activate "${UMA_CONDA_ENV}"

# --- CUDA module ----------------------------------------------------------
# Match the toolkit that torch was built against. torch 2.8 cu128 -> cuda 12.x.
# Polaris offers cuda/11.8 and cuda/12.9 (default). Override with UMA_CUDA_MODULE.
export UMA_CUDA_MODULE="${UMA_CUDA_MODULE:-cuda/12.9}"
module load "${UMA_CUDA_MODULE}" 2>/dev/null || true
# The cuda module does not reliably export CUDA_HOME under PrgEnv-nvidia, and a
# bare `nvcc` on PATH resolves to the nvhpc *compilers* dir (no libcudart).
# Resolve to a real toolkit: a dir with BOTH bin/nvcc AND lib64/libcudart.so.
# Prefer $CUDA_HOME if already valid, then the hpc_sdk cuda tree, then classic
# /usr/local/cuda. Override with UMA_CUDA_HOME.
_valid_cuda() { [[ -x "$1/bin/nvcc" && -e "$1/lib64/libcudart.so" ]]; }
_resolved=""
for _c in "${UMA_CUDA_HOME:-}" "${CUDA_HOME:-}" \
          /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/12.9 \
          /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/12.* \
          /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda \
          /usr/local/cuda-12.9 /usr/local/cuda; do
  [[ -n "${_c}" ]] || continue
  if _valid_cuda "${_c}"; then _resolved="$(cd "${_c}" && pwd)"; break; fi
done
if [[ -n "${_resolved}" ]]; then
  export CUDA_HOME="${_resolved}"
  export CUDAToolkit_ROOT="${_resolved}"
  # put the real toolkit bin AHEAD of nvhpc compilers so nvcc is the toolkit one
  export PATH="${_resolved}/bin:${PATH}"
else
  echo "[env_polaris] WARNING: no valid CUDA toolkit found (need bin/nvcc + lib64/libcudart.so)" >&2
fi

# --- host compiler (GCC) --------------------------------------------------
# torch 2.8 headers require GCC >= 13 constexpr semantics; the Cray default CC
# is nvc++ (fails torch static_asserts) and /usr/bin/gcc is 7.5 (too old).
# Load gcc-native/13 and pin CC/CXX so cmake/torch builds use GCC.
export UMA_GCC_MODULE="${UMA_GCC_MODULE:-gcc-native/13}"
module load "${UMA_GCC_MODULE}" 2>/dev/null || true
# Loading gcc-native swaps PrgEnv and UNSETS CUDA_HOME; CUDAToolkit_ROOT (a
# plain var the module system does not touch) survives, so re-assert from it.
if [[ -z "${CUDA_HOME:-}" && -n "${CUDAToolkit_ROOT:-}" ]]; then
  export CUDA_HOME="${CUDAToolkit_ROOT}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
# gcc-native provides gcc-13/g++-13; prefer explicit names, fall back to gcc.
if command -v g++-13 >/dev/null 2>&1; then
  export CC="${CC:-gcc-13}" CXX="${CXX:-g++-13}"
elif command -v g++ >/dev/null 2>&1 && [[ "$(g++ -dumpversion 2>/dev/null | cut -d. -f1)" -ge 13 ]]; then
  export CC="${CC:-gcc}" CXX="${CXX:-g++}"
fi

# --- UMA checkpoint + artifact -------------------------------------------
export UMA_CHECKPOINT="${UMA_CHECKPOINT:-/lus/eagle/projects/RAPINS/xiaoliyan/polaris/uma-s-1p2.pt}"
export UMA_ARTIFACT_DIR="${UMA_ARTIFACT_DIR:-${ENG}/artifacts/uma-s-1p2-omat-f64}"

# --- LD_LIBRARY_PATH (vesin + torch + cuda) ------------------------------
_vesin="${ENG}/third_party/vesin/lib"
_torch_lib="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null || true)"
_cuda_lib="${CUDA_HOME:-/usr/local/cuda}/lib64"
export LD_LIBRARY_PATH="${_vesin}:${_torch_lib}:${_cuda_lib}:${LD_LIBRARY_PATH:-}"

# Single-node product path: never fork Ray GP; NCCL only when devices>1.
export UMA_FORBID_RAY_GP="${UMA_FORBID_RAY_GP:-1}"

# NCCL for the multi-node edge-parallel path. The conda torch ships libnccl.so.2
# (versioned only) under nvidia/nccl; we vendor an unversioned symlink + header
# under the engine so cmake find_library(nccl) succeeds. Also put it on
# LD_LIBRARY_PATH for runtime.
export NCCL_ROOT="${NCCL_ROOT:-${ENG}/third_party/nccl}"
_nccl_real="$(python -c 'import os,glob;import nvidia.nccl as n;print(os.path.join(os.path.dirname(n.__file__),"lib"))' 2>/dev/null || true)"
if [[ -n "${_nccl_real}" ]]; then
  export LD_LIBRARY_PATH="${_nccl_real}:${LD_LIBRARY_PATH}"
fi

echo "[env_polaris] ROOT=${ROOT}"
echo "[env_polaris] conda=${CONDA_PREFIX}"
echo "[env_polaris] cuda module=${UMA_CUDA_MODULE}  torch_lib=${_torch_lib}"
echo "[env_polaris] UMA_CHECKPOINT=${UMA_CHECKPOINT}"
echo "[env_polaris] UMA_ARTIFACT_DIR=${UMA_ARTIFACT_DIR}"
