# Compute-node env for the lammps-uma-nitro harness (adapted from the Wave E
# pr_gate env.inc). No hen/ on PYTHONPATH. The checkpoint file is the only
# extra-repo input. Source after HERE= and REL= are set.
#
# Per-arm build dir: export NITRO_BUILD=<dir> before sourcing; the default is
# build-nitro/base. Never build into a directory shared with another arm.

module reset 2>/dev/null || true
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

FXPU_PREFIX="${FXPU_PREFIX:-/lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/fxpu}"
CONDA_ROOT="${CONDA_ROOT:-/lus/flare/projects/MOFA/xiaoliyan/software/miniforge3}"
if [ -r "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${FXPU_PREFIX}"
else
  export PATH="${FXPU_PREFIX}/bin:${PATH}"
  export CONDA_PREFIX="${FXPU_PREFIX}"
fi

ENG="${REL}/src/ML-UMA/uma-engine"
LMP_BUILD="${NITRO_BUILD:-${REL}/build-nitro/base}"
LMP="${LMP_BUILD}/lmp"
export PYTHONPATH="${ENG}/python"
unset UMA_HEN_ROOT
export UMA_CHECKPOINT="${UMA_CHECKPOINT:-/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/uma-cache/uma-s-1p2.pt}"
[[ -f "${UMA_CHECKPOINT}" ]] || { echo "FAIL: checkpoint missing ${UMA_CHECKPOINT}"; exit 1; }

export UMA_TASK=omat
export UMA_ALLOW_FAIRCHEM_MISMATCH=1
export UMA_ALLOW_LEGACY_METADATA=0
export TRACE_DEV=xpu
export RECONSTRUCT=0
export FXPU_WIGNER_PREP_CHUNK=65536
export FXPU_WIGNER_PREP_CHUNK_MODE=both
export UMA_EXPORT_CELL_LIST=1
export HF_HUB_OFFLINE=1 FAIRCHEM_OFFLINE=1 PYTHONUNBUFFERED=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset UMA_CKPT

# MPICH prefix. Override with MPICH=...; otherwise use the recorded Aurora fallback.
MPICH_FALLBACK="${MPICH_FALLBACK:-/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/mpich-5.0.0.aurora_test.3c70a61-v2spaxy}"
MPICH="${MPICH:-${MPICH_FALLBACK}}"
export MPICH MPICH_FALLBACK
if [[ ! -x "${MPICH}/bin/mpicc" ]]; then
  echo "NITRO_FAIL: MPICH not usable at ${MPICH} (set MPICH to a prefix with bin/mpicc)" >&2
  exit 1
fi

TORCH_LIB=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
ICPX_LIB=$(dirname "$(dirname "$(which icpx)")")/lib
# compat dir lives under pbs/out (git-excluded), not in the tracked tree
COMPAT="${REL}/pbs/out/.compat_${PBS_JOBID:-manual}"
mkdir -p "${COMPAT}"
ln -sf "${CONDA_PREFIX}/lib/libsycl.so.9" "${COMPAT}/libsycl.so.8"
SYS_FABRIC=$(ls -d /opt/cray/libfabric/*/lib64 2>/dev/null | sort -V | tail -1)
export LD_LIBRARY_PATH="${SYS_FABRIC}:${COMPAT}:${CONDA_PREFIX}/lib:${TORCH_LIB}:${ICPX_LIB}:${LD_LIBRARY_PATH:-}"
