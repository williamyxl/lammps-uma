#!/bin/bash
#SBATCH --job-name=v7-build
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/logs/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/logs/%x-%j.out

# V7 isolated build. NEVER writes build-uma/ or uma-engine/build-cpp-mp/:
# queued jobs (nsw-n8, n9-*) exec those binaries and must stay on the frozen
# 5d50357634 build for the whole campaign.
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
export BUILD_DIR=${ROOT}/build-uma-v7
export MP_BUILD_DIR=${ROOT}/src/ML-UMA/uma-engine/build-cpp-mp-v7

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID} V7 build -> ${BUILD_DIR}"
git -C "${ROOT}" rev-parse --short HEAD

bash "${ROOT}/scripts/build_lammps_uma.sh"

test -x "${BUILD_DIR}/lmp"
test -x "${MP_BUILD_DIR}/uma_libtorch_mp_worker"
# Guard: the frozen binaries must be byte-identical after this build.
md5sum "${ROOT}/build-uma/lmp" "${ROOT}/frozen/v6_5d50357634/lmp"
if ! cmp -s "${ROOT}/build-uma/lmp" "${ROOT}/frozen/v6_5d50357634/lmp"; then
  echo "V7_BUILD_FAIL frozen build-uma/lmp was modified"; exit 1
fi
echo "V7_BUILD_OK $(md5sum ${BUILD_DIR}/lmp | cut -d' ' -f1)"
