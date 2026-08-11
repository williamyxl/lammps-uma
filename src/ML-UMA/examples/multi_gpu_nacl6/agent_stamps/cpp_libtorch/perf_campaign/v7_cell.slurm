#!/bin/bash
#SBATCH --job-name=v7-cell
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=00:50:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/logs/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/logs/%x-%j.out

# One V7 cell: SP + NVT(300K) on the V7 binaries only.
#   sbatch --gpus-per-node=N --job-name=v7-<wave>-<sys>-n<N> \
#          --export=ALL,WAVE=w18,SYS=nacl6,NG=N v7_cell.slurm
set -uo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
CAMP=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign
WAVE=${WAVE:?set WAVE}
SYS=${SYS:?set SYS=nacl6|water888}
NG=${NG:?set NG}

case "$SYS" in
  nacl6)    EX=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6; NAT=1728; NST=${NSTEPS:-10} ;;
  water888) EX=${ROOT}/src/ML-UMA/examples/water888;        NAT=648;  NST=${NSTEPS:-10} ;;
  *) echo "bad SYS=$SYS"; exit 1 ;;
esac

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

# V7 binaries only. RECOMPILE=0 always: the build is a separate job, so a cell
# can never race the build or touch the frozen v6 binaries.
export RECOMPILE=0
export LMP_UMA=${ROOT}/build-uma-v7/lmp
export UMA_LIBTORCH_MP_WORKER=${ENG}/build-cpp-mp-v7/uma_libtorch_mp_worker
test -x "${LMP_UMA}" || { echo "V7_CELL_FAIL missing ${LMP_UMA}"; exit 1; }
test -x "${UMA_LIBTORCH_MP_WORKER}" || { echo "V7_CELL_FAIL missing worker"; exit 1; }

export UMA_USE_KOKKOS=0
export UMA_ARTIFACT_DIR=${ENG}/artifacts/uma-s-1p2-omat-f64-fast
export NGPUS=${NG} UMA_DEVICES=${NG} FAIRCHEM_WORKERS=${NG} NSTEPS=${NST}
export UMA_MP_NATOMS=${NAT} UMA_STRUCTURE_NATOMS=${NAT}
export UMA_MP_LOG_DIR=${CAMP}/matrix/v7_${WAVE}_${SYS}_ngpu${NG}/mp_logs
mkdir -p "${UMA_MP_LOG_DIR}" "${CAMP}/logs"

echo "JOBID=${SLURM_JOB_ID} V7 wave=${WAVE} sys=${SYS} ngpu=${NG} nsteps=${NST}"
echo "lmp=${LMP_UMA} md5=$(md5sum ${LMP_UMA} | cut -d' ' -f1)"

cd "${EX}"
python run_path_uma.py --ngpus "${NG}" --uma-devices "${NG}" --nsteps "${NST}"
RC=$?
echo "V7_CELL_DONE wave=${WAVE} sys=${SYS} ngpu=${NG} rc=${RC}"
