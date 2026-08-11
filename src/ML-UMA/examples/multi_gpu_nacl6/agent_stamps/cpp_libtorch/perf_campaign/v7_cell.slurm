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

# merge_mole fuses MoLE experts for ONE fixed composition, so each system needs
# the artifact exported for it. Using the NaCl-fused -f64-fast for water gives a
# physically wrong energy (measured: -3157.15 vs oracle -3143.39, dE 13.8 eV).
case "$SYS" in
  nacl6)    EX=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6; NAT=1728; NST=${NSTEPS:-10}
            ART=${ENG}/artifacts/uma-s-1p2-omat-f64-fast ;;
  # water: the devices=1 traced model lives in -water648, but the multi-GPU
  # shards (model_mp_w{2,4}_n648_r*) are in -f64-fast, which is what the locked
  # W8nk water bars used and which reproduces the oracle energy exactly.
  water888) EX=${ROOT}/src/ML-UMA/examples/water888;        NAT=648;  NST=${NSTEPS:-10}
            if [ "${NG}" = "1" ]; then
              ART=${ENG}/artifacts/uma-s-1p2-omat-f64-fast-water648
            else
              ART=${ENG}/artifacts/uma-s-1p2-omat-f64-fast
            fi ;;
  *) echo "bad SYS=$SYS"; exit 1 ;;
esac
test -f "${ART}/model_traced.pt" || { echo "V7_CELL_FAIL missing ${ART}/model_traced.pt"; exit 1; }

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
export UMA_ARTIFACT_DIR=${ART}
export NGPUS=${NG} UMA_DEVICES=${NG} FAIRCHEM_WORKERS=${NG} NSTEPS=${NST}
export UMA_MP_NATOMS=${NAT} UMA_STRUCTURE_NATOMS=${NAT}
export UMA_MP_LOG_DIR=${CAMP}/matrix/v7_${WAVE}_${SYS}_ngpu${NG}/mp_logs
mkdir -p "${UMA_MP_LOG_DIR}" "${CAMP}/logs"

echo "JOBID=${SLURM_JOB_ID} V7 wave=${WAVE} sys=${SYS} ngpu=${NG} nsteps=${NST}"
echo "lmp=${LMP_UMA} md5=$(md5sum ${LMP_UMA} | cut -d' ' -f1)"

cd "${EX}"

# ---------------------------------------------------------------------------
# External (shell-level) timing. Measured here in the SLURM script around the
# bash command, NOT from inside LAMMPS or the Python/C++ code, so the number is
# independent of any in-code counter. LAMMPS' own Pair/Loop timers are still
# parsed into timing.json for cross-checking, but SHELL_* below is the
# authoritative wall time for this cell.
#
# Differential method. Each invocation runs: run 0 (warmup+dump) + run 1 (NVE
# SP probe) + run NSTEPS (NVT). Two passes at different NSTEPS; everything
# except the NVT block is common and cancels in the difference:
#   pass A: NSTEPS=NA        pass B: NSTEPS=NB
#   ms/step = (tB - tA) * 1000 / (NB - NA)
# Divide by the STEP DELTA, not NB: pass A already performs NA NVT steps.
#
# NB must be large enough that the NVT block dominates run-to-run startup
# jitter. At NB=10 the signal was only ~5% of an 11 s wall, which is why the
# shell number disagreed with the LAMMPS Pair timer by ~2.6x.
# ---------------------------------------------------------------------------
NA=${NSTEPS_A:-2}
NB=${NSTEPS_B:-$(( NST > 50 ? NST : 50 ))}
DN=$(( NB - NA ))

T0=$(date +%s.%N)
python run_path_uma.py --ngpus "${NG}" --uma-devices "${NG}" --nsteps "${NA}" \
  > "$(dirname ${UMA_MP_LOG_DIR})/passA.out" 2>&1
RC_A=$?
T1=$(date +%s.%N)

python run_path_uma.py --ngpus "${NG}" --uma-devices "${NG}" --nsteps "${NB}"
RC=$?
T2=$(date +%s.%N)

SHELL_A_S=$(echo "$T1 - $T0" | bc -l)
SHELL_B_S=$(echo "$T2 - $T1" | bc -l)
SHELL_NVT_MS_PER_STEP=$(echo "($SHELL_B_S - $SHELL_A_S) * 1000 / ${DN}" | bc -l)

printf 'SHELL_TIMING wave=%s sys=%s ngpu=%s nsteps_a=%d nsteps_b=%d dn=%d pass_a_s=%.3f pass_b_s=%.3f nvt_ms_per_step_shell=%.3f rc_a=%d rc=%d\n' \
  "${WAVE}" "${SYS}" "${NG}" "${NA}" "${NB}" "${DN}" "${SHELL_A_S}" "${SHELL_B_S}" \
  "${SHELL_NVT_MS_PER_STEP}" "${RC_A}" "${RC}"
echo "V7_CELL_DONE wave=${WAVE} sys=${SYS} ngpu=${NG} rc=${RC}"
