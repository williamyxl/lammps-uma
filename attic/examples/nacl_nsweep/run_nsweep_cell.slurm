#!/bin/bash
#SBATCH --job-name=nsw-cell
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
# 112G matches every proven campaign run. Do not raise toward the 251G node
# limit: it makes the job fit only on a fully-idle node and start hours later
# (measured 23:09 -> 18:36 just by dropping 224G -> 112G).
#SBATCH --mem=112G
#SBATCH --time=01:30:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl_nsweep/logs/nsweep_n%a-%A.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl_nsweep/logs/nsweep_n%a-%A.out

# One N-sweep cell: structure -> per-N MP export -> LAMMPS NVT@300K on 4 GPUs
# -> (optional) ASE FP64 single-GPU ground-truth parity.
#
#   sbatch --export=ALL,NVAL=8 run_nsweep_cell.slurm
#
# OOM is an EXPECTED outcome near the ceiling: the cell records oom=true and
# exits 0 so the bisect driver can treat it as data, not as a broken job.
set -uo pipefail

ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
EX=${ROOT}/src/ML-UMA/examples/nacl_nsweep
N=${NVAL:?set NVAL}
NSTEPS=${NSTEPS:-10}
DEVICES=${UMA_DEVICES:-4}
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
# Per-N artifacts: MP TorchScript bakes partition offsets, so every N needs
# its own export. Kept out of the product artifact dir on purpose.
ART=${EX}/artifacts/nacl${N}
NATOMS=$((8 * N * N * N))
mkdir -p "${EX}/logs" "${EX}/results" "${ART}"

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID:-none} N=${N} natoms=${NATOMS} devices=${DEVICES} nsteps=${NSTEPS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Never rebuild here: the W17c wave owns build-uma with RECOMPILE=1 and two
# concurrent configures race the same CMake cache.
export RECOMPILE=0
export LMP_UMA=${LMP_UMA:-${ROOT}/build-uma/lmp}
export UMA_LIBTORCH_MP_WORKER=${UMA_LIBTORCH_MP_WORKER:-${ENG}/build-cpp-mp/uma_libtorch_mp_worker}
test -x "${LMP_UMA}" || { echo "NSWEEP_FAIL missing lmp ${LMP_UMA}"; exit 1; }
test -x "${UMA_LIBTORCH_MP_WORKER}" || { echo "NSWEEP_FAIL missing MP worker"; exit 1; }

export UMA_USE_KOKKOS=0
export UMA_PEER_TRANSPORT=nccl
export UMA_FORBID_RAY_GP=1
export UMA_ARTIFACT_DIR=${ART}
export UMA_MP_NATOMS=${NATOMS}
export UMA_STRUCTURE_NATOMS=${NATOMS}
export UMA_MP_LOG_DIR=${EX}/results/n${N}_${SLURM_JOB_ID}/mp_logs
mkdir -p "${UMA_MP_LOG_DIR}"

VESIN="${ENG}/third_party/vesin/lib"
TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${VESIN}:${TORCH_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# 1) structure -------------------------------------------------------------
python "${EX}/build_structure.py" --n "${N}" || {
  echo "NSWEEP_FAIL stage=structure N=${N}"; exit 1; }
XYZ=${EX}/structures/nacl${N}_rattle.extxyz
DATA=${EX}/structures/nacl${N}_nvt_300K_atomic_metal.data

# 2) per-N MP export (+ devices=1 traced for the parity oracle path) --------
if [[ ! -f "${ART}/model_mp_w${DEVICES}_n${NATOMS}_r0.pt" ]]; then
  cp -n "${ENG}/artifacts/uma-s-1p2-omat-f64-fast/metadata.json" "${ART}/" 2>/dev/null || true
  for R in $(seq 0 $((DEVICES - 1))); do
    echo "=== export N=${N} world=${DEVICES} rank=${R} ==="
    CUDA_VISIBLE_DEVICES=$R PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" \
      python "${ENG}/python/export_mp_artifact.py" \
        --checkpoint "${CKPT}" --output "${ART}" \
        --world "${DEVICES}" --ranks "${R}" \
        --dtype float64 --task omat --device cuda \
        --atoms "${XYZ}" \
        --execution-mode umas_fast_pytorch --merge-mole
    rc=$?
    if [[ $rc -ne 0 ]]; then
      python "${EX}/record_cell.py" --n "${N}" --stage export --status FAIL \
        --job "${SLURM_JOB_ID}" --note "export rank=${R} rc=${rc}"
      echo "NSWEEP_EXPORT_FAIL N=${N} rank=${R}"; exit 0
    fi
  done
fi
# Single-GPU traced export ONLY for N<=6. The traced export builds the whole
# graph on one GPU; N=6 (1728 atoms) is the measured maximum for one A100 40GB.
# At N=8 it OOMed (37.10 GiB allocated of 39.49) AFTER all four MP shards had
# already exported fine, so it was blocking the 4-GPU run for no reason.
# The MP path does not need model_traced.pt.
TRACED_MAX_N=${TRACED_MAX_N:-6}
if [[ "${N}" -le "${TRACED_MAX_N}" ]]; then
  test -f "${ART}/model_traced.pt" || \
    PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" python "${ENG}/python/w15_export_traced_fast.py" \
      --checkpoint "${CKPT}" --output "${ART}" --atoms "${XYZ}" \
      --dtype float64 --task omat --device cuda --out-name model_traced.pt || true
else
  echo "skip traced export: N=${N} > ${TRACED_MAX_N} (single-GPU OOM limit on A100 40GB)"
fi

# 3) LAMMPS NVT @300K on N GPUs -------------------------------------------
python "${EX}/run_cell.py" --n "${N}" --data "${DATA}" \
  --uma-devices "${DEVICES}" --nsteps "${NSTEPS}" --artifact "${ART}"
echo "NSWEEP_CELL_DONE N=${N} rc=$?"
