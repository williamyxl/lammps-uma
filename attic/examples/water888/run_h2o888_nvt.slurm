#!/bin/bash
#SBATCH --job-name=uma-h2o888-nvt
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/water888/logs/h2o888_nvt-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/water888/logs/h2o888_nvt-%j.out

# Flexible H2O NVT @ 300 K, 10 steps (data file includes Velocities).
# Timing = SLURM wall around the single lmp launch only (TEST_PROTOCOL.md).
#
#   sbatch --export=ALL,RECOMPILE=1,NGPUS=1,UMA_DEVICES=1 run_h2o888_nvt.slurm

set -euo pipefail

EX="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT="$(cd "${EX}/../../../.." && pwd)"
ENG="${ROOT}/src/ML-UMA/uma-engine"
ART="${ENG}/artifacts/uma-s-1p2-omat-f64"
DATA="${EX}/water_nvt_300K_atomic_metal.data"
LOGDIR="${EX}/logs"
WORKDIR="${EX}/work_nvt_${SLURM_JOB_ID:-manual}"
mkdir -p "${LOGDIR}" "${WORKDIR}"

export NGPUS="${NGPUS:-1}"
export UMA_DEVICES="${UMA_DEVICES:-${NGPUS}}"
export RECOMPILE="${RECOMPILE:-0}"
export UMA_FORBID_RAY_GP=1
export UMA_LIBTORCH_MP_WORKER="${UMA_LIBTORCH_MP_WORKER:-${ENG}/build-cpp-mp/uma_libtorch_mp_worker}"
unset UMA_PYTHON_GP_WORKER || true
unset UMA_ALLOW_RAY_GP || true

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID:-none} host=$(hostname) NGPUS=${NGPUS} UMA_DEVICES=${UMA_DEVICES}"

if [[ "${RECOMPILE}" == "1" ]]; then
  bash "${ROOT}/scripts/build_lammps_uma.sh"
  if [[ "${UMA_DEVICES}" -gt 1 ]]; then
    pushd "${ENG}/build-cpp-mp" >/dev/null
    cmake .. -DUMA_ENGINE_USE_CUDA=ON
    cmake --build . -j 8 --target uma_libtorch_mp_worker uma_engine
    popd >/dev/null
  fi
fi

LMP="${LMP_UMA:-${ROOT}/build-uma/lmp}"
test -x "${LMP}"
test -f "${ART}/model_traced.pt"
test -f "${DATA}"

# --- untimed prep: stage input + data (no extxyz conversion) ---
cp "${EX}/in.h2o888_nvt" "${WORKDIR}/in.h2o888_nvt"
cp "${DATA}" "${WORKDIR}/water_nvt_300K_atomic_metal.data"
NATOMS=$(awk '/^[[:space:]]*[0-9]+[[:space:]]+atoms[[:space:]]*$/{print $1; exit}' "${DATA}")

VESIN="${ENG}/third_party/vesin/lib"
TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${VESIN}:${TORCH_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
if [[ "${UMA_DEVICES}" -gt 1 ]]; then
  test -x "${UMA_LIBTORCH_MP_WORKER}"
  export UMA_MP_NATOMS="${NATOMS}"
  export UMA_STRUCTURE_NATOMS="${NATOMS}"
fi

cd "${WORKDIR}"
# W8nk product: no Kokkos unless UMA_USE_KOKKOS=1
export UMA_USE_KOKKOS="${UMA_USE_KOKKOS:-0}"
LMP_ARGS=()
if [[ "${UMA_USE_KOKKOS}" == "1" ]]; then
  LMP_ARGS=(-k on g "${NGPUS}" -sf kk)
fi
echo "=== TIMED: lmp ${LMP_ARGS[*]:-} -in in.h2o888_nvt (UMA_USE_KOKKOS=${UMA_USE_KOKKOS}) ==="
t0=$(date +%s.%N)
set +e
"${LMP}" "${LMP_ARGS[@]}" \
  -var UMA_DEVICES "${UMA_DEVICES}" \
  -in in.h2o888_nvt -log log.nvt
rc=$?
set -e
t1=$(date +%s.%N)
wall=$(awk -v a="${t0}" -v b="${t1}" 'BEGIN { printf "%.9f", b - a }')
echo "run_lmp rc=${rc} wall_s=${wall} NGPUS=${NGPUS} UMA_DEVICES=${UMA_DEVICES} nsteps=10 UMA_USE_KOKKOS=${UMA_USE_KOKKOS}"
echo "timing_source=slurm_launch_only"
echo "structure=water_nvt_300K_atomic_metal.data natoms=${NATOMS} flexible=1 shake=0"

python - <<PY
import json, os
from pathlib import Path
wall = float("${wall}")
natoms = int("${NATOMS}")
out = {
    "jobid": os.environ.get("SLURM_JOB_ID"),
    "path": "uma_h2o_nvt300_w8nk" if os.environ.get("UMA_USE_KOKKOS", "0") != "1" else "uma_kk_h2o_nvt300",
    "structure": "water_nvt_300K_atomic_metal.data",
    "natoms": natoms,
    "ensemble": "nvt",
    "temperature_K": 300.0,
    "nsteps": 10,
    "timestep_ps": 0.001,
    "flexible_water": True,
    "fix_shake": False,
    "velocities_from_data": True,
    "UMA_USE_KOKKOS": int(os.environ.get("UMA_USE_KOKKOS", "0") == "1"),
    "ngpus": int("${NGPUS}"),
    "uma_devices": int("${UMA_DEVICES}"),
    "slurm_wall_s": wall,
    "timing_source": "slurm_lmp_launch",
    "ms_total": wall * 1e3,
    "ms_per_step": (wall / 10.0) * 1e3,
    "workdir": str(Path(".").resolve()),
    "rc": int("${rc}"),
}
Path("timing_slurm.json").write_text(json.dumps(out, indent=2) + "\n")
print("wrote timing_slurm.json", out)
PY

tail -30 log.nvt 2>/dev/null || true
exit ${rc}
