#!/bin/bash
#SBATCH --job-name=uma-cpp-mp-smoke
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/smoke_mp-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/smoke_mp-%j.out

# Rebuild C++ MP runtime; gate devices=2 E+F vs devices=1 on STRUCT.
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART=${ENG}/artifacts/uma-s-1p2-omat-f64
STAMP=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch
STRUCT="${STRUCT:-${ENG}/artifacts/nacl64.txt}"
FORCE_TOL="${FORCE_TOL:-1e-6}"
ENERGY_TOL="${ENERGY_TOL:-1e-9}"

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=$SLURM_JOB_ID host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
echo "STRUCT=$STRUCT FORCE_TOL=$FORCE_TOL ENERGY_TOL=$ENERGY_TOL"
ls -lh "${ART}"/model_mp_w2_r*.pt

export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export RECOMPILE="${RECOMPILE:-1}"
# Force-green defaults (sweep 20925383): arBwd + sumF + escale=1/world (in worker).
unset UMA_SKIP_FORCE_GP_REDUCE || true
unset UMA_GRAD_ENERGY_SCALE || true
export UMA_ALLREDUCE_WITH_GRAD_BWD="${UMA_ALLREDUCE_WITH_GRAD_BWD:-1}"
export UMA_MP_LOG_DIR="${STAMP}/worker_logs_${SLURM_JOB_ID:-manual}"
mkdir -p "${UMA_MP_LOG_DIR}"
echo "UMA_MP_LOG_DIR=${UMA_MP_LOG_DIR}"
echo "UMA_ALLREDUCE_WITH_GRAD_BWD=${UMA_ALLREDUCE_WITH_GRAD_BWD}"

echo "=== rebuild uma-engine ==="
mkdir -p "${ENG}/build-cpp-mp"
cd "${ENG}/build-cpp-mp"
cmake .. -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" -DUMA_ENGINE_USE_CUDA=ON
cmake --build . -j "${SLURM_CPUS_PER_TASK:-8}"

export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${ENG}/third_party/vesin/lib:${LD_LIBRARY_PATH:-}"
export UMA_LIBTORCH_MP_WORKER="${ENG}/build-cpp-mp/uma_libtorch_mp_worker"
export PATH="${ENG}/build-cpp-mp:${PATH}"

echo "=== kokkos_peer_device_smoke ==="
"${ENG}/build-cpp-mp/kokkos_peer_device_smoke"

WORKDIR="${STAMP}/force_gate_${SLURM_JOB_ID:-manual}"
mkdir -p "${WORKDIR}"
F1="${WORKDIR}/f_d1.bin"
F2="${WORKDIR}/f_d2.bin"

echo "=== uma_parity_cli devices=1 ==="
"${ENG}/build-cpp-mp/uma_parity_cli" "${ART}" "$STRUCT" --devices 1 --write-forces "$F1" | tee "${WORKDIR}/d1.txt"

echo "=== uma_parity_cli devices=2 ==="
set +e
"${ENG}/build-cpp-mp/uma_parity_cli" "${ART}" "$STRUCT" --devices 2 --write-forces "$F2" | tee "${WORKDIR}/d2.txt"
RC=$?
set -e
echo "=== worker logs (${UMA_MP_LOG_DIR}) ==="
ls -la "${UMA_MP_LOG_DIR}" || true
for f in "${UMA_MP_LOG_DIR}"/worker_r*.log; do
  [[ -f "$f" ]] || continue
  echo "----- $f -----"
  tail -40 "$f" || true
done
if [[ $RC -ne 0 ]]; then
  echo "parity_cli devices=2 failed rc=$RC"
  exit "$RC"
fi

echo "=== gate E+F vs devices=1 ==="
PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" python - <<PY
import re, struct, sys
from pathlib import Path
wd = Path("${WORKDIR}")
d1 = (wd / "d1.txt").read_text()
d2 = (wd / "d2.txt").read_text()
e1 = float(re.search(r"energy=([-+0-9.eE]+)", d1).group(1))
e2 = float(re.search(r"energy=([-+0-9.eE]+)", d2).group(1))
n1 = int(re.search(r"n=(\d+)", d1).group(1))
n2 = int(re.search(r"n=(\d+)", d2).group(1))
assert n1 == n2, (n1, n2)
f1 = struct.unpack(f"{3*n1}d", (wd / "f_d1.bin").read_bytes())
f2 = struct.unpack(f"{3*n2}d", (wd / "f_d2.bin").read_bytes())
de = abs(e2 - e1)
max_df = max(abs(a - b) for a, b in zip(f1, f2))
# per-atom L2 force delta
import math
max_df_atom = 0.0
for i in range(n1):
    dx = f1[3*i]-f2[3*i]; dy = f1[3*i+1]-f2[3*i+1]; dz = f1[3*i+2]-f2[3*i+2]
    max_df_atom = max(max_df_atom, math.sqrt(dx*dx+dy*dy+dz*dz))
etol = float("${ENERGY_TOL}")
ftol = float("${FORCE_TOL}")
print(f"GATE n={n1} E_d1={e1:.12f} E_d2={e2:.12f} dE={de:.3e}")
print(f"GATE max|dF_comp|={max_df:.6e} max|dF_atom|={max_df_atom:.6e} ftol={ftol:g} etol={etol:g}")
ok_e = de <= etol
ok_f = max_df <= ftol
print("E_GREEN" if ok_e else "E_NOT_GREEN")
print("F_GREEN" if ok_f else "F_NOT_GREEN")
import json
(wd / "gate.json").write_text(json.dumps({
    "n": n1, "E_d1": e1, "E_d2": e2, "dE": de,
    "max_dF": max_df, "max_dF_atom": max_df_atom,
    "E_green": ok_e, "F_green": ok_f,
}, indent=2) + "\n")
sys.exit(0 if (ok_e and ok_f) else 2)
PY

echo "SMOKE_OK job=$SLURM_JOB_ID E+F green vs devices=1"
