#!/bin/bash
#SBATCH --job-name=uma-parity
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Delta A100: export TorchScript artifacts (if needed), build uma/kk LAMMPS,
# then run 4-path parity (ASE FP64 / FairChem fix-ext / uma double / uma mixed)
# on rattled NaCl 3x3x3, Al FCC 3x3x3, Si 3x3x3, Si 4x4x4.
#
# Submit from anywhere:
#   sbatch /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/delta_parity/run_parity.slurm
#
# Optional env (pass via sbatch --export=ALL,ONLY_SYSTEM=nacl,...):
#   ONLY_SYSTEM=nacl|al|si|si4|all
#   SKIP_EXPORT=1  SKIP_BUILD=1  SKIP_FREEZE=1

set -euo pipefail

ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
EX=${ROOT}/src/ML-UMA/examples/delta_parity
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
CONDA_SH=/u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
LMP_FC=${LMP_FC:-/u/xyan11/miniforge3-x86_64/envs/uma312/bin/lmp}

# Drop conflicting CUDA modules from the login/user env (cudatoolkit vs cuda/12.8).
module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8

source "${CONDA_SH}"
conda activate uma312

export UMA_CHECKPOINT="${CKPT}"
export LMP_FC
export LMP_UMA="${LMP_UMA:-${ROOT}/build-uma/lmp}"
export PYTHONUNBUFFERED=1
export ONLY_SYSTEM="${ONLY_SYSTEM:-all}"

cd "${ROOT}"
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true
echo "python=$(which python)  torch=$(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "LMP_FC=${LMP_FC}  LMP_UMA=${LMP_UMA}  ONLY_SYSTEM=${ONLY_SYSTEM}"

# --- 1. Vendor vesin for x86_64 ---
bash "${ENG}/scripts/vendor_vesin_for_arch.sh"

# --- 2. Export TorchScript artifacts (skipped if present) ---
if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then
  export_one() {
    local dtype=$1 out=$2
    if [[ -f "${out}/model_traced.pt" ]]; then
      echo "artifact ok: ${out}/model_traced.pt"
      return 0
    fi
    echo ">>> exporting ${out} dtype=${dtype}"
    PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" python "${ENG}/python/export_omat.py" \
      --checkpoint "${CKPT}" \
      --dtype "${dtype}" \
      --device cuda \
      --output "${out}"
  }
  export_one float32 "${ENG}/artifacts/uma-s-1p2-omat"
  export_one float64 "${ENG}/artifacts/uma-s-1p2-omat-f64"
fi

# --- 3. Build local uma/kk LAMMPS ---
if [[ ! -x "${LMP_UMA}" ]]; then
  echo ">>> building LAMMPS (binary missing) -> ${ROOT}/build-uma"
  bash "${ROOT}/scripts/build_lammps_uma.sh"
elif [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo ">>> building LAMMPS -> ${ROOT}/build-uma"
  bash "${ROOT}/scripts/build_lammps_uma.sh"
fi
test -x "${LMP_UMA}" || { echo "ERROR: missing ${LMP_UMA}"; exit 1; }

# --- 4. Freeze rattled structures (CPU) ---
if [[ "${SKIP_FREEZE:-0}" != "1" ]]; then
  python "${EX}/freeze_structures.py"
fi

# --- 5. 4-path parity ---
export SKIP_BUILD=1   # structures already frozen
python "${EX}/run_parity.py"

# --- 6. Reports (also invoked from run_parity.py; re-run is cheap) ---
python "${EX}/write_reports.py"

echo "DONE. Results under ${EX}/results/"
ls -la "${EX}/results/" || true
test -f "${EX}/results/PARITY_REPORT.md" && echo "MD report: ${EX}/results/PARITY_REPORT.md"
test -f "${EX}/results/uma-delta-parity.canvas.tsx" && echo "Canvas: ${EX}/results/uma-delta-parity.canvas.tsx"
