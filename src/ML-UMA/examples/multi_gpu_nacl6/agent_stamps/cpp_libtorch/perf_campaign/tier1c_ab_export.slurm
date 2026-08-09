#!/bin/bash
#SBATCH --job-name=uma-t1c-ab
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/tier1c_ab-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/tier1c_ab-%j.out

# Tier1c A/B: isolate merge_mole vs umas_fast (Tier1b still dE~2e-5 vs general ASE).
#   A: execution_mode=general + merge_mole=True  → art *-f64-merge
#   B: execution_mode=umas_fast_pytorch + merge_mole=False → art *-f64-fastnomole
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART_SRC=${ENG}/artifacts/uma-s-1p2-omat-f64
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
NACL_XYZ=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz
WATER_XYZ=${ROOT}/src/ML-UMA/examples/water888/water_nvt_300K.extxyz

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID:-none} host=$(hostname) Tier1c A/B export"

export_variant() {
  local ART="$1"
  local MODE="$2"
  local MERGE="$3"
  local TAG="$4"
  mkdir -p "${ART}"
  cp -n "${ART_SRC}/metadata.json" "${ART}/" || cp "${ART_SRC}/metadata.json" "${ART}/"
  cp -n "${ART_SRC}/model_state.pt" "${ART}/" || true
  cp -n "${ART_SRC}/model_traced.pt" "${ART}/" || true
  echo "=== VARIANT ${TAG} mode=${MODE} merge_mole=${MERGE} art=${ART} ==="
  export_world() {
    local W="$1" ATOMS="$2" LABEL="$3"
    echo "--- ${TAG} world=${W} ${LABEL} ---"
    local EXTRA=(--execution-mode "${MODE}")
    if [[ "${MERGE}" == "1" ]]; then EXTRA+=(--merge-mole); fi
    for R in $(seq 0 $((W - 1))); do
      CUDA_VISIBLE_DEVICES=$R PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" \
        python "${ENG}/python/export_mp_artifact.py" \
          --checkpoint "${CKPT}" \
          --output "${ART}" \
          --world "${W}" \
          --ranks "${R}" \
          --dtype float64 \
          --task omat \
          --device cuda \
          --atoms "${ATOMS}" \
          "${EXTRA[@]}"
    done
  }
  # NaCl@2 first (fast parity probe); then water@2; then @4 both
  export_world 2 "${NACL_XYZ}" nacl1728
  export_world 2 "${WATER_XYZ}" water648
  export_world 4 "${NACL_XYZ}" nacl1728
  export_world 4 "${WATER_XYZ}" water648
  echo "VARIANT_OK ${TAG}"
}

export_variant "${ENG}/artifacts/uma-s-1p2-omat-f64-merge" general 1 merge
export_variant "${ENG}/artifacts/uma-s-1p2-omat-f64-fastnomole" umas_fast_pytorch 0 fastnomole

echo "TIER1C_AB_EXPORT_OK job=${SLURM_JOB_ID}"
