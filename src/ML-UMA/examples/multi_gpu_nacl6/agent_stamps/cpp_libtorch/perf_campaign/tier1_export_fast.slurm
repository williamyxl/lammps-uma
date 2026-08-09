#!/bin/bash
#SBATCH --job-name=uma-t1-exp
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/tier1_export-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/tier1_export-%j.out

# Tier1: export MP TorchScript with umas_fast_pytorch + merge_mole into a
# separate artifact dir (do not overwrite general-mode product artifacts).
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART_SRC=${ENG}/artifacts/uma-s-1p2-omat-f64
ART=${ENG}/artifacts/uma-s-1p2-omat-f64-fast
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
WATER_EX=${ROOT}/src/ML-UMA/examples/water888
NACL_XYZ=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz
WATER_DATA=${WATER_EX}/water_nvt_300K_atomic_metal.data
WATER_XYZ=${WATER_EX}/water_nvt_300K.extxyz

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID:-none} host=$(hostname) Tier1 fast export"
mkdir -p "${ART}"
# Reuse metadata + state from general f64 artifact.
cp -n "${ART_SRC}/metadata.json" "${ART}/" || cp "${ART_SRC}/metadata.json" "${ART}/"
cp -n "${ART_SRC}/model_state.pt" "${ART}/" || cp "${ART_SRC}/model_state.pt" "${ART}/"

python - <<PY
from ase.io import read, write
a = read("${WATER_DATA}", format="lammps-data", atom_style="atomic", units="metal")
a.pbc = True
write("${WATER_XYZ}", a)
print("wrote ${WATER_XYZ} natoms=", len(a))
PY

export_world() {
  local W="$1"
  local ATOMS="$2"
  local TAG="$3"
  echo "=== Tier1 export_mp world=${W} atoms=${ATOMS} (${TAG}) ==="
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
        --state-dict "${ART}/model_state.pt" \
        --atoms "${ATOMS}" \
        --execution-mode umas_fast_pytorch \
        --merge-mole
  done
  ls -lh "${ART}"/model_mp_w${W}_n*_r*.pt | tail -n 20
}

# Water888 MP @2/@4
export_world 2 "${WATER_XYZ}" water648
export_world 4 "${WATER_XYZ}" water648
# NaCl6 MP @2/@4
export_world 2 "${NACL_XYZ}" nacl1728
export_world 4 "${NACL_XYZ}" nacl1728

echo "TIER1_EXPORT_OK job=${SLURM_JOB_ID} art=${ART}"
