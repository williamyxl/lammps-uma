#!/bin/bash
#SBATCH --job-name=mn-w8-export
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=00:50:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/uma-engine/tests/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/uma-engine/tests/%x-%j.out

# Export the 8 MP shards (model_mp_w8_n4096_r0..7) for NaCl 8x8x8, so a 2-node
# / 8-GPU run can use the SAME edge-parallel MP path as the 4-GPU ground truth
# (job 21026029, w4 shards). Each shard is a single-GPU export; 8 of them fit
# on one 4-GPU node run two waves of 4.
set -uo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
CKPT=/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt
XYZ=${ROOT}/src/ML-UMA/examples/nacl_nsweep/structures/nacl8_rattle.extxyz
ART=${ROOT}/src/ML-UMA/examples/nacl_nsweep/artifacts/nacl8
WORLD=8

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID} export w${WORLD} shards for nacl8 (4096 atoms)"
test -f "${XYZ}" || { echo "MN_W8_FAIL missing ${XYZ}"; exit 1; }
cp -n "${ENG}/artifacts/uma-s-1p2-omat-f64-fast/metadata.json" "${ART}/" 2>/dev/null || true

for R in 0 1 2 3 4 5 6 7; do
  GPU=$((R % 4))
  echo "=== export world=${WORLD} rank=${R} on GPU ${GPU} ==="
  CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" \
    python "${ENG}/python/export_mp_artifact.py" \
      --checkpoint "${CKPT}" --output "${ART}" \
      --world "${WORLD}" --ranks "${R}" \
      --dtype float64 --task omat --device cuda \
      --atoms "${XYZ}" \
      --execution-mode umas_fast_pytorch --merge-mole &
  # 4 GPUs -> launch in waves of 4
  if [[ $((R % 4)) -eq 3 ]]; then wait; fi
done
wait

n=$(ls "${ART}"/model_mp_w8_n4096_r*.pt 2>/dev/null | wc -l)
echo "w8 shards present: ${n}/8"
[[ "${n}" -eq 8 ]] && echo "MN_W8_EXPORT_OK" || { echo "MN_W8_EXPORT_FAIL"; exit 1; }
