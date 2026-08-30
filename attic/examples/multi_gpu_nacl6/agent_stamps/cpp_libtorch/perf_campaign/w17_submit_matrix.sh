#!/bin/bash
# Submit remaining W17 cells after NaCl@2 graph PASS. Usage: bash w17_submit_matrix.sh <afterok_jobid>
set -euo pipefail
DEP="${1:?afterok jobid}"
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ART=${ROOT}/src/ML-UMA/uma-engine/artifacts/uma-s-1p2-omat-f64-fast-cgraph
CAMP=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign
EXP_COMMON=ALL,RECOMPILE=0,UMA_USE_KOKKOS=0,UMA_ARTIFACT_DIR=${ART},UMA_CUDA_GRAPH=1,UMA_EDGE_PAD=1
N4=$(sbatch --parsable --dependency=afterok:${DEP} \
  --chdir=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6 --gpus-per-node=4 --job-name=w17-nacl-n4 \
  --export=${EXP_COMMON},NGPUS=4,UMA_DEVICES=4,FAIRCHEM_WORKERS=4,NSTEPS=10,UMA_MP_LOG_DIR=${CAMP}/matrix/nacl6_uma_ufast_w17_ngpu4/mp_logs \
  ${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/run_path_uma.slurm)
H2=$(sbatch --parsable --dependency=afterok:${DEP} \
  --chdir=${ROOT}/src/ML-UMA/examples/water888 --gpus-per-node=2 --job-name=w17-h2o-n2 \
  --export=${EXP_COMMON},NGPUS=2,UMA_DEVICES=2,FAIRCHEM_WORKERS=2,NSTEPS=100,UMA_MP_LOG_DIR=${CAMP}/matrix/water888_uma_ufast_w17_ngpu2/mp_logs \
  ${ROOT}/src/ML-UMA/examples/water888/run_path_uma.slurm)
H4=$(sbatch --parsable --dependency=afterok:${DEP} \
  --chdir=${ROOT}/src/ML-UMA/examples/water888 --gpus-per-node=4 --job-name=w17-h2o-n4 \
  --export=${EXP_COMMON},NGPUS=4,UMA_DEVICES=4,FAIRCHEM_WORKERS=4,NSTEPS=100,UMA_MP_LOG_DIR=${CAMP}/matrix/water888_uma_ufast_w17_ngpu4/mp_logs \
  ${ROOT}/src/ML-UMA/examples/water888/run_path_uma.slurm)
mkdir -p ${CAMP}/matrix/nacl6_uma_ufast_w17_ngpu4/mp_logs \
  ${CAMP}/matrix/water888_uma_ufast_w17_ngpu2/mp_logs \
  ${CAMP}/matrix/water888_uma_ufast_w17_ngpu4/mp_logs
echo "nacl4=$N4 h2o2=$H2 h2o4=$H4"
