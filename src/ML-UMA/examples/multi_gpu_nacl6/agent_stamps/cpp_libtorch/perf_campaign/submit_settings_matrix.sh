#!/bin/bash
# Submit missing settings-matrix cells (ASE legal + FC gen + uma legal).
# Skips illegal ufast_nomole and FC gmerge/ufast (known crash).
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
NACL=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6
WATER=${ROOT}/src/ML-UMA/examples/water888
ENG=${ROOT}/src/ML-UMA/uma-engine
CAMP=${NACL}/agent_stamps/cpp_libtorch/perf_campaign
OUT=${CAMP}/matrix
JOBS=${CAMP}/matrix_jobs.txt
mkdir -p "${OUT}"
: > "${JOBS}"

art_for() {
  case "$1" in
    gen) echo "${ENG}/artifacts/uma-s-1p2-omat-f64" ;;
    gmerge) echo "${ENG}/artifacts/uma-s-1p2-omat-f64-merge" ;;
    ufast) echo "${ENG}/artifacts/uma-s-1p2-omat-f64-fast" ;;
    *) echo "bad tag $1" >&2; exit 2 ;;
  esac
}

mode_merge() {
  case "$1" in
    gen) echo "general 0" ;;
    gmerge) echo "general 1" ;;
    ufast) echo "umas_fast_pytorch 1" ;;
  esac
}

submit_nacl() {
  local path=$1 tag=$2 ngpu=$3
  read -r mode merge <<< "$(mode_merge "${tag}")"
  local art; art=$(art_for "${tag}")
  local rdir=${OUT}/nacl6_${path}_${tag}_ngpu${ngpu}
  mkdir -p "${rdir}"
  local script=${NACL}/run_ngpu${ngpu}_${path}.slurm
  if [[ "${path}" == "uma_double" || "${path}" == "uma" ]]; then
    script=${NACL}/run_ngpu${ngpu}_uma_double.slurm
    path=uma_double
  fi
  local export_vars="ALL,NGPUS=${ngpu},UMA_DEVICES=${ngpu},ONLY_PATHS=${path},FAIRCHEM_WORKERS=${ngpu},RECOMPILE=0,MERGE_RESULTS=0,SKIP_DIST_DESTROY=1,FAIRCHEM_EXECUTION_MODE=${mode},FAIRCHEM_MERGE_MOLE=${merge},RESULTS_DIR=${rdir},UMA_ARTIFACT_DIR=${art}"
  local jid
  jid=$(sbatch --parsable --job-name="mx-${path}-${tag}-n${ngpu}" --export="${export_vars}" "${script}")
  echo "nacl6 path=${path} tag=${tag} @${ngpu} job=${jid}" | tee -a "${JOBS}"
}

submit_water() {
  local path=$1 tag=$2 ngpu=$3
  read -r mode merge <<< "$(mode_merge "${tag}")"
  local art; art=$(art_for "${tag}")
  local script=${WATER}/run_path_${path}.slurm
  local jid
  jid=$(sbatch --parsable --gpus-per-node="${ngpu}" \
    --job-name="mx-h2o-${path}-${tag}-n${ngpu}" \
    --export=ALL,NGPUS=${ngpu},FAIRCHEM_WORKERS=${ngpu},UMA_DEVICES=${ngpu},NSTEPS=100,FAIRCHEM_EXECUTION_MODE=${mode},FAIRCHEM_MERGE_MOLE=${merge},UMA_ARTIFACT_DIR=${art},RECOMPILE=0 \
    "${script}")
  echo "water888 path=${path} tag=${tag} @${ngpu} job=${jid}" | tee -a "${JOBS}"
}

# --- ASE missing: NaCl @1 gmerge/ufast; water gmerge/ufast @1/2/4 ---
for tag in gmerge ufast; do
  submit_nacl ase "${tag}" 1
done
for ngpu in 1 2 4; do
  for tag in gmerge ufast; do
    submit_water ase "${tag}" "${ngpu}"
  done
done

# --- uma missing high priority: @1 ufast both; gmerge @4 NaCl + water @2/@4; water already has ufast @2/@4 ---
for sys_submit in nacl water; do
  :
done
submit_nacl uma_double ufast 1
submit_water uma ufast 1
submit_nacl uma_double gmerge 1
submit_nacl uma_double gmerge 4
submit_water uma gmerge 2
submit_water uma gmerge 4
submit_water uma gmerge 1
submit_nacl uma_double gen 1
submit_water uma gen 1

echo "WROTE ${JOBS}"
cat "${JOBS}"
