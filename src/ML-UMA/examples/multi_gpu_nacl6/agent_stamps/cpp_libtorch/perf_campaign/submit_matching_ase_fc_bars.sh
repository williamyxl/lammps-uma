#!/bin/bash
# Submit ASE + FC matching-settings speed bars (measure once, then lock).
# Settings: general+merge_mole and umas_fast_pytorch+merge_mole @2/@4.
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
NACL=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6
WATER=${ROOT}/src/ML-UMA/examples/water888
CAMP=${NACL}/agent_stamps/cpp_libtorch/perf_campaign
REF=${CAMP}/ref_ase_fc
JOBS=${CAMP}/matching_ase_fc_jobs.txt
mkdir -p "${REF}" "${WATER}/logs"
: > "${JOBS}"

submit_nacl() {
  # Path-isolated SP + NVT@300K (NSTEPS=10) — same recipe as future campaign gates.
  local path=$1 ngpu=$2 mode=$3 merge=$4 tag=$5
  local path_key=$path
  if [[ "${path}" == "uma_double" || "${path}" == "uma" ]]; then
    path_key=uma
  fi
  local script=${NACL}/run_path_${path_key}.slurm
  local jid
  jid=$(sbatch --parsable \
    --chdir="${NACL}" \
    --gpus-per-node="${ngpu}" \
    --job-name="ref-nacl-${path_key}-${tag}-n${ngpu}" \
    --export=ALL,NGPUS=${ngpu},UMA_DEVICES=${ngpu},FAIRCHEM_WORKERS=${ngpu},NSTEPS=10,RECOMPILE=0,FAIRCHEM_EXECUTION_MODE=${mode},FAIRCHEM_MERGE_MOLE=${merge} \
    "${script}")
  echo "nacl6 path=${path_key} tag=${tag} @${ngpu} job=${jid} NSTEPS=10 NVT@300K" | tee -a "${JOBS}"
}

submit_water() {
  local path=$1 ngpu=$2 mode=$3 merge=$4 tag=$5
  local script=${WATER}/run_path_${path}.slurm
  local jid
  jid=$(sbatch --parsable \
    --chdir="${WATER}" \
    --gpus-per-node="${ngpu}" \
    --job-name="h2o-ref-${path}-${tag}-n${ngpu}" \
    --export=ALL,NGPUS=${ngpu},FAIRCHEM_WORKERS=${ngpu},NSTEPS=100,FAIRCHEM_EXECUTION_MODE=${mode},FAIRCHEM_MERGE_MOLE=${merge} \
    "${script}")
  echo "water888 path=${path} tag=${tag} @${ngpu} job=${jid}" | tee -a "${JOBS}"
}

# tag=gmerge → general + merge_mole
# tag=ufast  → umas_fast_pytorch + merge_mole
for ngpu in 2 4; do
  for path in ase fc; do
    submit_nacl "${path}" "${ngpu}" general 1 gmerge
    submit_nacl "${path}" "${ngpu}" umas_fast_pytorch 1 ufast
    submit_water "${path}" "${ngpu}" general 1 gmerge
    submit_water "${path}" "${ngpu}" umas_fast_pytorch 1 ufast
  done
done

echo "WROTE ${JOBS}"
cat "${JOBS}"
