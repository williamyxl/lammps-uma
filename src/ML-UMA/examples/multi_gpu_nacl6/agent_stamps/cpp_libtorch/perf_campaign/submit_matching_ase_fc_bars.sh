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
  local path=$1 ngpu=$2 mode=$3 merge=$4 tag=$5
  local script=${NACL}/run_ngpu${ngpu}_${path}.slurm
  local rdir=${REF}/nacl6_${path}_${tag}_ngpu${ngpu}
  mkdir -p "${rdir}"
  local jid
  jid=$(sbatch --parsable \
    --job-name="ref-${path}-${tag}-n${ngpu}" \
    --export=ALL,NGPUS=${ngpu},UMA_DEVICES=${ngpu},ONLY_PATHS=${path},FAIRCHEM_WORKERS=${ngpu},RECOMPILE=0,MERGE_RESULTS=0,SKIP_DIST_DESTROY=1,FAIRCHEM_EXECUTION_MODE=${mode},FAIRCHEM_MERGE_MOLE=${merge},RESULTS_DIR=${rdir} \
    "${script}")
  echo "nacl6 path=${path} tag=${tag} @${ngpu} job=${jid} results=${rdir}" | tee -a "${JOBS}"
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
