# Shared functions for the lammps-uma-nitro PBS jobs.
# Needs: HERE, REL, and (after env.inc) LMP, UMA_CHECKPOINT.
# shellcheck shell=bash

nitro_die() {
  echo "NITRO_FAIL: $*" >&2
  exit 1
}

# Abort unless HEAD is exactly the tagged commit and tracked files are clean.
# Untracked files are ignored on purpose (outputs and builds are git-excluded).
nitro_check_tree() {
  [[ -n "${NITRO_TAG:-}" ]] || nitro_die "NITRO_TAG not set (qsub -v NITRO_TAG=<tag>)"
  local tagsha head
  tagsha=$(git -C "${REL}" rev-parse -q --verify "refs/tags/${NITRO_TAG}^{commit}") \
    || nitro_die "tag ${NITRO_TAG} does not exist"
  head=$(git -C "${REL}" rev-parse HEAD)
  [[ "${head}" == "${tagsha}" ]] \
    || nitro_die "HEAD ${head} is not tag ${NITRO_TAG} (${tagsha})"
  git -C "${REL}" diff --quiet && git -C "${REL}" diff --cached --quiet \
    || nitro_die "tracked files are dirty"
  echo "TREE_OK tag=${NITRO_TAG} sha=${head}"
}

# Runtime environment shared by every lmp run (FP64, AC off, 12 tiles).
nitro_runtime_env() {
  export UMA_AC=off UMA_MN_CKPT=0 UMA_MP_PERF=1 UMA_MEM_PEAK=1
  export OMP_NUM_THREADS=1
  export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi CCL_ZE_IPC_EXCHANGE=pidfd
  export CCL_WORKER_COUNT=1 CCL_KVS_MODE=mpi
  export FI_PROVIDER="${CCL_FI_PROVIDER:-cxi}"
  echo "aurora_ccl: FI_PROVIDER=${FI_PROVIDER}"
}

# nitro_write_in ENS EXTRA NSTEPS DUMPEVERY
#   ENS        name; writes ${RUNDIR}/in.ENS
#   EXTRA      one LAMMPS line (the integrator fix)
#   NSTEPS     run length
#   DUMPEVERY  0 = no force dump; N>0 = dump forces at steps 0, N, 2N, ...
nitro_write_in() {
  local ens="$1" extra="$2" nsteps="$3" dumpevery="$4"
  local dumpblock=""
  if [[ "${dumpevery}" -gt 0 ]]; then
    dumpblock="dump            fdump all custom ${dumpevery} forces.${ens}.dump id type fx fy fz
dump_modify     fdump sort id format float %.17g"
  fi
  cat > "${RUNDIR}/in.${ens}" <<EOF
units           metal
atom_style      atomic
boundary        p p p
newton          off
read_data       data.nacl
mass            1 22.989769
mass            2 35.453
pair_style      uma precision double devices 1
pair_coeff      * * \${uma_art} Na Cl
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
thermo          1
thermo_style    custom step temp pe etotal press vol
thermo_modify   norm no format float %.16e
velocity        all create 300.0 4928459 mom yes rot yes dist gaussian
${dumpblock}
${extra}
timestep        0.001
run             ${nsteps}
EOF
}

# nitro_run_ens ENS : run in.ENS under mpiexec on one node. Returns lmp exit code.
nitro_run_ens() {
  local ens="$1"
  echo "########## ${ens} $(date -Is) ##########"
  local rc=0
  (
    cd "${RUNDIR}" || exit 99
    mpiexec -n "${W}" --ppn "${PPN}" gpu_tile_compact.sh \
      env OMP_NUM_THREADS=1 \
          UMA_AC=off UMA_MN_CKPT=0 UMA_MP_PERF=1 UMA_MEM_PEAK=1 \
          UMA_MP_NATOMS="${NAT}" UMA_CHECKPOINT="${UMA_CHECKPOINT}" \
          UMA_ALLOW_LEGACY_METADATA=0 UMA_ALLOW_FAIRCHEM_MISMATCH=1 \
          LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
      "${LMP}" -in "in.${ens}" -log "log.${ens}" -var uma_art "${ART}" \
      > "lmp.${ens}.log" 2>&1
  ) && rc=0 || rc=$?
  echo "  ${ens} lmp exit=${rc}"
  grep -E 'Loop time of|Performance:|virial refused|UMA_HEN_ROOT' \
    "${RUNDIR}/log.${ens}" "${RUNDIR}/lmp.${ens}.log" 2>/dev/null | head -20 || true
  if [[ "${rc}" -ne 0 ]]; then
    tail -40 "${RUNDIR}/lmp.${ens}.log" || true
  fi
  return "${rc}"
}
