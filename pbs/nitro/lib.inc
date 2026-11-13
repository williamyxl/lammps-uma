# Shared functions for the lammps-uma-nitro PBS jobs.
# Needs: HERE, REL, and (after env.inc) LMP, UMA_CHECKPOINT.
# shellcheck shell=bash

nitro_die() {
  echo "NITRO_FAIL: $*" >&2
  exit 1
}

# Test systems (NITRO_SYSTEM=nacl|zif8). Defaults to nacl.
# shellcheck disable=SC1091
source "${HERE}/systems.inc"
nitro_system_init

# Abort unless HEAD is exactly the tagged commit, tracked files are clean, and
# src/ and pbs/nitro/ have no untracked non-ignored files.
nitro_check_tree() {
  [[ -n "${NITRO_TAG:-}" ]] || nitro_die "NITRO_TAG not set (qsub -v NITRO_TAG=<tag>)"
  local tagsha head stray
  tagsha=$(git -C "${REL}" rev-parse -q --verify "refs/tags/${NITRO_TAG}^{commit}") \
    || nitro_die "tag ${NITRO_TAG} does not exist"
  head=$(git -C "${REL}" rev-parse HEAD)
  [[ "${head}" == "${tagsha}" ]] \
    || nitro_die "HEAD ${head} is not tag ${NITRO_TAG} (${tagsha})"
  git -C "${REL}" diff --quiet && git -C "${REL}" diff --cached --quiet \
    || nitro_die "tracked files are dirty"
  stray=$(git -C "${REL}" ls-files --others --exclude-standard -- src pbs/nitro)
  if [[ -n "${stray}" ]]; then
    nitro_die "untracked non-ignored files under src/ or pbs/nitro: ${stray}"
  fi
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
read_data       ${SYS_DATA}
${SYS_MASSES}
pair_style      uma precision double devices 1
pair_coeff      * * \${uma_art} ${SYS_ELEMENTS}
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
    # NITRO_SLOT_ENV is a space-separated list of KEY=VAL with no spaces in values.
    # It is applied after the defaults so a slot can override a default.
    # shellcheck disable=SC2086
    mpiexec -n "${W}" --ppn "${PPN}" gpu_tile_compact.sh \
      env OMP_NUM_THREADS=1 \
          UMA_AC=off UMA_MN_CKPT=0 UMA_MP_PERF=1 UMA_MEM_PEAK=1 \
          UMA_MP_NATOMS="${NAT}" UMA_CHECKPOINT="${UMA_CHECKPOINT}" \
          UMA_ALLOW_LEGACY_METADATA=0 UMA_ALLOW_FAIRCHEM_MISMATCH=1 \
          ${NITRO_SLOT_ENV:-} \
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

# Resolve MPICH after env.inc has set MPICH. Fail with the path that was tried.
nitro_require_mpich() {
  [[ -n "${MPICH:-}" && -x "${MPICH}/bin/mpicc" ]] \
    || nitro_die "MPICH not usable at '${MPICH:-unset}' (set MPICH to a prefix with bin/mpicc; fallback is MPICH_FALLBACK in env.inc)"
  echo "MPICH_OK ${MPICH}"
}

# Actual fairchem distribution vs the version this plan expects.
nitro_fairchem_note() {
  python3 - <<'PY'
import importlib.metadata as md
expected = "2.21.0"
found = None
for dist in ("fairchem-core", "fairchem"):
    try:
        found = md.version(dist)
        break
    except md.PackageNotFoundError:
        continue
if found is None:
    print("fairchem=unavailable expected=2.21.0")
    print("WARN: fairchem version unavailable; UMA_ALLOW_FAIRCHEM_MISMATCH=1 is set")
elif found != expected:
    print(f"fairchem={found} expected={expected}")
    print(f"WARN: fairchem {found} != {expected}; UMA_ALLOW_FAIRCHEM_MISMATCH=1 is set")
else:
    print(f"fairchem={found} expected={expected}")
PY
}

# Build lmp into NITRO_BUILD from NITRO_SRC (default: this clone).
# Requires env.inc (MPICH, CONDA_PREFIX, TORCH_LIB, python).
nitro_cmake_build() {
  nitro_require_mpich
  local src="${NITRO_SRC:-${REL}}"
  local bld="${NITRO_BUILD:?NITRO_BUILD not set}"
  local eng="${src}/src/ML-UMA/uma-engine"
  local gcc="${GCC_PREFIX:-/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/gcc-13.4.0-22p27zg}"
  local torch_cmake sycl_lib sycl_inc
  torch_cmake=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share','cmake'))")
  sycl_lib="${CONDA_PREFIX}/lib"
  sycl_inc="${CONDA_PREFIX}/include"
  local clog="${OUTROOT:-/tmp}/cmake.log"
  local blog="${OUTROOT:-/tmp}/build.log"
  echo "########## cmake src=${src} bld=${bld} $(date -Is) ##########"
  cmake -S "${src}/cmake" -B "${bld}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D CMAKE_C_COMPILER="${gcc}/bin/gcc" \
    -D CMAKE_CXX_COMPILER="${gcc}/bin/g++" \
    -D CMAKE_CXX_STANDARD=17 \
    -D MPI_CXX_COMPILER="${MPICH}/bin/mpicxx" \
    -D MPI_C_COMPILER="${MPICH}/bin/mpicc" \
    -D CMAKE_PREFIX_PATH="${torch_cmake}" \
    -D PKG_KOKKOS=OFF \
    -D PKG_ML-UMA=ON \
    -D UMA_ENGINE_USE_XPU=ON \
    -D UMA_ENGINE_USE_CUDA=OFF \
    -D UMA_ENGINE_USE_XCCL=ON \
    -D UMA_ICPX="$(which icpx)" \
    -D ONECCL_ROOT="${CONDA_PREFIX}" \
    -D UMA_ENGINE_ROOT="${eng}" \
    -D SYCL_LIBRARY="${sycl_lib}/libsycl.so" \
    -D SYCL_LIBRARY_DIR="${sycl_lib}" \
    -D SYCL_INCLUDE_DIR="${sycl_inc}" \
    -D SYCL_INCLUDE_SYCL_DIR="${sycl_inc}/sycl" \
    -D CMAKE_CXX_FLAGS="-O2 -Wno-tautological-compare" \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,${sycl_lib} -Wl,-rpath,${TORCH_LIB} -Wl,-rpath,${MPICH}/lib" \
    > "${clog}" 2>&1 || nitro_die "cmake configure failed (see ${clog})"
  echo "########## cmake --build $(date -Is) ##########"
  cmake --build "${bld}" -j 32 > "${blog}" 2>&1 || nitro_die "cmake build failed (see ${blog})"
  [[ -x "${bld}/lmp" ]] || nitro_die "missing lmp ${bld}/lmp"
  LMP="${bld}/lmp"
  echo "BUILD_OK ${LMP}"
}

# Artifact lifetime. A job deletes only directories it marked at export start.
# NITRO_REF / reused artifact paths are never marked, so they are never removed.
nitro_install_cleanup_trap() {
  NITRO_OWNED_LIST="${OUTROOT}/.owned_artifacts"
  : > "${NITRO_OWNED_LIST}"
  trap 'nitro_cleanup_owned' EXIT
}

nitro_own_artifact() {
  local art="$1"
  mkdir -p "${art}"
  echo "job=${PBS_JOBID:-manual}" > "${art}/.nitro_owned"
  echo "${art}" >> "${NITRO_OWNED_LIST}"
}

nitro_cleanup_owned() {
  if [[ "${KEEP_ARTIFACT:-0}" == "1" ]]; then
    echo "ARTIFACT_KEPT (KEEP_ARTIFACT=1)"
    return 0
  fi
  [[ -f "${NITRO_OWNED_LIST:-}" ]] || return 0
  local art
  while IFS= read -r art; do
    [[ -n "${art}" && -f "${art}/.nitro_owned" ]] || continue
    if grep -q "job=${PBS_JOBID:-manual}" "${art}/.nitro_owned"; then
      rm -rf "${art}"
      echo "ARTIFACT_DELETED ${art}"
    fi
  done < "${NITRO_OWNED_LIST}"
  return 0
}

# Export ranks. The W=12 gate leaves NITRO_EXPORT_RANKS unset.
# Single-rank and microbenchmark jobs set NITRO_EXPORT_RANKS=1.
nitro_export_world() {
  if [[ "${NITRO_EXPORT_RANKS:-0}" == "1" ]]; then
    echo 1
  else
    echo "${W}"
  fi
}

# Export into ${ART} from ${RUNDIR}/structure.xyz. Rank count is $1, or
# nitro_export_world when $1 is omitted. Marks nothing; the caller owns ${ART}.
nitro_export_artifact() {
  local world="${1:-$(nitro_export_world)}"
  local eng="${NITRO_EXPORT_ENG:-${ENG}}"
  export OUT="${ART}" EXPORT_WORLD="${world}" EXPORT_ATOMS="${RUNDIR}/structure.xyz"
  export EDGE_AC_CHUNK="${EDGE_AC_CHUNK:-65536}" UMA_MP_NATOMS="${NAT}"
  export ENG_EXPORT="${eng}/python/export_blocks_xpu.py"
  echo "########## export ranks=${world} nat=${NAT} $(date -Is) ##########"
  mpiexec -n "${world}" --ppn "${world}" gpu_tile_compact.sh \
    bash "${HERE}/export_one_rank.sh"
  # W==1 writes model_traced.pt into ${ART} itself. W>1 writes ${ART}/w${W}/r${R}/.
  if [[ "${world}" == "1" ]]; then
    [[ -f "${ART}/model_traced.pt" && -f "${ART}/metadata.json" ]] || {
      tail -30 "${ART}/exp_r0.log" 2>/dev/null || true
      nitro_die "export incomplete: missing ${ART}/model_traced.pt"
    }
  else
    local okn
    okn=$(find "${ART}/w${world}" -name model_traced.pt | wc -l)
    [[ "${okn}" -eq "${world}" ]] || {
      tail -30 "${ART}/exp_r0.log" 2>/dev/null || true
      nitro_die "export incomplete got=${okn} want=${world}"
    }
    cp -f "${ART}/w${world}/r0/metadata.json" "${ART}/metadata.json"
  fi
  echo "EXPORT_OK ${ART}"
}
