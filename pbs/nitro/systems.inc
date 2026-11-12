# Test systems for the lammps-uma-nitro harness. Sourced by lib.inc.
# Select with NITRO_SYSTEM=nacl (default) or NITRO_SYSTEM=zif8.
# Both systems use the same protocol: W=12, FP64, NVT and NPT, T=300 K,
# velocity seed 4928459, timestep 0.001 ps, same fixes, same pass bars.
# Only the structure differs.
# shellcheck shell=bash

# nitro_system_init : set the SYS_* variables for ${NITRO_SYSTEM}.
#   SYS_LABEL    text for logs
#   SYS_DATA     LAMMPS data file name written by the generator
#   SYS_DIR      sub-directory name used by the gate job
#   SYS_NAT      required atom count
#   SYS_ELEMENTS pair_coeff element list, in LAMMPS type order
#   SYS_MASSES   "mass" lines for the input script
#   SYS_REF_*    recorded reference PE (eV); empty = not recorded yet
nitro_system_init() {
  NITRO_SYSTEM="${NITRO_SYSTEM:-nacl}"
  case "${NITRO_SYSTEM}" in
    nacl)
      SYS_LABEL="NaCl rocksalt 14x14x14"
      SYS_DATA=data.nacl
      SYS_DIR=n14
      SYS_NAT=21952
      SYS_ELEMENTS="Na Cl"
      SYS_MASSES="mass            1 22.989769
mass            2 35.453"
      # Recorded Wave 0 / Wave E values for NaCl 14^3 W=12 FP64 (uma-s-1p2)
      SYS_REF_STEP0_PE=-7.4142191907631001e+04
      SYS_REF_NVT_PE10=-7.4095578889731943e+04
      SYS_REF_NPT_PE10=-7.4098199120459016e+04
      ;;
    zif8)
      SYS_LABEL="ZIF-8 3x3x3 (rattled 0.05 A, seed 0)"
      SYS_DATA=data.zif8
      SYS_DIR=z3
      SYS_NAT=7452
      SYS_ELEMENTS="H C N Zn"
      SYS_MASSES="mass            1 1.00784
mass            2 12.011
mass            3 14.0067
mass            4 65.38"
      # Not recorded yet. Stage H7 records them from the first BASE gate job.
      # Until then run_gate.pbs runs in record-only mode (no pass bar).
      SYS_REF_STEP0_PE=""
      SYS_REF_NVT_PE10=""
      SYS_REF_NPT_PE10=""
      ;;
    *)
      nitro_die "unknown NITRO_SYSTEM '${NITRO_SYSTEM}' (want nacl or zif8)"
      ;;
  esac
}

# nitro_make_cell OUTDIR : generate ${SYS_DATA} and structure.xyz, check atom count.
nitro_make_cell() {
  local out="$1"
  case "${NITRO_SYSTEM}" in
    nacl) python "${HERE}/make_nacl.py" 14 "${out}" || return 1 ;;
    zif8) python "${HERE}/make_zif8.py" 3 "${out}" || return 1 ;;
  esac
  local nat
  nat="$(awk '/atoms/{print $1; exit}' "${out}/${SYS_DATA}")"
  [[ "${nat}" -eq "${SYS_NAT}" ]] || {
    echo "NITRO_FAIL: ${SYS_LABEL} has ${nat} atoms; require ${SYS_NAT}" >&2
    return 1
  }
}
