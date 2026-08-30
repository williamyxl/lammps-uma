#!/usr/bin/env bash
# Rebuild LAMMPS (if WRITE landed C++ changes) then submit gp_round jobs.
# Default: ngpu1 baseline first, then ngpu2 after ngpu1 completes.
#
# Usage:
#   ./gp_round/rebuild_and_submit.sh           # dry-run checklist only
#   ./gp_round/rebuild_and_submit.sh --submit # rebuild + path-isolated gp jobs
#   ./gp_round/rebuild_and_submit.sh --submit --ngpu4  # also 4-GPU path jobs

set -euo pipefail

GP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EX="$(cd "${GP}/.." && pwd)"
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma

SUBMIT=0
SUBMIT_NGPU4=0
for arg in "$@"; do
  case "$arg" in
    --submit) SUBMIT=1 ;;
    --ngpu4) SUBMIT_NGPU4=1 ;;
  esac
done

echo "=== gp_round rebuild_and_submit ==="
# Prefer gp_round stamp (this campaign); fall back to suite-root stamp.
WRITE_STAMP=""
for cand in "${GP}/.write_agent_done.json" "${EX}/.write_agent_done.json"; do
  if [[ -f "$cand" ]]; then WRITE_STAMP="$cand"; break; fi
done
echo "WRITE stamp: ${WRITE_STAMP:-MISSING}"
if [[ -n "${WRITE_STAMP}" ]]; then
  python - <<'PY' "${WRITE_STAMP}"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  rebuild_required={d.get('rebuild_required', d.get('rebuild_needed'))}")
print(f"  cpp_changed={d.get('cpp_changed')}")
print(f"  devices_gt1_backend={d.get('devices_gt1_backend')}")
print(f"  completed_at={d.get('completed_at', d.get('utc'))}")
PY
else
  echo "  MISSING — do not submit until WRITE lands pair_style devices N"
fi

echo ""
echo "=== Pre-flight (login node) ==="
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312
python -m py_compile "${EX}/run_multigpu.py" "${EX}/collect_results.py" \
  "${EX}/parity_gates.py" "${EX}/write_multigpu_reports.py"
cd "${EX}"
python -c "from load_geometry import load_nacl6_fixed; a=load_nacl6_fixed(); print(f'geometry natoms={len(a)}')"
python -c "
from run_multigpu import uma_pair_style_line
assert uma_pair_style_line('double', 1) == 'pair_style uma/kk precision double devices 1'
assert uma_pair_style_line('double', 2) == 'pair_style uma precision double devices 2'
assert uma_pair_style_line('mixed', 4) == 'pair_style uma precision mixed devices 4'
print('pair_style line OK')
"

if [[ "${SUBMIT}" != "1" ]]; then
  echo ""
  echo "DRY-RUN only. To submit:"
  echo "  cd ${EX} && ./gp_round/rebuild_and_submit.sh --submit"
  echo "See gp_round/DRY_RUN_CHECKLIST.md"
  exit 0
fi

if [[ -z "${WRITE_STAMP}" ]]; then
  echo "ERROR: .write_agent_done.json missing (checked gp_round/ and suite root) — WRITE must land devices N first" >&2
  exit 2
fi
# Require evidence of devices N (not a stale pre-GP stamp).
if ! python - <<'PY' "${WRITE_STAMP}"
import json, sys
d = json.load(open(sys.argv[1]))
ok = bool(d.get("devices_gt1_backend") or d.get("pair_style_example") or
          ("devices" in str(d.get("pair_style_example", "")).lower()) or
          d.get("role") == "WRITE" and "graph" in json.dumps(d).lower())
sys.exit(0 if ok else 1)
PY
then
  echo "ERROR: WRITE stamp looks pre-GP (no devices_gt1_backend / pair_style_example) — refuse submit" >&2
  exit 2
fi

echo ""
echo "=== Rebuild ==="
export RECOMPILE=1
bash "${ROOT}/scripts/build_lammps_uma.sh"
test -x "${ROOT}/build-uma/lmp" || { echo "ERROR: build-uma/lmp missing" >&2; exit 1; }

mkdir -p "${GP}"

cd "${EX}"
# Path-isolated submit (one ONLY_PATHS per job). Mixed disabled — uma_double only.
export RECOMPILE=0  # already rebuilt above
if [[ "${SUBMIT_NGPU4}" == "1" ]]; then
  "${EX}/submit_path_jobs.sh" --gp --ngpus 1,2,4
else
  "${EX}/submit_path_jobs.sh" --gp --ngpus 1,2
fi
JOBIDS="${GP}/.jobids_isolated"
cp -f "${JOBIDS}" "${GP}/.jobids"
echo "Wrote ${JOBIDS} (and ${GP}/.jobids)"
cat "${JOBIDS}"
