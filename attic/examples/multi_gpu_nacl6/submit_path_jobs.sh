#!/usr/bin/env bash
# Submit path-isolated jobs (one ONLY_PATHS per SLURM job — VRAM isolation).
#
# Usage:
#   ./submit_path_jobs.sh                  # ase/fc/double × ngpu1,2,4 (suite)
#   ./submit_path_jobs.sh --gp             # gp_round uma_double only
#   ./submit_path_jobs.sh --ngpus 1,2      # subset of GPU counts
#   ./submit_path_jobs.sh --paths ase,fc   # subset of paths
#   RECOMPILE=1 ./submit_path_jobs.sh --gp # rebuild on first job via env
#
# Mixed precision (uma_mixed) disabled — FP64 only.
# Jobs that share a results tree are chained with afterok to avoid merge races.
set -euo pipefail

EX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${EX}"

MODE=suite  # suite | gp
NGPUS_LIST=(1 2 4)
PATHS_SUITE=(ase fc uma_double)
PATHS_GP=(uma_double)
PATHS_CUSTOM=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gp) MODE=gp; shift ;;
    --suite) MODE=suite; shift ;;
    --ngpus)
      IFS=',' read -r -a NGPUS_LIST <<<"$2"
      shift 2
      ;;
    --paths)
      IFS=',' read -r -a PATHS_CUSTOM <<<"$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ "$MODE" == "gp" ]]; then
  if [[ ${#PATHS_CUSTOM[@]} -gt 0 ]]; then
    PATHS=("${PATHS_CUSTOM[@]}")
  else
    PATHS=("${PATHS_GP[@]}")
  fi
  JOBIDS="${EX}/gp_round/.jobids_isolated"
  script_for() { echo "${EX}/gp_round/run_ngpu${1}_${2}.slurm"; }
else
  if [[ ${#PATHS_CUSTOM[@]} -gt 0 ]]; then
    PATHS=("${PATHS_CUSTOM[@]}")
  else
    PATHS=("${PATHS_SUITE[@]}")
  fi
  JOBIDS="${EX}/.jobids_isolated"
  script_for() { echo "${EX}/run_ngpu${1}_${2}.slurm"; }
fi

mkdir -p "$(dirname "$JOBIDS")"
: > "${JOBIDS}"

prev=""
first=1
for n in "${NGPUS_LIST[@]}"; do
  for p in "${PATHS[@]}"; do
    script="$(script_for "$n" "$p")"
    if [[ ! -f "$script" ]]; then
      echo "ERROR: missing $script — run ./generate_path_jobs.sh" >&2
      exit 1
    fi
    export_list="ALL,RECOMPILE=0"
    if [[ "${RECOMPILE:-0}" == "1" && "$first" == "1" ]]; then
      export_list="ALL,RECOMPILE=1"
    fi
    dep=()
    if [[ -n "$prev" ]]; then
      dep=(--dependency="afterok:${prev}")
    fi
    jid=$(sbatch --parsable --export="${export_list}" "${dep[@]}" "$script")
    echo "$(basename "$script") ${jid}${prev:+ (afterok:${prev})}" | tee -a "${JOBIDS}"
    prev=$jid
    first=0
  done
done

echo "Wrote ${JOBIDS}"
cat "${JOBIDS}"
