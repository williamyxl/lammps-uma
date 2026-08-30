#!/usr/bin/env bash
# Submit Phase G1 OOM probes in parallel (independent RESULTS_DIR; no afterok).
#
# Usage:
#   ./submit_oom_sweep.sh                 # N=8,10,12 × all paths
#   ./submit_oom_sweep.sh --n 8,10        # subset
#   RECOMPILE=1 ./submit_oom_sweep.sh --n 8  # rebuild once then probes
set -euo pipefail

EX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
NS=(8 10 12)
# Mixed precision (uma_mixed) disabled — FP64 only.
PATHS=(ase fc uma_double)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) IFS=',' read -r -a NS <<<"$2"; shift 2 ;;
    --paths) IFS=',' read -r -a PATHS <<<"$2"; shift 2 ;;
    *) echo "unknown $1" >&2; exit 1 ;;
  esac
done

if [[ "${RECOMPILE:-0}" == "1" ]]; then
  echo "=== rebuild build-uma (RECOMPILE=1) ==="
  module unload cudatoolkit 2>/dev/null || true
  module load cuda/12.8 cmake/3.31.8
  source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
  conda activate uma312
  bash "${ROOT}/scripts/build_lammps_uma.sh"
fi

JOBIDS="${EX}/.jobids_oom_sweep"
: > "${JOBIDS}"
mkdir -p "${EX}/results/geom_sweep"

cd "${EX}"
for n in "${NS[@]}"; do
  nn=$(printf '%02d' "$n")
  for p in "${PATHS[@]}"; do
    rdir="${EX}/results/geom_sweep/N${nn}/${p}"
    mkdir -p "${rdir}"
    jid=$(sbatch --parsable \
      --job-name="oom-N${nn}-${p}" \
      --export=ALL,NACL_N=${n},ONLY_PATHS=${p},RESULTS_DIR=${rdir},RECOMPILE=0,N_TIMING=1,MERGE_RESULTS=0 \
      "${EX}/run_oom_probe.slurm")
    echo "N=${n} path=${p} job=${jid} dir=${rdir}" | tee -a "${JOBIDS}"
  done
done

echo "Wrote ${JOBIDS} (parallel; no afterok)"
cat "${JOBIDS}"
