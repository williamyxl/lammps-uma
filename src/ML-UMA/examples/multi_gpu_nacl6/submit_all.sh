#!/bin/bash
# Submit ngpu1/2/4 jobs; record jobids in .jobids
set -euo pipefail

EX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${EX}"

JOBIDS_FILE="${EX}/.jobids"
: > "${JOBIDS_FILE}"

for script in run_ngpu1.slurm run_ngpu2.slurm run_ngpu4.slurm; do
  if [[ ! -f "${script}" ]]; then
    echo "ERROR: missing ${script}" >&2
    exit 1
  fi
  jid=$(sbatch --parsable "${script}")
  echo "${script} ${jid}" | tee -a "${JOBIDS_FILE}"
done

echo "Wrote ${JOBIDS_FILE}"
cat "${JOBIDS_FILE}"
