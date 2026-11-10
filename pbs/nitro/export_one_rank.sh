#!/bin/bash
# One rank of W=12 graph-parallel export under mpiexec + gpu_tile_compact.sh.
set -u
R="${PALS_RANKID:-${PMI_RANK:-}}"
if [[ -z "${R}" ]]; then
  echo "EXPORT_FAIL no global rank id" >&2
  exit 1
fi
export EXPORT_RANK="${R}"
export OMP_NUM_THREADS=1
LOG="${OUT}/exp_r${R}.log"
mkdir -p "${OUT}/w${EXPORT_WORLD}/r${R}"
python -u "${ENG_EXPORT}" > "${LOG}" 2>&1
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  echo "EXPORT_FAIL rank=${R} rc=${rc}" >&2
  tail -20 "${LOG}" >&2 || true
  exit "${rc}"
fi
if [[ ! -f "${OUT}/w${EXPORT_WORLD}/r${R}/model_traced.pt" ]]; then
  echo "EXPORT_FAIL rank=${R} missing model_traced.pt" >&2
  exit 1
fi
echo "export rank=${R} OK"
