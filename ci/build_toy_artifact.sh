#!/bin/bash
# A1 close-out (audit rev 29 §G.27.5): build the CPU toy artifact that
# ci/tier2_opt_equivalence.sh needs, so the A1 numeric gate RUNS in a fresh clone
# instead of SKIPping. The traced UMA-s weights are ~90 MB, too large to commit;
# instead we build the artifact on demand from the checkpoint (~4 min, CPU, login
# node, no allocation). The export path is itself exercised as a side benefit.
#
#   bash ci/build_toy_artifact.sh                 # -> $UMA_TOY_ARTIFACT (or default)
#   bash ci/build_toy_artifact.sh --strict         # SKIP -> exit 2
#
# Emits, in $OUT (default ci/.toy_artifact):
#   model_traced.pt + metadata.json   (w15 CPU trace, 8-atom NaCl)
#   struct.txt                         (uma_parity_cli input for the same system)
# Idempotent: re-uses an existing valid artifact.
set -uo pipefail
LU="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENG="${LU}/src/ML-UMA/uma-engine"
OUT="${UMA_TOY_ARTIFACT:-${LU}/ci/.toy_artifact}"
CKPT="${UMA_CHECKPOINT:-${HEN:-/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen}/uma-cache/uma-s-1p2.pt}"
BASE_META="${UMA_TOY_BASE_META:-}"

REQUIRE="${UMA_CI_REQUIRE_TOY:-0}"; [ "${1:-}" = "--strict" ] && REQUIRE=1
skip() { echo "TOY SKIP: $1"; [ "$REQUIRE" = 1 ] && { echo "  (strict -> FAIL)"; exit 2; }; exit 0; }

# Already built? (model + struct present)
if [ -f "${OUT}/model_traced.pt" ] && [ -f "${OUT}/struct.txt" ]; then
  echo "TOY OK (cached): ${OUT}"; echo "${OUT}"; exit 0
fi

python -c 'import torch' 2>/dev/null || skip "no torch (activate fxpu first)"
[ -f "${CKPT}" ] || skip "no checkpoint at ${CKPT} (set UMA_CHECKPOINT)"
# a base metadata.json to seed the plain (non-AC) toy metadata
if [ -z "${BASE_META}" ]; then
  BASE_META="$(find "${LU}/scripts/out" -name metadata.json 2>/dev/null | head -1)"
fi
[ -n "${BASE_META}" ] && [ -f "${BASE_META}" ] || skip "no base metadata.json to seed from (set UMA_TOY_BASE_META)"

mkdir -p "${OUT}"
export HF_HUB_OFFLINE=1 FAIRCHEM_OFFLINE=1 PYTHONUNBUFFERED=1 UMA_CHECKPOINT="${CKPT}"

echo "TOY: geometry (8-atom NaCl) ..."
OUT="${OUT}" BASE_META="${BASE_META}" N=1 python "${LU}/scripts/virial_make_atoms.py" \
  || { echo "TOY FAIL: make_atoms"; exit 1; }

echo "TOY: CPU trace export (w15, ~4 min) ..."
PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" python "${ENG}/python/w15_export_traced_fast.py" \
  --checkpoint "${CKPT}" --output "${OUT}" --atoms "${OUT}/nacl.extxyz" \
  --device cpu --dtype float64 \
  || { echo "TOY FAIL: w15 export"; exit 1; }

echo "TOY: struct.txt for uma_parity_cli ..."
python - "${OUT}" <<'PY' || { echo "TOY FAIL: struct.txt"; exit 1; }
import sys
out = sys.argv[1]
Z = {"Na": 11, "Cl": 17}
lines = open(f"{out}/nacl.extxyz").read().splitlines()
n = int(lines[0]); rows = [str(n)]
for l in lines[2:2 + n]:
    p = l.split(); rows.append(f"{Z[p[0]]} {p[1]} {p[2]} {p[3]}")
rows.append("5.64 0 0 0 5.64 0 0 0 5.64")
open(f"{out}/struct.txt", "w").write("\n".join(rows) + "\n")
print(f"wrote {out}/struct.txt")
PY

echo "TOY BUILT: ${OUT}"
echo "${OUT}"
