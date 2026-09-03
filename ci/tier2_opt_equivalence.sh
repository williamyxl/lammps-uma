#!/bin/bash
# A1/S1 (audit rev 26 §G.18.6) — the Tier-2 NUMERIC opt-equivalence gate.
#
# This is the item the audit called "the only thing between A- and A": until now
# the opt2 (jit.freeze) / opt4 (no-recompute) / opt5 (UMA_CHUNK_RETAIN_K)
# equivalence claims were "numerically equivalent by construction" — asserted in
# comments and only ever exercised by the expensive XPU G4 suite. This gate
# MEASURES them: one CPU-traced toy artifact + real forwards through the actual
# engine, comparing energy and forces bit-for-bit across every memory strategy.
#
# Everything runs on a LOGIN NODE: no XPU, no allocation, no multi-GB artifact.
#
#   bash ci/tier2_opt_equivalence.sh            # SKIPs (exit 0) if env absent
#   bash ci/tier2_opt_equivalence.sh --strict   # SKIP -> exit 2 (automated caller)
#
# Requires: fxpu conda env (CPU LibTorch), cmake, and a toy artifact. Build the
# toy artifact once with:
#   OUT=$UMA_TOY_ARTIFACT BASE_META=<any metadata.json> N=1 \
#     python scripts/virial_make_atoms.py
#   python src/ML-UMA/uma-engine/python/w15_export_traced_fast.py \
#     --checkpoint $UMA_CHECKPOINT --output $UMA_TOY_ARTIFACT \
#     --atoms $UMA_TOY_ARTIFACT/nacl.extxyz --device cpu --dtype float64
# (~90 MB, so it is NOT committed; point UMA_TOY_ARTIFACT at it. TRACE on CPU
# works — the belief that a toy export needed an XPU was wrong.)
set -uo pipefail
LU="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENG="${LU}/src/ML-UMA/uma-engine"
BUILD="${ENG}/build-cpu-ci"
TOY="${UMA_TOY_ARTIFACT:-}"

REQUIRE="${UMA_CI_REQUIRE_OPTEQ:-0}"; [ "${1:-}" = "--strict" ] && REQUIRE=1
skip() { echo "OPTEQ SKIP: $1"; [ "$REQUIRE" = 1 ] && { echo "  (strict -> FAIL)"; exit 2; }; exit 0; }

# A1 close-out (audit rev 29 §G.27.5): if no artifact is provided, BUILD one so
# the gate runs in a fresh clone instead of SKIPping (~4 min CPU, login node). The
# 90 MB traced weights are too large to commit; building on demand also exercises
# the export path. UMA_TOY_NO_BUILD=1 forces the old skip-if-absent behaviour.
if [ -z "${TOY}" ] || [ ! -f "${TOY}/model_traced.pt" ] || [ ! -f "${TOY}/struct.txt" ]; then
  if [ "${UMA_TOY_NO_BUILD:-0}" = "1" ]; then
    skip "no toy artifact and UMA_TOY_NO_BUILD=1"
  fi
  echo "OPTEQ: no toy artifact -> building one (ci/build_toy_artifact.sh)"
  built="$(UMA_TOY_ARTIFACT="${TOY}" bash "${LU}/ci/build_toy_artifact.sh" ${1:+"$1"} | tail -1)" || \
    skip "toy artifact build failed"
  TOY="${built}"
fi
[ -f "${TOY}/model_traced.pt" ] || skip "no model_traced.pt in ${TOY}"
[ -f "${TOY}/struct.txt" ] || skip "no struct.txt in ${TOY}"
CLI="${BUILD}/uma_parity_cli"
[ -x "${CLI}" ] || skip "uma_parity_cli not built (run ci/tier2_cpu_build.sh first)"

TORCH_LIB="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null)" \
  || skip "no torch (activate fxpu first)"
export LD_LIBRARY_PATH="${BUILD}/compat:${CONDA_PREFIX:-}/lib:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

run_cfg() {  # $1 = "VAR=VAL" (or "" for baseline); echoes "energy fmax"
  local out
  out=$(env $1 "${CLI}" "${TOY}" "${TOY}/struct.txt" 2>/dev/null | grep -E "^n=") || return 1
  echo "$out" | sed -E 's/.*energy=([-0-9.]+) eV  fmax=([0-9.e+-]+).*/\1 \2/'
}

echo "=== A1/S1 opt-equivalence gate (CPU toy artifact: ${TOY}) ==="
BASE="$(run_cfg "UMA_CKPT=0")" || { echo "OPTEQ FAIL: baseline forward failed"; exit 1; }
echo "baseline (UMA_CKPT=0):        ${BASE}"

fail=0
# Every strategy below is a MEMORY strategy only; the math must be identical.
#   UMA_CKPT=1            opt2/AC: checkpoint the whole module (recompute in bwd)
#   UMA_NO_RECOMPUTE=1    opt4 master: retain ALL activations, never recompute
#   UMA_CHUNK_RETAIN_K=k  opt5: retain the first k edge-chunks per block
for cfg in "UMA_CKPT=1" "UMA_NO_RECOMPUTE=1" "UMA_NO_RECOMPUTE_BLOCK=1" \
           "UMA_NO_RECOMPUTE_EDEG=1" "UMA_CHUNK_RETAIN_K=1" \
           "UMA_CHUNK_RETAIN_K=2" "UMA_CHUNK_RETAIN_K=3"; do
  got="$(run_cfg "${cfg}")" || { echo "  ${cfg}: FORWARD FAILED"; fail=$((fail+1)); continue; }
  if [ "${got}" = "${BASE}" ]; then
    echo "  OK  ${cfg}: ${got}"
  else
    echo "  ⛔ MISMATCH ${cfg}: ${got}  (expected ${BASE})"
    fail=$((fail+1))
  fi
done

if [ "$fail" -gt 0 ]; then
  echo "OPTEQ FAIL: ${fail} configuration(s) are NOT numerically equivalent"
  exit 1
fi
echo "OPTEQ PASS (energy+forces bit-identical across all opt2/opt4/opt5 strategies)"
