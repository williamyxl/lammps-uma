#!/bin/bash
# Local CI driver (CODE_QUALITY.md Sprint 4 / Part C.3).
#
# Runs on a login node in < 1 minute, NO queue allocation, NO checkpoint, NO XPU:
#   Tier 0 : static/grep guards        (bash + python3)
#   Tier 1 : hermetic unit tests        (python3 + numpy; pytest optional)
#
# Usage:
#   bash ci/ci_local.sh              # Tier 0 + Tier 1 (plain python3 runners)
#   bash ci/ci_local.sh --pytest     # also run via pytest if available
#   UMA_TIER0_STRICT=1 bash ci/ci_local.sh   # treat Sprint-6 debt as hard fail
#
# Exit nonzero if any tier fails. A src/ML-UMA change must keep this green.
set -uo pipefail
LU="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LU"
rc=0

echo "############ Tier 0: static guards ############"
# Sprint 6 cleared the REPORT debt (foreign paths, .pbs preambles, spike imports),
# so STRICT is the default now — a regression hard-fails. Override with
# UMA_TIER0_STRICT=0 only if intentionally reintroducing tracked debt.
UMA_TIER0_STRICT="${UMA_TIER0_STRICT:-1}" bash ci/tier0_guards.sh || rc=1

echo
echo "############ Tier 1: hermetic unit tests (python3) ############"
t1_fail=0
for t in ci/tests/test_*.py; do
  [ -e "$t" ] || continue
  echo "--- $t ---"
  python3 "$t" || { t1_fail=$((t1_fail+1)); }
done
if [ "$t1_fail" -gt 0 ]; then echo "TIER1 FAIL ($t1_fail file(s))"; rc=1; else echo "TIER1 PASS"; fi

if [ "${1:-}" = "--tier2" ]; then
  echo
  echo "############ Tier 2: CPU engine build + CTest ############"
  # Needs the fxpu env (torch + cmake); ~3-5 min first build. Login node, no alloc.
  bash ci/tier2_cpu_build.sh || rc=1
fi

if [ "${1:-}" = "--asan" ]; then
  echo
  echo "############ ASAN: CPU engine + lifetime harness (A3) ############"
  # -fsanitize=address CPU build; runs test_lifetime_asan (P0'.4 dangling-callback
  # redefine + A3 Shm control-block cycle) + CPU CTests under ASAN. Login node.
  bash ci/asan_build.sh || rc=1
fi

if [ "${1:-}" = "--pytest" ]; then
  echo
  echo "############ Tier 1 via pytest (if available) ############"
  if python3 -c "import pytest" 2>/dev/null; then
    python3 -m pytest -q ci/tests || rc=1
  else
    echo "pytest not installed in this env — skipped (plain runners already ran)"
  fi
fi

echo
if [ "$rc" -eq 0 ]; then echo "CI_LOCAL PASS"; else echo "CI_LOCAL FAIL"; fi
exit $rc
