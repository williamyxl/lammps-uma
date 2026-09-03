#!/bin/bash
# Tier 0 static guards (CODE_QUALITY.md Part C.3.0 / Sprint 4).
#
# Runs anywhere with just bash + python3 (no torch, no allocation, < 5 s). Two
# classes of check:
#   HARD  : must pass now — a failure here blocks a src/ML-UMA change.
#   REPORT: counts pre-existing debt that Sprint 6 (P3.2/P3.3/P5'.7) cleans up.
#           These flip to HARD at the end of Sprint 6 (set UMA_TIER0_STRICT=1 to
#           preview the strict behavior).
#
# Exit nonzero iff a HARD check fails (or any REPORT check fails under STRICT).
set -uo pipefail
LU="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LU"
STRICT="${UMA_TIER0_STRICT:-0}"
hard_fail=0
report_fail=0

say()  { printf '%s\n' "$*"; }
hdr()  { printf '\n== %s ==\n' "$*"; }

# ---- HARD 1: every tracked Python file parses -------------------------------
hdr "HARD: python ast-parse (src/ML-UMA python + scripts)"
py_bad=0
while IFS= read -r f; do
  python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" 2>/dev/null \
    || { say "  PARSE FAIL: $f"; py_bad=$((py_bad+1)); }
done < <(find src/ML-UMA -name '*.py' 2>/dev/null; ls scripts/*.py 2>/dev/null; ls ci/*.py 2>/dev/null)
if [ "$py_bad" -eq 0 ]; then say "  OK: all Python files parse"; else hard_fail=$((hard_fail+py_bad)); fi

# ---- HARD 2: uma_gates is the single tolerance source & imports -------------
hdr "HARD: uma_gates.py imports and exposes the gate table"
python3 - <<'PY' || hard_fail=$((hard_fail+1))
import sys
sys.path.insert(0, "scripts")
import uma_gates as g
for fn in ("e_tol_per_atom_mev","f_tol","agfd_tol","min_sample","fd_eps"):
    v = getattr(g, fn)()
    assert isinstance(v,(int,float)) and v>0, (fn,v)
print("  OK: uma_gates table", g.e_tol_per_atom_mev(), g.f_tol(), g.agfd_tol(),
      g.min_sample(), g.fd_eps())
PY

# ---- HARD 2b: comparators use uma_gates, not local tolerance copies [S9/P1.5]
# Any scripts/*{compare,parity}*.py that defines a local F_TOL/MIN_SAMPLE must also
# `import uma_gates` — otherwise editing the single source silently leaves that
# comparator on stale numbers (the G11 latent defect). Makes P1.5 an enforced
# invariant, not a one-time migration.
hdr "HARD: comparators source tolerances from uma_gates [S9/P1.5/G11]"
tol_bad=0
for f in scripts/*compare*.py scripts/*parity*.py; do
  [ -e "$f" ] || continue
  has_local=$(grep -cE '(^|[^_])(F_TOL|MIN_SAMPLE)[[:space:]]*=|environ(\.get\(|\[)"(F_TOL|MIN_SAMPLE)"' "$f")
  [ "$has_local" -eq 0 ] && continue
  grep -q "import uma_gates" "$f" || { say "  FLAG: $f defines local F_TOL/MIN_SAMPLE without importing uma_gates"; tol_bad=$((tol_bad+1)); }
done
if [ "$tol_bad" -eq 0 ]; then say "  OK: comparators single-source their tolerances"
else hard_fail=$((hard_fail+tol_bad)); fi

# ---- HARD 3: mandatory gate driver is fail-closed ---------------------------
hdr "HARD: mandatory gate n16_ase_parity.pbs uses set -euo pipefail"
if grep -q "set -euo pipefail" scripts/n16_ase_parity.pbs; then
  say "  OK"
else
  say "  FAIL: n16_ase_parity.pbs missing set -euo pipefail"; hard_fail=$((hard_fail+1))
fi

# ---- HARD 3b: barostat/virial guard uses nprocs, not stale mn_active (E1) ----
# init_style() runs before compute() sets mn_active, so the multi-node barostat
# refusal must key on comm->nprocs (valid at init) to avoid NPT-with-zero-virial on
# the GP path. Regression guard for audit finding E1.
hdr "HARD: barostat virial guard uses comm->nprocs (not stale mn_active) [E1]"
if grep -q "const bool multinode = (comm->nprocs > 1);" src/ML-UMA/pair_uma.cpp \
   && grep -q "virial_supported = !multinode" src/ML-UMA/pair_uma.cpp; then
  say "  OK"
else
  say "  FAIL: pair_uma.cpp virial_supported must derive from comm->nprocs (E1)"
  hard_fail=$((hard_fail+1))
fi

# ---- HARD 3c: exactly ONE CheckpointModuleFn definition (E.7.4 #1) -----------
# A duplicated custom autograd Function is the divergence risk that produced the
# P0'.3 silent-physics bug. The one definition lives in checkpoint_module.h; no TU
# may declare a private copy.
hdr "HARD: single CheckpointModuleFn definition (no duplicate) [E.7.4 #1]"
ckpt_defs=$(grep -rlE "struct[[:space:]]+CheckpointModuleFn" \
              src/ML-UMA/uma-engine/src src/ML-UMA/uma-engine/include 2>/dev/null | wc -l | tr -d ' ')
if [ "$ckpt_defs" -eq 1 ]; then say "  OK: 1 definition (checkpoint_module.h)"
else say "  FAIL: ${ckpt_defs} CheckpointModuleFn definitions (must be 1 — shared header)"; \
     grep -rlE "struct[[:space:]]+CheckpointModuleFn" src/ML-UMA/uma-engine/src \
       src/ML-UMA/uma-engine/include 2>/dev/null | sed 's/^/    /'; \
     hard_fail=$((hard_fail+1)); fi

FOREIGN='/work/nvme|/mnt/d|/u/xyan11|/opt/nvidia/hpc_sdk/[^ ]*/25\.3'

# ---- HARD 4: no foreign paths in COMPILED LIBRARY source --------------------
# The runtime C++/Python that ships in the pair style must not embed another
# machine's absolute paths (a silent wrong-file hazard). Excludes examples/,
# tests/, docs/, *.slurm, and spike_* (historical/demo, covered by REPORT below).
hdr "HARD: no foreign machine paths in compiled library source (P3.3/P5'.7)"
lib_fp=$(grep -rlE "${FOREIGN}" \
           src/ML-UMA/pair_uma.cpp src/ML-UMA/pair_uma.h \
           src/ML-UMA/uma-engine/src src/ML-UMA/uma-engine/include \
           src/ML-UMA/uma-engine/python 2>/dev/null \
         | grep -vE "/(tests|docs)/|/spike_|/attic/" | wc -l | tr -d ' ')
if [ "$lib_fp" -eq 0 ]; then say "  OK: library source clean"
else say "  FAIL: $lib_fp library file(s) with foreign paths:"; \
     grep -rlE "${FOREIGN}" src/ML-UMA/pair_uma.* src/ML-UMA/uma-engine/src \
       src/ML-UMA/uma-engine/include src/ML-UMA/uma-engine/python 2>/dev/null \
       | grep -vE "/(tests|docs)/|/spike_|/attic/" | sed 's/^/    /'; \
      hard_fail=$((hard_fail+lib_fp)); fi

# ---- HARD 4b: no hardcoded absolute user-workdir/hen path in ANY library python -
# G15/S7 (audit rev 16): the production exporters hardcoded another user's absolute
# `/lus/.../<user>/workdir/hen` path — a portability defect the FOREIGN pattern above
# missed (it is a current-machine project path, not a "foreign" one) and that HARD 4
# excluded for spike_*. Ban hardcoded absolute .../workdir/hen literals in ALL
# uma-engine python (spike included); use UMA_HEN_ROOT (uma_hen.py) instead.
hdr "HARD: no hardcoded absolute .../workdir/hen path in library python [G15/S7]"
hen_hard=$(grep -rlE '"/[^"]*/workdir/hen' src/ML-UMA/uma-engine/python 2>/dev/null \
             | grep -v "/uma_hen.py" | wc -l | tr -d ' ')
if [ "$hen_hard" -eq 0 ]; then say "  OK: no hardcoded hen path (uses UMA_HEN_ROOT)"
else say "  FAIL: $hen_hard python file(s) hardcode an absolute workdir/hen path:"; \
     grep -rlE '"/[^"]*/workdir/hen' src/ML-UMA/uma-engine/python 2>/dev/null \
       | grep -v "/uma_hen.py" | sed 's/^/    /'; hard_fail=$((hard_fail+hen_hard)); fi

# ---- HARD 5: every UMA_* env var read in library source is documented (E3) --
# Grep the compiled library source for getenv("UMA_*") / environ["UMA_*"] and fail
# if any name is missing from docs/ENV_VARS.md. Makes the ENV_VARS.md completeness
# claim TRUE and catches future undocumented vars (audit finding E3).
hdr "HARD: every UMA_* env var read in library source is documented [E3]"
env_undoc=0
if [ -f docs/ENV_VARS.md ]; then
  used=$(grep -rhoE 'getenv\("UMA_[A-Z0-9_]+"\)|environ(\.get\(|\[)"UMA_[A-Z0-9_]+"' \
           src/ML-UMA/pair_uma.cpp src/ML-UMA/pair_uma.h \
           src/ML-UMA/uma-engine/src src/ML-UMA/uma-engine/include \
           src/ML-UMA/uma-engine/python 2>/dev/null \
         | grep -oE 'UMA_[A-Z0-9_]+' | sort -u)
  for v in ${used}; do
    grep -q "\`${v}\`\|${v} " docs/ENV_VARS.md || { say "  UNDOCUMENTED: ${v}"; env_undoc=$((env_undoc+1)); }
  done
  if [ "$env_undoc" -eq 0 ]; then say "  OK: all UMA_* library env vars documented"
  else hard_fail=$((hard_fail+env_undoc)); fi
else
  say "  FAIL: docs/ENV_VARS.md missing"; hard_fail=$((hard_fail+1))
fi

# ---- HARD 6: vendored headers + CI harness are git-tracked [F.7.1/R1] --------
# The whole point of Sprint 3-6 (guards, vendored nlohmann/json, env pins) is only
# durable if it is in the repository. A clean clone must build and must carry the
# harness. This guard fails if any build-critical or harness file is untracked
# (the exact defect that took the grade A- -> B+ in rev 11).
hdr "HARD: vendored headers + CI harness are git-tracked [F.7.1/R1]"
untracked=0
for f in src/ML-UMA/uma-engine/third_party/nlohmann/json.hpp \
         ci/ci_local.sh ci/tier0_guards.sh ci/tier2_cpu_build.sh \
         pyproject.toml requirements.txt docs/ENV_VARS.md docs/TESTING.md \
         scripts/uma_gates.py scripts/_pbs_common.sh; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { say "  UNTRACKED: $f"; untracked=$((untracked+1)); }
done
# A hardcoded list can only catch files someone REMEMBERED to add: ci/asan_build.sh
# (§G.19) and the new §G.20 tests were all absent from it. Sweep the harness
# directories wholesale so a newly added guard/test cannot be silently untracked —
# the F.7.1 failure mode ("guards exist but are unguarded at HEAD") in miniature.
for f in $(ls ci/*.sh ci/*.supp ci/tests/*.py 2>/dev/null) \
         $(ls src/ML-UMA/uma-engine/tests/*.cpp 2>/dev/null); do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 \
    || { say "  UNTRACKED harness file: $f"; untracked=$((untracked+1)); }
done
# Also: every vendored header actually #included by compiled library source.
for hdr_inc in $(grep -rhoE '#include <nlohmann/[a-z_]+\.hpp>' \
                   src/ML-UMA/uma-engine/src 2>/dev/null | grep -oE 'nlohmann/[a-z_]+\.hpp' | sort -u); do
  git ls-files --error-unmatch "src/ML-UMA/uma-engine/third_party/${hdr_inc}" >/dev/null 2>&1 \
    || { say "  UNTRACKED vendored header: ${hdr_inc}"; untracked=$((untracked+1)); }
done
if [ "$untracked" -eq 0 ]; then say "  OK: build-critical + harness files tracked"
else hard_fail=$((hard_fail+untracked)); fi

# ---- REPORT 1: foreign paths in examples/slurm/docs (historical, Delta era) -
# Excludes build trees (compiled .o/.a bake the build-time path; they are gitignored
# and regenerated) and the attic/ (retired Delta-era files kept for reference).
hdr "REPORT: foreign machine paths in tracked non-build source (P5'.6/.7, Sprint 6)"
fp=$(grep -rlE "${FOREIGN}" src/ML-UMA 2>/dev/null \
      | grep -vE "/build-xpu|/build-cpp|/build-lmp|\.o$|\.a$|/attic/" \
      | wc -l | tr -d ' ')
say "  files with foreign paths (excl. build trees + attic): $fp (target 0)"
[ "$fp" -gt 0 ] && { report_fail=$((report_fail+1)); \
  grep -rlE "${FOREIGN}" src/ML-UMA 2>/dev/null \
    | grep -vE "/build-xpu|/build-cpp|/build-lmp|\.o$|\.a$|/attic/" | sed 's/^/    /'; }

# ---- HARD 4c: PairUMA:: method-size ratchet (E.9.1 + F.20-A5/A5b, G.6) ------
# Started as a compute()-only dispatcher guard (§E.8.3 #3); generalised per §F.20 A5
# to EVERY PairUMA:: method, so the next monolith (load_predictor = 216 L) is
# surfaced and a future long method cannot hide a defect. REPORT (advisory threshold);
# A5's exit criterion is "no function in pair_uma.cpp > 130 lines".
hdr "HARD: PairUMA:: method-size ratchet, <=130 except A5 baseline [G.6/F.20-A5b]"
over=$(awk '/^[a-zA-Z].*PairUMA::[a-zA-Z_]+\(/{name=$0; s=NR}
            s&&/^\}/{n=NR-s+1; if(n>130){sub(/\(.*/,"",name); sub(/.*PairUMA::/,"",name); print n" "name} s=0}' \
          src/ML-UMA/pair_uma.cpp 2>/dev/null)
# RATCHET (G.6 / F.20-A5b): the two known over-130 methods (load_predictor,
# run_compute_dd) are informational A5 lift targets; but a NEW over-130 method, or
# any GROWTH of the set, HARD-fails — an open-ended "informational" state is how
# compute() reached 225 lines. Baseline = the exact known set; anything outside it
# fails. When A5 splits these, lower the baseline (ideally to none).
# A5c (G.12.2): the ratchet bounds baseline SIZE, not just membership — a baseline
# method may only SHRINK, never grow (load_predictor drifted 216->218 in one commit
# under a membership-only ratchet). Baseline = "name:max_lines"; a new over-130
# method, OR a baseline method exceeding its recorded size, HARD-fails. State can
# only improve. When A5 splits a method below 130, drop its baseline entry.
# A5 DONE (audit rev 26 §G.18.6): load_predictor() split into init_mpi_peer() +
# uma_select_compute_device() (both <130), run_compute_dd() split into
# pad_dd_edges() (<130). Baseline is now EMPTY — A5's exit criterion "no method
# > 130 lines" is met and the ratchet HARD-fails on ANY over-130 method.
UMA_A5_BASELINE=""
new_over=0; grew=0
if [ -n "$over" ]; then
  while read -r n nm; do
    [ -z "$nm" ] && continue
    cap=""; for b in ${UMA_A5_BASELINE}; do
      [ "${b%%:*}" = "$nm" ] && cap="${b##*:}"; done
    if [ -z "$cap" ]; then
      say "  ⛔ NEW OVER-130 method (ratchet): ${nm}() = ${n} lines — split it (F.20-A5)"; new_over=$((new_over+1))
    elif [ "$n" -gt "$cap" ]; then
      say "  ⛔ BASELINE GREW (A5c ratchet): ${nm}() = ${n} lines > recorded ${cap} — shrink it, don't grow it"; grew=$((grew+1))
    else
      say "  OVER-130 (known A5 target, <= recorded ${cap}): ${nm}() = ${n} lines"
    fi
  done <<< "$over"
fi
if [ "$((new_over+grew))" -gt 0 ]; then
  say "  FAIL: ${new_over} new + ${grew} grown over-130 method(s) (A5/A5c ratchet)"; hard_fail=$((hard_fail+new_over+grew))
else
  say "  OK: over-130 set within the A5 baseline sizes (${UMA_A5_BASELINE})"
fi

# ---- REPORT 2: .pbs without set -euo pipefail -------------------------------
hdr "REPORT: scripts/*.pbs without 'set -euo pipefail' (P1.3/P3.2, Sprint 6)"
tot=0; miss=0
for f in scripts/*.pbs; do
  [ -e "$f" ] || continue
  tot=$((tot+1))
  # Accept `set -euo pipefail` OR `set -uo pipefail` — fail-expecting tests (that
  # intentionally run a command expected to exit nonzero) legitimately omit -e.
  grep -qE "set -e?uo pipefail" "$f" || miss=$((miss+1))
done
say "  $miss of $tot .pbs missing the preamble (target 0 by end of Sprint 6)"
[ "$miss" -gt 0 ] && report_fail=$((report_fail+1))

# ---- REPORT 3: library python importing spike_*/attic modules --------------
hdr "REPORT: library modules importing spike_*/attic (P5'.6, Sprint 6)"
si=$(grep -rlE "import (spike_|attic)" src/ML-UMA/uma-engine/python 2>/dev/null \
      | grep -vE "spike_|test_" | wc -l | tr -d ' ')
say "  library modules importing spike_/attic: $si (target 0)"
[ "$si" -gt 0 ] && report_fail=$((report_fail+1))

# ---- summary ----------------------------------------------------------------
hdr "Tier 0 summary"
say "  HARD failures:   $hard_fail"
say "  REPORT findings: $report_fail (Sprint 6 cleanup debt)"
if [ "$hard_fail" -gt 0 ]; then
  say "TIER0 FAIL (hard)"; exit 1
fi
if [ "$STRICT" = "1" ] && [ "$report_fail" -gt 0 ]; then
  say "TIER0 FAIL (strict: report findings present)"; exit 1
fi
say "TIER0 PASS${STRICT:+ (strict=$STRICT)}"
exit 0
