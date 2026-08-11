#!/bin/bash
# V7 optimization loop driver. 2-minute tick; advances the 1 -> 2 -> 4 GPU
# ladder for a wave only when the previous rung has PASSED E/F.
#
#   bash v7_loop.sh <wave> [build_jobid]
#
# Ladder per wave: nacl6@1, water888@1 -> @2 -> @4.
# Escalation is gated on parity, not just completion: a rung that fails E/F
# stops the wave rather than spending GPU hours on wider runs of broken code.
CAMP="$(cd "$(dirname "$0")" && pwd)"
WAVE="${1:?usage: v7_loop.sh <wave> [build_jobid]}"
DEP="${2:-}"
LOG="${CAMP}/v7_loop_${WAVE}.log"
STATE="${CAMP}/.v7_state_${WAVE}"
CONDA=/u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
TICK=120

log() { printf '%s %s\n' "$(date -Iseconds)" "$*" >> "$LOG"; }

submit() {  # submit <sys> <ngpu>
  local sys=$1 ng=$2 dep_arg=()
  [ -n "$DEP" ] && dep_arg=(--dependency=afterok:"$DEP")
  sbatch --parsable "${dep_arg[@]}" \
    --gpus-per-node="$ng" --job-name="v7-${WAVE}-${sys}-n${ng}" \
    --export=ALL,WAVE="${WAVE}",SYS="${sys}",NG="${ng}" \
    "${CAMP}/v7_cell.slurm"
}

gate() {   # gate <sys> <ngpu> <jobid> -> rc 0 pass / 2 ef fail / 3 no results
  source "$CONDA" && conda activate uma312
  python "${CAMP}/v7_gate.py" --wave "$WAVE" --sys "$1" --ngpu "$2" \
         --job "$3" --write >> "$LOG" 2>&1
  return $?
}

jstate() {
  local s
  s=$(squeue -j "$1" -h -o '%T' 2>/dev/null | tr -d ' ')
  [ -z "$s" ] && s=$(sacct -j "$1" -n -X -o State 2>/dev/null | head -1 | tr -d ' ')
  echo "${s:-UNKNOWN}"
}

touch "$STATE"
declare -A JOB
# Ladder: widen only after the narrower rung passes parity.
LADDER=("nacl6:1" "water888:1" "nacl6:2" "water888:2" "nacl6:4" "water888:4")

log "V7 loop start wave=${WAVE} dep=${DEP:-none}"
for rung in "${LADDER[@]}"; do
  sys="${rung%%:*}"; ng="${rung##*:}"
  if grep -q "^done ${sys} ${ng} " "$STATE" 2>/dev/null; then
    log "skip ${sys}@${ng} (already done)"; continue
  fi
  jid=$(submit "$sys" "$ng")
  JOB["$sys:$ng"]=$jid
  log "submit ${sys}@${ng} job=${jid}"

  # wait for terminal state on the 2-min tick
  while true; do
    st=$(jstate "$jid")
    case "$st" in
      COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY) break ;;
    esac
    log "tick ${sys}@${ng} job=${jid} state=${st}"
    sleep "$TICK"
  done
  log "terminal ${sys}@${ng} job=${jid} state=${st}"

  gate "$sys" "$ng" "$jid"; rc=$?
  case $rc in
    0) log "GATE PASS ${sys}@${ng} job=${jid}"
       echo "done ${sys} ${ng} ${jid} PASS" >> "$STATE" ;;
    2) log "GATE EF_FAIL ${sys}@${ng} job=${jid} — STOPPING wave ${WAVE}"
       echo "done ${sys} ${ng} ${jid} EF_FAIL" >> "$STATE"
       log "V7 loop stop (parity regression)"; exit 2 ;;
    *) log "GATE NO_RESULTS ${sys}@${ng} job=${jid} — STOPPING wave ${WAVE}"
       echo "done ${sys} ${ng} ${jid} NO_RESULTS" >> "$STATE"
       log "V7 loop stop (cell produced nothing)"; exit 3 ;;
  esac
  # only the first cell needs the build dependency
  DEP=""
done

log "V7 loop complete wave=${WAVE}"
source "$CONDA" && conda activate uma312
python "${CAMP}/v7_summary.py" --wave "$WAVE" >> "$LOG" 2>&1
exit 0
