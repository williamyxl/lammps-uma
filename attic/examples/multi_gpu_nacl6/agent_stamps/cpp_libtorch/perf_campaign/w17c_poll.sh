#!/bin/bash
# W17c drain poller: watch export + NaCl@2 graph gate, log transitions only.
# Usage: bash w17c_poll.sh [export_jobid] [gate_jobid]
CAMP="$(cd "$(dirname "$0")" && pwd)"
EXP="${1:-21015028}"
GATE="${2:-21015029}"
LOG="${CAMP}/w17c_poll.log"
prev=""
while true; do
  now=""
  for j in "$EXP" "$GATE"; do
    s=$(squeue -j "$j" -h -o '%T' 2>/dev/null | tr -d ' ')
    if [ -z "$s" ]; then
      s=$(sacct -j "$j" -n -X -o State 2>/dev/null | head -1 | tr -d ' ')
    fi
    [ -z "$s" ] && s=UNKNOWN
    now="${now}${j}=${s} "
  done
  if [ "$now" != "$prev" ]; then
    printf '%s %s\n' "$(date -Iseconds)" "$now" >> "$LOG"
    prev="$now"
  fi
  case "$now" in
    *"${GATE}=COMPLETED"*|*"${GATE}=FAILED"*|*"${GATE}=CANCELLED"*|*"${GATE}=TIMEOUT"*)
      printf '%s TERMINAL gate=%s -> run w17_gate.py\n' "$(date -Iseconds)" "$GATE" >> "$LOG"
      exit 0
      ;;
  esac
  sleep 120
done
