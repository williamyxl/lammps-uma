#!/bin/bash
# N-sweep drain poller: when no nsw-* job is active, advance the bisect.
# Logs transitions only. Stops when the driver reports DONE.
EX="$(cd "$(dirname "$0")" && pwd)"
LOG="${EX}/nsweep_poll.log"
CONDA=/u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
prev=""
while true; do
  active=$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -c '^nsw-')
  if [ "$active" -eq 0 ]; then
    # shellcheck disable=SC1090
    source "$CONDA" && conda activate uma312
    out=$(cd "$EX" && python nsweep_driver.py --next 2>&1 | tail -2)
    printf '%s %s\n' "$(date -Iseconds)" "$out" >> "$LOG"
    case "$out" in
      *DONE*) printf '%s sweep complete\n' "$(date -Iseconds)" >> "$LOG"; exit 0 ;;
    esac
  else
    now=$(squeue -u "$USER" -h -o '%i:%j:%T' 2>/dev/null | grep '^.*nsw-' | tr '\n' ' ')
    if [ "$now" != "$prev" ]; then
      printf '%s %s\n' "$(date -Iseconds)" "$now" >> "$LOG"
      prev="$now"
    fi
  fi
  sleep 180
done
