#!/bin/bash
# DEPRECATED: multi-path submit. Prefer path-isolated jobs.
#   ./submit_path_jobs.sh
#   ./submit_path_jobs.sh --gp
set -euo pipefail
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "NOTE: submit_all.sh now forwards to submit_path_jobs.sh (VRAM isolation)."
exec "${EX}/submit_path_jobs.sh" "$@"
