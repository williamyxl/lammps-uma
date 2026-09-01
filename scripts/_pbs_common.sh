# Shared PBS preamble for UMA campaign jobs (P3.2 / P1.3).
#
# Source this AFTER `source .../activate_fxpu.sh` (the oneAPI setvars script
# references unset vars, so `set -u` before it crashes):
#
#   source "${HEN}/scripts/activate_fxpu.sh"
#   source "$(dirname "$0")/_pbs_common.sh"   # or an absolute path
#
# It enables strict-mode error handling so a failed step aborts the job instead of
# silently continuing and reporting a false PASS.
set -euo pipefail
