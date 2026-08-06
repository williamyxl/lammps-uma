#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
source /home/xyan11/miniforge3/etc/profile.d/conda.sh && conda activate uma312
LMP="$ROOT/lammps/build-uma/lmp"
TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
VESIN="$ROOT/uma-engine/third_party/vesin/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda/lib64:${VESIN}:${TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$(dirname "$0")"
exec "$LMP" -k on g 1 -sf kk -in in.nacl_sp "$@"
