#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
EXAMPLE="$(cd "$(dirname "$0")" && pwd)"

source /home/xyan11/miniforge3/etc/profile.d/conda.sh
conda activate uma312

LMP="${LMP_BIN:-$ROOT/lammps/build-uma/lmp}"
if [[ ! -x "$LMP" ]]; then
  LMP="$ROOT/lammps/build-uma/lmp_kokkos_cuda"
fi
if [[ ! -x "$LMP" ]]; then
  echo "ERROR: LAMMPS binary not found. Build with PKG_KOKKOS + PKG_ML-UMA first." >&2
  echo "  Expected: $ROOT/lammps/build-uma/lmp" >&2
  exit 1
fi

TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ARTIFACT="${UMA_ARTIFACT:-$ROOT/lammps/src/ML-UMA/uma-engine/artifacts/uma-s-1p2-omat}"
if [[ ! -f "$ARTIFACT/model_traced.pt" ]]; then
  echo "ERROR: missing artifact at $ARTIFACT (run export_omat.py first)" >&2
  exit 1
fi

cd "$EXAMPLE"
# Rewrite pair_coeff path if needed via env — in script uses relative path.
exec "$LMP" -k on g 1 -sf kk -in in.nacl_minim "$@"
