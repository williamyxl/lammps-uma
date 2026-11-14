#!/bin/bash
# icpx line for the step 12 op. Flags match the step 9 SyclExtension build:
# spir64_gen and spir64, AOT for pvc. Quoting stays in this script.
set -euo pipefail
ICPX="${ICPX:-/opt/aurora/26.181.0/oneapi/compiler/latest/bin/icpx}"
exec "${ICPX}" -fsycl -O2 -std=c++17 -fPIC -fp-model=precise \
  -D_GLIBCXX_USE_CXX11_ABI=1 \
  -fsycl-targets=spir64_gen,spir64 \
  -Xs "-device pvc" \
  "$@"
