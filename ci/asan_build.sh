#!/bin/bash
# A3 / G.5 item 5 (audit rev 26 §G.18): AddressSanitizer CPU build of the engine +
# the lifetime harness, on a login node. Parity runs prove numbers are stable; they
# cannot see a use-after-free / double-free / uninitialised-mutex. This builds the
# CPU engine with -fsanitize=address and runs test_lifetime_asan (the P0'.4
# dangling-callback redefine cycle + the A3 Shm control-block init/destroy cycle) +
# graph_shard_smoke/test_m0/test_m3 under ASAN.
#
#   bash ci/asan_build.sh            # SKIPs (exit 0) if env absent
#   bash ci/asan_build.sh --strict   # SKIP -> exit 2 (for automated callers)
#
# Requires the fxpu conda env (CPU LibTorch) + cmake. No XPU, no allocation.
set -uo pipefail
LU="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENG="${LU}/src/ML-UMA/uma-engine"
BUILD="${ENG}/build-asan-ci"
REQUIRE="${UMA_CI_REQUIRE_ASAN:-0}"; [ "${1:-}" = "--strict" ] && REQUIRE=1
skip() { echo "ASAN SKIP: $1"; [ "$REQUIRE" = 1 ] && { echo "  (strict -> FAIL)"; exit 2; }; exit 0; }

module load cmake 2>/dev/null || true
command -v cmake >/dev/null 2>&1 || skip "cmake not available"
TORCH_PREFIX="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)/share/cmake"
[ -d "${TORCH_PREFIX}" ] || skip "no torch cmake prefix (activate fxpu first)"

COMPAT="${BUILD}/compat"; mkdir -p "${COMPAT}"
[ -e "${CONDA_PREFIX:-}/lib/libsycl.so.9" ] && ln -sf "${CONDA_PREFIX}/lib/libsycl.so.9" "${COMPAT}/libsycl.so.8"

rm -rf "${BUILD}/CMakeCache.txt" "${BUILD}/CMakeFiles" 2>/dev/null || true
ASAN="-fsanitize=address -fno-omit-frame-pointer -g"
cmake -S "${ENG}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUMA_ENGINE_USE_CUDA=OFF -DUMA_ENGINE_USE_XPU=OFF \
  -DCMAKE_PREFIX_PATH="${TORCH_PREFIX}" \
  -DUMA_CTEST_LD_PREFIX="${COMPAT}:${CONDA_PREFIX:-}/lib" \
  -DCMAKE_CXX_FLAGS="${ASAN}" -DCMAKE_EXE_LINKER_FLAGS="${ASAN}" \
  >/dev/null 2>"${BUILD}.cmake.err" || { echo "ASAN FAIL: configure"; tail -20 "${BUILD}.cmake.err"; exit 1; }

cmake --build "${BUILD}" --target uma_engine test_lifetime_asan \
  graph_shard_smoke test_m0_device_binding test_m3_gather_scatter -j 16 \
  >"${BUILD}.build.log" 2>&1 || { echo "ASAN FAIL: build"; tail -30 "${BUILD}.build.log"; exit 1; }
echo "ASAN build OK"

# LSan can't unwind through some torch static dtors on this stack; leak-check the
# harness itself but do not fail on torch's own one-time allocations.
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:${ASAN_OPTIONS:-}"
export LSAN_OPTIONS="suppressions=${LU}/ci/lsan.supp:${LSAN_OPTIONS:-}"
export LD_LIBRARY_PATH="${COMPAT}:${CONDA_PREFIX:-}/lib:$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"

( cd "${BUILD}" && ctest --output-on-failure \
    -R "test_lifetime_asan|graph_shard_smoke|test_m0_device_binding|test_m3_gather_scatter" ) \
  || { echo "ASAN FAIL: ctest (sanitizer found a memory error)"; exit 1; }

echo "ASAN PASS (lifetime harness + CPU CTests clean under -fsanitize=address)"
