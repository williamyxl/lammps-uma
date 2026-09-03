#!/bin/bash
# Tier 2 CI: CPU-only LibTorch build of uma_engine + CTest, on a login node.
#
# Builds the engine with UMA_ENGINE_USE_CUDA=OFF / UMA_ENGINE_USE_XPU=OFF against
# the CPU LibTorch that ships inside the fxpu conda torch (libtorch_cpu). No XPU, no
# CUDA, no allocation. Compiles the hermetic C++ self-tests and runs `ctest`
# (graph_shard_smoke = GP node-partition coverage, P1.6). This proves the C++ core
# compiles + the registered CTest passes without a compute node.
#
# Usage (login node, fxpu activated):
#   source /lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/scripts/activate_fxpu.sh
#   bash ci/tier2_cpu_build.sh
#
# E.10.2 (audit rev 9): by default a missing env/cmake prints SKIP and exits 0
# (developer convenience). In a non-interactive/automated caller that reads the
# exit code, that is FAIL-OPEN — "build passed" is indistinguishable from "build
# never ran". Set UMA_CI_REQUIRE_TIER2=1 (or pass --strict) to turn every SKIP
# into exit 2, so an automated runner cannot read a missing env as green.
set -uo pipefail
LU="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENG="${LU}/src/ML-UMA/uma-engine"
BUILD="${ENG}/build-cpu-ci"

REQUIRE_TIER2="${UMA_CI_REQUIRE_TIER2:-0}"
[ "${1:-}" = "--strict" ] && REQUIRE_TIER2=1
# Emit a SKIP: fail-closed (exit 2) under strict, permissive (exit 0) otherwise.
skip() { echo "TIER2 SKIP: $1"; [ "${REQUIRE_TIER2}" = "1" ] && { echo "  (UMA_CI_REQUIRE_TIER2=1 -> treating SKIP as FAILURE)"; exit 2; }; exit 0; }

module load cmake 2>/dev/null || true
command -v cmake >/dev/null 2>&1 || skip "cmake not available"

TORCH_PREFIX="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)/share/cmake"
if [ ! -d "${TORCH_PREFIX}" ]; then
  skip "no torch cmake prefix (activate fxpu first)"
fi
echo "Torch cmake prefix: ${TORCH_PREFIX}"

# libsycl compat dir (fxpu torch is the XPU build; libtorch_cpu needs libsycl.so.9,
# link expects .so.8) — injected into the CTest ENVIRONMENT via UMA_CTEST_LD_PREFIX.
COMPAT="${BUILD}/compat"; mkdir -p "${COMPAT}"
if [ -e "${CONDA_PREFIX:-}/lib/libsycl.so.9" ]; then
  ln -sf "${CONDA_PREFIX}/lib/libsycl.so.9" "${COMPAT}/libsycl.so.8"
fi

rm -rf "${BUILD}/CMakeCache.txt" "${BUILD}/CMakeFiles" 2>/dev/null || true
cmake -S "${ENG}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUMA_ENGINE_USE_CUDA=OFF -DUMA_ENGINE_USE_XPU=OFF \
  -DCMAKE_PREFIX_PATH="${TORCH_PREFIX}" \
  -DUMA_CTEST_LD_PREFIX="${COMPAT}:${CONDA_PREFIX:-}/lib" \
  >/dev/null 2>"${BUILD}.cmake.err" || { echo "TIER2 FAIL: cmake configure"; tail -20 "${BUILD}.cmake.err"; exit 1; }

# Build the library + the CPU hermetic self-tests (not the whole tree).
cmake --build "${BUILD}" --target uma_engine graph_shard_smoke \
  test_m0_device_binding test_m3_gather_scatter \
  test_lifetime_asan test_transport_table -j 16 \
  >"${BUILD}.build.log" 2>&1 || { echo "TIER2 FAIL: build"; tail -30 "${BUILD}.build.log"; exit 1; }
echo "TIER2 build OK: libuma_engine + graph_shard_smoke + test_m0 + test_m3 + lifetime + transport_table"

# fxpu's torch is the XPU build, so even libtorch_cpu pulls in libsycl.so.9; the
# link warns it "may conflict with libsycl.so.8". Provide a compat shim + torch/
# conda lib dirs so the CTest binary loads the right SYCL at runtime.
TORCH_LIB="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"
COMPAT="${BUILD}/compat"; mkdir -p "${COMPAT}"
if [ -e "${CONDA_PREFIX:-}/lib/libsycl.so.9" ]; then
  ln -sf "${CONDA_PREFIX}/lib/libsycl.so.9" "${COMPAT}/libsycl.so.8"
fi
export LD_LIBRARY_PATH="${COMPAT}:${CONDA_PREFIX:-}/lib:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

# CTest: run ALL CPU-registered tests. A3/A7 (audit rev 26 §G.18.6) added
# test_lifetime_asan (lifetime cycles) and test_transport_table (transport enum /
# name / stride contract); they were registered but Tier-2 was not running them —
# the non-ASAN CPU gate now covers the same set the ASAN gate does.
( cd "${BUILD}" && ctest --output-on-failure \
    -R "graph_shard_smoke|test_m0_device_binding|test_m3_gather_scatter|test_lifetime_asan|test_transport_table" ) \
  || { echo "TIER2 FAIL: ctest"; exit 1; }

echo "TIER2 PASS"
