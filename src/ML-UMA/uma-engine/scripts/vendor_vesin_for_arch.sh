#!/usr/bin/env bash
# Re-vendor vesin-torch shared library + header for the current CPU arch + LibTorch minor.
# uma-engine vendor: copy aarch64 wheel artifacts into third_party/vesin/.
#
# Usage:
#   bash scripts/vendor_vesin_for_arch.sh           # install if needed + copy
#   bash scripts/vendor_vesin_for_arch.sh --check   # verify vendored .so matches arch
#   bash scripts/vendor_vesin_for_arch.sh --force   # re-copy even if arch matches
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UMA_ENGINE_ROOT="${UMA_ENGINE_ROOT:-${REPO}}"
VESIN_DEST="${VESIN_ROOT:-${UMA_ENGINE_ROOT}/third_party/vesin}"
VESIN_VERSION="${VESIN_VERSION:-0.5.8}"
MODE="${1:-}"

mkdir -p "${VESIN_DEST}/lib" "${VESIN_DEST}/include"

HOST_ARCH="$(uname -m)"
case "${HOST_ARCH}" in
  x86_64) WANT_FILE="x86-64" ;;
  aarch64) WANT_FILE="aarch64" ;;
  *)
    echo "ERROR: unsupported uname -m=${HOST_ARCH}" >&2
    exit 1
    ;;
esac

_lib_arch() {
  file -b "$1" 2>/dev/null | grep -oE 'x86-64|aarch64|ARM aarch64' | head -1 || true
}

_check_vendored() {
  local so="${VESIN_DEST}/lib/libvesin_torch.so"
  [[ -f "${so}" ]] || return 1
  local got
  got="$(_lib_arch "${so}")"
  case "${HOST_ARCH}" in
    x86_64) [[ "${got}" == "x86-64" ]] ;;
    aarch64) [[ "${got}" == *"aarch64"* ]] ;;
  esac
}

if [[ "${MODE}" == "--check" ]]; then
  if _check_vendored; then
    echo "OK: ${VESIN_DEST}/lib/libvesin_torch.so matches ${HOST_ARCH}"
    file "${VESIN_DEST}/lib/libvesin_torch.so"
    exit 0
  fi
  echo "FAIL: vendored vesin missing or wrong arch (want ${WANT_FILE})" >&2
  file "${VESIN_DEST}/lib/libvesin_torch.so" 2>/dev/null || echo "(no file)" >&2
  exit 1
fi

if [[ "${MODE}" != "--force" ]] && _check_vendored; then
  echo "vesin already vendored for ${HOST_ARCH}: ${VESIN_DEST}/lib/libvesin_torch.so"
  file "${VESIN_DEST}/lib/libvesin_torch.so"
  exit 0
fi

if ! python3 -c "import torch" 2>/dev/null; then
  echo "ERROR: activate conda env with torch before vendoring vesin" >&2
  exit 1
fi

TORCH_MINOR="$(python3 -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")"
echo "Host arch=${HOST_ARCH} torch=${TORCH_MINOR} vesin-torch==${VESIN_VERSION}"

if ! python3 -c "import vesin.torch" 2>/dev/null; then
  echo ">>> pip install vesin-torch==${VESIN_VERSION}"
  pip install -q "vesin-torch==${VESIN_VERSION}"
fi

PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
WHEEL_ROOT="${CONDA_PREFIX:-}/lib/python${PYVER}/site-packages/vesin/torch/torch-${TORCH_MINOR}"
SO_SRC="${WHEEL_ROOT}/lib/libvesin_torch.so"
HDR_SRC="${WHEEL_ROOT}/include/vesin_torch.hpp"

if [[ ! -f "${SO_SRC}" ]]; then
  echo "ERROR: no vesin wheel for torch-${TORCH_MINOR} at ${WHEEL_ROOT}" >&2
  echo "Available:" >&2
  ls -d "${CONDA_PREFIX}/lib/python${PYVER}/site-packages/vesin/torch/torch-"* 2>/dev/null >&2 || true
  exit 1
fi

GOT="$(_lib_arch "${SO_SRC}")"
case "${HOST_ARCH}" in
  x86_64)
    [[ "${GOT}" == "x86-64" ]] || { echo "ERROR: pip wheel arch=${GOT}, want x86-64" >&2; exit 1; }
    ;;
  aarch64)
    [[ "${GOT}" == *"aarch64"* ]] || { echo "ERROR: pip wheel arch=${GOT}, want aarch64" >&2; exit 1; }
    ;;
esac

cp -f "${SO_SRC}" "${VESIN_DEST}/lib/libvesin_torch.so"
cp -f "${HDR_SRC}" "${VESIN_DEST}/include/vesin_torch.hpp"
echo "${VESIN_VERSION}" > "${VESIN_DEST}/VERSION"

echo "Vendored vesin-torch ${VESIN_VERSION} (torch-${TORCH_MINOR}) -> ${VESIN_DEST}"
file "${VESIN_DEST}/lib/libvesin_torch.so"
