"""Export stand-in for uma_sycl::silu_affine. Tracing only.

The schema string is the one in pbs/nitro/s1/uma_sycl_schema.h. The C++ kernel
replaces this at load time.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch.library import Library, impl

_SCHEMA_H = Path(__file__).resolve().parents[1] / "s1" / "uma_sycl_schema.h"
_LIB: Library | None = None


def schema() -> str:
    text = _SCHEMA_H.read_text()
    match = re.search(r'"([^"]+)"', text.split("UMA_SYCL_SILU_AFFINE_SCHEMA", 1)[1])
    if match is None:
        raise RuntimeError(f"no schema string in {_SCHEMA_H}")
    return match.group(1)


def ensure() -> Library:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = Library("uma_sycl", "DEF")
    lib.define(schema())

    @impl(lib, "silu_affine", "CompositeExplicitAutograd")
    def _silu_affine(x, w, b):
        z = x * w + b
        return z * torch.sigmoid(z)

    _LIB = lib
    return lib
