"""Python-side registration of the DD k=4 halo-exchange op for tracing.

The C++ engine registers ``uma_halo::exchange`` (halo_context.cpp) for RUNTIME,
but ``torch.jit.trace`` in the exporter runs in a Python process that does not
load libuma_engine, so the op must also exist as a ``torch.library`` op to be
recorded into the traced graph. This mirrors ``uma_peer_ops`` / ``uma_ckpt_ops``.

At trace time the exchange is the IDENTITY: the single-process exporter has no
ghosts, and the traced graph only needs the op NODE present. At runtime the C++
HaloContext performs the real owned<->ghost movement (forward refresh, backward
ghost-grad -> owner accumulate) via LAMMPS comm.
"""
from __future__ import annotations

import torch
from torch.library import Library, impl

_LIB_NAME = "uma_halo"
_lib: Library | None = None


def _ensure_lib() -> Library:
    global _lib
    if _lib is not None:
        return _lib
    _lib = Library(_LIB_NAME, "DEF")
    _lib.define("exchange(Tensor x) -> Tensor")

    @impl(_lib, "exchange", "CompositeExplicitAutograd")
    def _exchange_impl(x: torch.Tensor) -> torch.Tensor:
        # Trace stand-in: identity. Runtime C++ overwrites ghost rows (fwd) and
        # accumulates ghost grads onto owners (bwd). Return a view-free clone so
        # the traced graph records a real op boundary.
        return x.clone()

    return _lib


def install_halo_ops() -> None:
    _ensure_lib()
