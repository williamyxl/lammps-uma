"""Edge-chunked FairChem ``prepare_wigner`` for Intel XPU FP64 autograd.

Intel XPU can give incorrect FP64 reverse-mode gradients when UMA's first
full-edge Wigner-preparation einsum is one large contraction::

    torch.einsum("mk,nkj->nmj", to_m, wigner)

Forward energy stays correct; autograd forces disagree with -dE/dx above
~2e5 edges. Chunking only the edge dimension is the same math as the stock
FairChem kernel, split along E. Disable with ``FXPU_SKIP_WIGNER_PREP_CHUNK=1``.

This file lives in uma-engine so export does not depend on a sibling hen/ tree.
"""
from __future__ import annotations

import os


def apply_xpu_prepare_wigner_chunking(
    chunk_size: int | None = None,
    mode: str | None = None,
    *,
    force: bool = False,
) -> str:
    """Install an edge-chunked ``ExecutionBackend.prepare_wigner``.

    Idempotent for the same ``(chunk_size, mode)``. Pass ``force=True`` or
    change env/args to retune.
    """
    import torch
    from fairchem.core.models.uma.nn.execution_backends import ExecutionBackend

    if chunk_size is None:
        chunk_size = int(os.environ.get("FXPU_WIGNER_PREP_CHUNK", "65536"))
    if mode is None:
        mode = os.environ.get("FXPU_WIGNER_PREP_CHUNK_MODE", "both")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if mode not in ("both", "forward", "inverse"):
        raise ValueError(f"bad Wigner prepare chunk mode: {mode!r}")

    prev = getattr(ExecutionBackend, "_fxpu_wigner_prepare_cfg", None)
    cfg = (int(chunk_size), str(mode))
    if (
        not force
        and getattr(ExecutionBackend, "_fxpu_wigner_prepare_chunked", False)
        and prev == cfg
    ):
        return f"wignerPrepareChunked(chunk={cfg[0]},mode={cfg[1]})"

    _chunk = int(chunk_size)
    _mode = str(mode)

    @staticmethod
    def prepare_wigner(
        wigner,
        wigner_inv,
        mapping_reduced,
        coefficient_index,
    ):
        if coefficient_index is not None:
            wigner = wigner.index_select(1, coefficient_index)
            wigner_inv = wigner_inv.index_select(2, coefficient_index)
        to_m = mapping_reduced.to_m.to(wigner.dtype)
        if _mode in ("both", "forward"):
            forward_parts = []
            for start in range(0, wigner.shape[0], _chunk):
                end = min(start + _chunk, wigner.shape[0])
                forward_parts.append(
                    torch.einsum("mk,nkj->nmj", to_m, wigner[start:end])
                )
            wigner_prepared = torch.cat(forward_parts, dim=0)
        else:
            wigner_prepared = torch.einsum(
                "mk,nkj->nmj", to_m, wigner
            )
        if _mode in ("both", "inverse"):
            inverse_parts = []
            for start in range(0, wigner_inv.shape[0], _chunk):
                end = min(start + _chunk, wigner_inv.shape[0])
                inverse_parts.append(
                    torch.einsum(
                        "njk,mk->njm",
                        wigner_inv[start:end],
                        to_m,
                    )
                )
            inverse_prepared = torch.cat(inverse_parts, dim=0)
        else:
            inverse_prepared = torch.einsum(
                "njk,mk->njm", wigner_inv, to_m
            )
        return wigner_prepared, inverse_prepared

    ExecutionBackend.prepare_wigner = prepare_wigner
    ExecutionBackend._fxpu_wigner_prepare_chunked = True
    ExecutionBackend._fxpu_wigner_prepare_cfg = cfg
    return f"wignerPrepareChunked(chunk={cfg[0]},mode={cfg[1]})"
