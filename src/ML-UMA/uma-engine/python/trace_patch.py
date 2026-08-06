"""
Monkeypatches for TorchScript-friendly UMA inference during export.

MOLE.forward uses Python list intervals derived from mole_sizes; when traced,
interval endpoints become constants from the example graph and break other sizes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fairchem.core.models.uma.nn.mole import MOLE

_ORIGINAL_MOLE_FORWARD = MOLE.forward


def mole_forward_traceable(self, x: torch.Tensor) -> torch.Tensor:
    """Trace-friendly MOLE forward (single-system batches)."""
    with torch.autocast(device_type=self.weights.device.type, enabled=False):
        weights = torch.einsum(
            "eoi, be->boi",
            self.weights,
            self.global_mole_tensors.expert_mixing_coefficients,
        )

    mole_sizes = self.global_mole_tensors.mole_sizes
    if mole_sizes.numel() == 0:
        raise RuntimeError("mole_sizes must be set before MOLE forward")

    # Single bucket: apply one expert to the full activation (typical single-system path).
    if mole_sizes.numel() == 1:
        return F.linear(x, weights[0], self.bias)

    # Multi-bucket: tensor splits (no Python int from traced .tolist()).
    splits = mole_sizes.to(device=x.device, dtype=torch.long)
    chunks = torch.split(x, splits.tolist())
    outs = [F.linear(chunk, weights[i], self.bias) for i, chunk in enumerate(chunks)]
    return torch.cat(outs, dim=0)


def apply_trace_patches() -> None:
    MOLE.forward = mole_forward_traceable


def restore_trace_patches() -> None:
    MOLE.forward = _ORIGINAL_MOLE_FORWARD
