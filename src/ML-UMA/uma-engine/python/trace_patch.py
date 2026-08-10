"""
Monkeypatches for TorchScript-friendly UMA inference during export.

1) MOLE.forward — avoid Python list intervals from mole_sizes (size constants).
2) Wigner edge-frame roll (gamma) — replace torch.rand with zeros so the
   traced graph has no aten::rand (CUDA-graph capture safe).

FairChem notes the same for their CUDA-graph Wigner helper:
  rotation_cuda_graph.py: `# gamma = torch.zeros_like(alpha)`
Random gamma is for SO(2) training augmentation; inference MD can fix gamma=0.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fairchem.core.models.uma.nn.mole import MOLE

_ORIGINAL_MOLE_FORWARD = MOLE.forward

# Bound originals for restore (imported names on escn_md must be rebound too).
_ORIG_AXIS_ANGLE = None
_ORIG_EULER = None
_ORIG_EULER_CG = None


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

    if mole_sizes.numel() == 1:
        return F.linear(x, weights[0], self.bias)

    splits = mole_sizes.to(device=x.device, dtype=torch.long)
    chunks = torch.split(x, splits.tolist())
    outs = [F.linear(chunk, weights[i], self.bias) for i, chunk in enumerate(chunks)]
    return torch.cat(outs, dim=0)


def _axis_angle_wigner_fixed_gamma(
    edge_distance_vec: torch.Tensor,
    lmax: int,
    gamma: torch.Tensor | None = None,
    coeffs: Any = None,
    U_blocks: Any = None,
    custom_kernels: Any = None,
):
    """Wrap FairChem hybrid Wigner: default gamma=0 instead of torch.rand."""
    assert _ORIG_AXIS_ANGLE is not None
    if gamma is None:
        if edge_distance_vec.dim() == 1:
            n = 1
            device = edge_distance_vec.device
            dtype = edge_distance_vec.dtype
        else:
            n = edge_distance_vec.shape[0]
            device = edge_distance_vec.device
            dtype = edge_distance_vec.dtype
        gamma = torch.zeros(n, dtype=dtype, device=device)
    return _ORIG_AXIS_ANGLE(
        edge_distance_vec,
        lmax,
        gamma=gamma,
        coeffs=coeffs,
        U_blocks=U_blocks,
        custom_kernels=custom_kernels,
    )


def _init_edge_rot_euler_angles_fixed(edge_distance_vec: torch.Tensor):
    """Euler-path twin of FairChem CUDA-graph comment: zeros instead of rand_like."""
    # Mirror fairchem.core.models.uma.common.rotation.init_edge_rot_euler_angles
    # but gamma = zeros (capture-safe).
    from fairchem.core.models.uma.common.rotation import Safeacos, Safeatan2

    xyz = torch.nn.functional.normalize(edge_distance_vec).clamp(-1.0, 1.0)
    x, y, z = torch.split(xyz, 1, dim=1)
    beta = Safeacos.apply(y.squeeze(-1))
    alpha = Safeatan2.apply(x.squeeze(-1), z.squeeze(-1))
    gamma = torch.zeros_like(alpha)
    return -gamma, -beta, -alpha


def _init_edge_rot_euler_angles_wigner_cuda_graph_fixed(edge_distance_vec: torch.Tensor):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))
    xyz = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))
    mask = xyz[:, 1].abs().isclose(xyz.new_ones(1))
    beta = torch.acos(xyz[:, 1])
    alpha = torch.atan2(xyz[:, 0], xyz[:, 2])
    gamma = torch.zeros_like(alpha)  # was torch.rand_like
    return mask, -gamma, -beta, -alpha


def apply_trace_patches() -> None:
    global _ORIG_AXIS_ANGLE, _ORIG_EULER, _ORIG_EULER_CG

    MOLE.forward = mole_forward_traceable

    import fairchem.core.models.uma.common.quaternion.wigner_d_hybrid as wigner_mod
    import fairchem.core.models.uma.common.rotation as rotation_mod
    import fairchem.core.models.uma.escn_md as escn_md

    if _ORIG_AXIS_ANGLE is None:
        _ORIG_AXIS_ANGLE = wigner_mod.axis_angle_wigner_hybrid
    if _ORIG_EULER is None:
        _ORIG_EULER = rotation_mod.init_edge_rot_euler_angles
    try:
        import fairchem.core.models.uma.common.rotation_cuda_graph as rot_cg

        if _ORIG_EULER_CG is None:
            _ORIG_EULER_CG = rot_cg.init_edge_rot_euler_angles_wigner_cuda_graph
        rot_cg.init_edge_rot_euler_angles_wigner_cuda_graph = (
            _init_edge_rot_euler_angles_wigner_cuda_graph_fixed
        )
    except Exception:
        rot_cg = None  # optional

    wigner_mod.axis_angle_wigner_hybrid = _axis_angle_wigner_fixed_gamma
    escn_md.axis_angle_wigner_hybrid = _axis_angle_wigner_fixed_gamma

    rotation_mod.init_edge_rot_euler_angles = _init_edge_rot_euler_angles_fixed
    escn_md.init_edge_rot_euler_angles = _init_edge_rot_euler_angles_fixed


def restore_trace_patches() -> None:
    global _ORIG_AXIS_ANGLE, _ORIG_EULER, _ORIG_EULER_CG

    MOLE.forward = _ORIGINAL_MOLE_FORWARD

    if _ORIG_AXIS_ANGLE is not None:
        import fairchem.core.models.uma.common.quaternion.wigner_d_hybrid as wigner_mod
        import fairchem.core.models.uma.escn_md as escn_md

        wigner_mod.axis_angle_wigner_hybrid = _ORIG_AXIS_ANGLE
        escn_md.axis_angle_wigner_hybrid = _ORIG_AXIS_ANGLE

    if _ORIG_EULER is not None:
        import fairchem.core.models.uma.common.rotation as rotation_mod
        import fairchem.core.models.uma.escn_md as escn_md

        rotation_mod.init_edge_rot_euler_angles = _ORIG_EULER
        escn_md.init_edge_rot_euler_angles = _ORIG_EULER

    if _ORIG_EULER_CG is not None:
        try:
            import fairchem.core.models.uma.common.rotation_cuda_graph as rot_cg

            rot_cg.init_edge_rot_euler_angles_wigner_cuda_graph = _ORIG_EULER_CG
        except Exception:
            pass
