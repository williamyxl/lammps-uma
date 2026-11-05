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

# P5'.1: this exporter monkey-patches ~25 fairchem/torch globals and HAND-REIMPLEMENTS
# several fairchem forward bodies. An upstream rename fails loudly, but an upstream
# semantic reorder produces a plausible WRONG energy silently. The cheap 80% guard is
# to pin the fairchem version the exporter was written against and refuse to run on a
# different one. Escape hatch: UMA_ALLOW_FAIRCHEM_MISMATCH=1 (logged).
_REQUIRED_FAIRCHEM = "2.21.0"
_REQUIRED_TORCH = "2.13.0"  # +xpu build; compared on the leading version only


def _assert_fairchem_version() -> None:
    import os
    import warnings

    allow = os.environ.get("UMA_ALLOW_FAIRCHEM_MISMATCH", "0") == "1"
    try:
        import fairchem.core as _fc
        fc_ver = getattr(_fc, "__version__", None)
        if fc_ver is None:
            from importlib.metadata import version as _v
            fc_ver = _v("fairchem-core")
    except Exception as exc:  # noqa: BLE001
        fc_ver = f"<unknown: {exc}>"
    try:
        tor_ver = torch.__version__
    except Exception as exc:  # noqa: BLE001
        tor_ver = f"<unknown: {exc}>"

    fc_ok = isinstance(fc_ver, str) and fc_ver.split("+")[0] == _REQUIRED_FAIRCHEM
    tor_ok = isinstance(tor_ver, str) and tor_ver.split("+")[0] == _REQUIRED_TORCH
    if fc_ok and tor_ok:
        return
    msg = (
        "UMA exporter version mismatch: the trace patches + hand-reimplemented "
        "fairchem forwards were written against fairchem-core=="
        f"{_REQUIRED_FAIRCHEM}, torch=={_REQUIRED_TORCH}. Found fairchem-core="
        f"{fc_ver}, torch={tor_ver}. A silent semantic drift here yields a WRONG "
        "artifact. Pin the environment (see repo requirements.txt) or set "
        "UMA_ALLOW_FAIRCHEM_MISMATCH=1 to override at your own risk."
    )
    if allow:
        warnings.warn(msg + " [OVERRIDDEN by UMA_ALLOW_FAIRCHEM_MISMATCH=1]",
                      RuntimeWarning, stacklevel=2)
        return
    raise RuntimeError(msg)


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


# ---------------------------------------------------------------------------
# Shape-generalization patches (quaternion hybrid Wigner path).
#
# torch.jit.trace bakes the trace-time edge count into the quaternion Wigner
# tensors, so a module traced at N=n0 fails at any other N. graspa-mlip
# (scripts/xpu_uma_export_odac_traced.py, validated to 1e-13 eV on XPU)
# localized exactly two baked sites and made their leading dim symbolic:
#   1) axis_angle_wigner_hybrid: gamma leading dim tied to a live tensor,
#   2) wigner_d_from_quaternion_hybrid: D leading dim + D[:,0,0] symbolic.
#
# CRITICAL difference for FORCES: graspa uses a *random* gamma (rand_like);
# Wigner-D is rotation invariant so energy is unaffected, but a random roll
# per forward would make autograd forces (one draw) inconsistent with finite
# difference (separate draws at +/- eps). We therefore keep gamma
# DETERMINISTIC (zeros) while keeping its leading dim symbolic via
# zeros_like(...) on a live [N] slice. Deterministic gamma=0 is what
# _axis_angle_wigner_fixed_gamma / the CUDA-graph helper already use.
# ---------------------------------------------------------------------------
_SHAPE_PATCH_RESTORE: list = []


def _install_shape_patches() -> None:
    import fairchem.core.models.uma.common.quaternion.wigner_d_hybrid as _wdh

    _orig_axis = _wdh.axis_angle_wigner_hybrid

    def _axis_symbolic(edge_distance_vec, lmax, gamma=None, *,
                       coeffs=None, U_blocks=None, custom_kernels=None):
        if edge_distance_vec.dim() == 1:
            edge_distance_vec = edge_distance_vec.unsqueeze(0)
        edge_normalized = torch.nn.functional.normalize(edge_distance_vec, dim=-1)
        if gamma is None:
            # Deterministic gamma=0 with SYMBOLIC leading dim (tied to live tensor).
            gamma = torch.zeros_like(edge_normalized[:, 0])
        q_edge_to_y = _wdh.quaternion_edge_to_y_stable(edge_normalized)
        q_gamma = _wdh.quaternion_y_rotation(gamma)
        q_combined = _wdh.quaternion_multiply(q_gamma, q_edge_to_y)
        D = _wdh.wigner_d_from_quaternion_hybrid(
            q_combined, lmax, coeffs=coeffs, U_blocks=U_blocks,
            custom_kernels=custom_kernels,
        )
        D_inv = D.transpose(1, 2).contiguous()
        return D, D_inv

    _wdh.axis_angle_wigner_hybrid = _axis_symbolic
    _SHAPE_PATCH_RESTORE.append(
        lambda: setattr(_wdh, "axis_angle_wigner_hybrid", _orig_axis)
    )

    _orig_hybrid = _wdh.wigner_d_from_quaternion_hybrid

    def _hybrid_symbolic(q, lmax, coeffs=None, U_blocks=None, custom_kernels=None):
        size = (lmax + 1) ** 2
        row = torch.ones_like(q[:, :1]).unsqueeze(-1)          # [N,1,1] symbolic
        D = torch.zeros(1, size, size, dtype=q.dtype, device=q.device) * row
        D = D.clone()
        D[:, 0, 0] = torch.ones_like(q[:, 0])                  # symbolic RHS
        if lmax >= 1:
            D[:, 1:4, 1:4] = _wdh.quaternion_to_rotation_matrix(q)
        if lmax >= 2:
            D[:, 4:9, 4:9] = _wdh.quaternion_to_wigner_d_l2_einsum(
                q, custom_kernels.C_l2)
        if lmax >= 4:
            D_l3, D_l4 = _wdh.quaternion_to_wigner_d_l3l4_batched(
                q, custom_kernels.C_combined_l3l4, custom_kernels.monomials_l4)
            D[:, 9:16, 9:16] = D_l3
            D[:, 16:25, 16:25] = D_l4
        elif lmax >= 3:
            D[:, 9:16, 9:16] = _wdh.quaternion_to_wigner_d_matmul(
                q, 3, custom_kernels.C_l3, custom_kernels.monomials_l3)
        if lmax >= 5:
            # G18/S7 (audit rev 16): make the fallback LOUD. The symbolic/traceable
            # path is only implemented up to l=4 (UMA-s-1p2 uses lmax=4). For lmax>=5
            # we fall back to the ORIGINAL fairchem hybrid — correct numerically, but
            # NOT the chunked/shape-generic variant, so a >l4 model would silently
            # export without the trace-shape guarantees. Warn once so it is not a
            # silent behavior change for a future model.
            import warnings
            warnings.warn(
                f"trace_patch: lmax={lmax} > 4 exceeds the traceable symbolic Wigner "
                f"path (implemented to l=4); falling back to the original fairchem "
                f"hybrid. Export shape-genericity is not guaranteed for l>4.",
                RuntimeWarning, stacklevel=2)
            return _orig_hybrid(q, lmax, coeffs=coeffs, U_blocks=U_blocks,
                                custom_kernels=custom_kernels)
        return D

    _wdh.wigner_d_from_quaternion_hybrid = _hybrid_symbolic
    _SHAPE_PATCH_RESTORE.append(
        lambda: setattr(_wdh, "wigner_d_from_quaternion_hybrid", _orig_hybrid)
    )


def _restore_shape_patches() -> None:
    while _SHAPE_PATCH_RESTORE:
        _SHAPE_PATCH_RESTORE.pop()()


_CKPT_PASSTHROUGH_RESTORE: list = []


def _install_checkpoint_passthrough() -> None:
    """Make torch.utils.checkpoint.checkpoint a traceable passthrough.

    (h) Traceable internal activation checkpointing: escn's edge_wise.forward
    edge-chunk loop calls torch.utils.checkpoint.checkpoint(forward_chunk, ...)
    per edge chunk. That does NOT survive torch.jit.trace (_NoopSaveInputs). We
    replace it with a DIRECT call so the chunk loop still executes chunk-by-chunk
    and traces as a plain sequence of per-chunk ops. At runtime the C++
    CheckpointModuleFn runs the whole module forward under no_grad, so each
    chunk's SO2-conv intermediates are freed after the chunk (peak = one chunk,
    not all edges) and recomputed in backward. This reproduces eager AC's memory
    profile in the pure-traced path. Chunk COUNT bakes at trace N (N-specific
    shards), which is fine.
    """
    import torch.utils.checkpoint as _ckpt

    orig = _ckpt.checkpoint

    def _passthrough(function, *args, use_reentrant=None, context_fn=None,
                     determinism_check=None, debug=None, **kwargs):
        return function(*args, **kwargs)

    _ckpt.checkpoint = _passthrough
    _CKPT_PASSTHROUGH_RESTORE.append(lambda: setattr(_ckpt, "checkpoint", orig))
    # escn_md_block imported the symbol by module ref (torch.utils.checkpoint.
    # checkpoint), so patching the module attribute above suffices.


def _restore_checkpoint_passthrough() -> None:
    while _CKPT_PASSTHROUGH_RESTORE:
        _CKPT_PASSTHROUGH_RESTORE.pop()()


def apply_trace_patches(shape_generic: bool = False,
                        checkpoint_passthrough: bool = False) -> None:
    global _ORIG_AXIS_ANGLE, _ORIG_EULER, _ORIG_EULER_CG

    # P5'.1: refuse to patch a fairchem/torch the exporter was not written for.
    _assert_fairchem_version()

    MOLE.forward = mole_forward_traceable

    if checkpoint_passthrough:
        _install_checkpoint_passthrough()

    if shape_generic:
        # Symbolic-dim quaternion Wigner: replaces the fixed-gamma wrappers below.
        _install_shape_patches()
        import fairchem.core.models.uma.common.quaternion.wigner_d_hybrid as wigner_mod
        import fairchem.core.models.uma.escn_md as escn_md
        # escn_md imported axis_angle_wigner_hybrid by name; rebind to symbolic.
        escn_md.axis_angle_wigner_hybrid = wigner_mod.axis_angle_wigner_hybrid
        return

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

    _restore_checkpoint_passthrough()

    # Restore shape-generic patches (if any) and rebind escn_md name.
    if _SHAPE_PATCH_RESTORE:
        _restore_shape_patches()
        try:
            import fairchem.core.models.uma.common.quaternion.wigner_d_hybrid as wigner_mod
            import fairchem.core.models.uma.escn_md as escn_md
            escn_md.axis_angle_wigner_hybrid = wigner_mod.axis_angle_wigner_hybrid
        except Exception:
            pass

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
