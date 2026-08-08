"""LibTorch-compatible peer collective ops for UMA graph-parallel export.

Registered as ``torch.ops.uma_peer.*`` so ``torch.jit.trace`` records them into
``model_mp_w*_r*.pt``. C++ registers the same schemas and implements them with
``uma::kokkos_peer`` + a process-wide PeerContext (no MPI / c10d / Ray).

Export-time kernels are single-rank stand-ins (identity / local pad-cat) so
tracing can run on CPU or one GPU; runtime C++ replaces the kernels.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.library import Library, impl

_LIB_NAME = "uma_peer"
_lib: Library | None = None
_world = 1
_rank = 0


def set_export_rank(rank: int, world: int) -> None:
    global _rank, _world
    if world < 1 or rank < 0 or rank >= world:
        raise ValueError(f"invalid rank/world: {rank}/{world}")
    _rank = int(rank)
    _world = int(world)


def export_rank() -> int:
    return _rank


def export_world() -> int:
    return _world


def _ensure_lib() -> Library:
    global _lib
    if _lib is not None:
        return _lib
    _lib = Library(_LIB_NAME, "DEF")
    _lib.define("rank() -> int")
    _lib.define("world() -> int")
    _lib.define("all_gather_nodes(Tensor local, int n_atoms) -> Tensor")
    _lib.define("all_reduce_sum(Tensor local) -> Tensor")

    @impl(_lib, "rank", "CompositeExplicitAutograd")
    def _rank_impl() -> int:
        return int(_rank)

    @impl(_lib, "world", "CompositeExplicitAutograd")
    def _world_impl() -> int:
        return int(_world)

    @impl(_lib, "all_gather_nodes", "CompositeExplicitAutograd")
    def _gather_impl(local: torch.Tensor, n_atoms: int) -> torch.Tensor:
        # Export stand-in (single process): place this rank's shard into the
        # correct slice of a zero-filled [n_atoms, ...] buffer.
        world = int(_world)
        rank = int(_rank)
        sizes = [n_atoms // world + (1 if i < (n_atoms % world) else 0) for i in range(world)]
        pad = max(sizes) if sizes else 0
        x = local
        if x.size(0) < pad:
            z = torch.zeros(
                (pad - x.size(0),) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device
            )
            x = torch.cat([x, z], dim=0)
        full = torch.zeros((n_atoms,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
        start = int(sum(sizes[:rank]))
        nloc = int(sizes[rank])
        if nloc > 0:
            full[start : start + nloc] = x[:nloc]
        return full.contiguous()

    @impl(_lib, "all_reduce_sum", "CompositeExplicitAutograd")
    def _reduce_impl(local: torch.Tensor) -> torch.Tensor:
        return local.clone()

    return _lib


def install_export_ops() -> None:
    _ensure_lib()


def patch_fairchem_gp_utils(world: int, rank: int) -> Callable[[], None]:
    """Patch FairChem gp_utils so gather/reduce call uma_peer ops; return restore."""
    install_export_ops()
    set_export_rank(rank, world)

    from fairchem.core.common import gp_utils

    saved: dict[str, object] = {}

    def _keep(name: str) -> None:
        saved[name] = getattr(gp_utils, name)

    for name in (
        "initialized",
        "get_gp_rank",
        "get_gp_world_size",
        "get_gp_group",
        "gather_from_model_parallel_region",
        "gather_from_model_parallel_region_sum_grad",
        "reduce_from_model_parallel_region",
        "pad_input",
        "size_list_fn",
    ):
        if hasattr(gp_utils, name):
            _keep(name)

    def initialized() -> bool:
        return True

    def get_gp_rank() -> int:
        # Plain Python int — baked into per-rank TorchScript at trace time.
        # Do not call uma_peer.rank() here (custom ops cannot return Python int).
        return int(_rank)

    def get_gp_world_size() -> int:
        return int(_world)

    def get_gp_group():
        return None

    def pad_input(inp: torch.Tensor, padded_size: int) -> torch.Tensor:
        if inp.shape[0] == padded_size:
            return inp
        z = torch.zeros(
            (padded_size - inp.shape[0],) + tuple(inp.shape[1:]),
            device=inp.device,
            dtype=inp.dtype,
        )
        return torch.cat([inp, z], dim=0)

    def size_list_fn(size: int, parts: int) -> list[int]:
        return [size // parts + (1 if idx < size % parts else 0) for idx in range(parts)]

    def gather_from_model_parallel_region(inp: torch.Tensor, natoms: int) -> torch.Tensor:
        w = get_gp_world_size()
        pad = natoms // w + (1 if natoms % w else 0)
        return torch.ops.uma_peer.all_gather_nodes(pad_input(inp, pad), int(natoms))

    def gather_from_model_parallel_region_sum_grad(inp: torch.Tensor, natoms: int) -> torch.Tensor:
        return gather_from_model_parallel_region(inp, natoms)

    def reduce_from_model_parallel_region(inp: torch.Tensor) -> torch.Tensor:
        return torch.ops.uma_peer.all_reduce_sum(inp)

    def all_reduce_with_grad_peer(tensor, op=None, group=None):  # noqa: ANN001
        # Stand-in for torch.distributed.nn.functional.all_reduce (used by escn_md).
        return torch.ops.uma_peer.all_reduce_sum(tensor)

    gp_utils.initialized = initialized  # type: ignore[assignment]
    gp_utils.get_gp_rank = get_gp_rank  # type: ignore[assignment]
    gp_utils.get_gp_world_size = get_gp_world_size  # type: ignore[assignment]
    gp_utils.get_gp_group = get_gp_group  # type: ignore[assignment]
    gp_utils.pad_input = pad_input  # type: ignore[assignment]
    gp_utils.size_list_fn = size_list_fn  # type: ignore[assignment]
    gp_utils.gather_from_model_parallel_region = gather_from_model_parallel_region  # type: ignore[assignment]
    gp_utils.gather_from_model_parallel_region_sum_grad = (  # type: ignore[assignment]
        gather_from_model_parallel_region_sum_grad
    )
    gp_utils.reduce_from_model_parallel_region = reduce_from_model_parallel_region  # type: ignore[assignment]

    # escn_md binds: from torch.distributed.nn.functional import all_reduce as all_reduce_with_grad
    import torch.distributed.nn.functional as dist_nn

    saved["dist_nn_all_reduce"] = dist_nn.all_reduce
    dist_nn.all_reduce = all_reduce_with_grad_peer  # type: ignore[assignment]
    try:
        import fairchem.core.models.uma.escn_md as escn_md

        saved["escn_md_all_reduce_with_grad"] = escn_md.all_reduce_with_grad
        escn_md.all_reduce_with_grad = all_reduce_with_grad_peer
    except Exception:
        pass

    # Autograd Function classes call dist directly — replace forward bodies.
    # Prefer the wrapper functions above; also neutralize Function.forward so any
    # residual call sites do not touch c10d.
    if hasattr(gp_utils, "GatherFromModelParallelRegionSumGradPadded"):
        Cls = gp_utils.GatherFromModelParallelRegionSumGradPadded

        def _fwd(ctx, input: torch.Tensor):  # noqa: ANN001
            ctx.rank = get_gp_rank()
            w = get_gp_world_size()
            # n_atoms ≈ unpadded total; input is padded local.
            # Reconstruct n_atoms from padded_size * world - (pad - last_size) is hard;
            # use input.size(0) * world as upper bound then callers narrow.
            n_atoms_guess = int(input.shape[0] * w)
            full = torch.ops.uma_peer.all_gather_nodes(input, n_atoms_guess)
            pad = int(input.shape[0])
            parts = []
            for r in range(w):
                sl = full.narrow(0, r * pad, min(pad, max(0, full.size(0) - r * pad)))
                if sl.size(0) < pad:
                    sl = torch.cat(
                        [sl, torch.zeros((pad - sl.size(0),) + tuple(input.shape[1:]),
                                         dtype=input.dtype, device=input.device)],
                        dim=0,
                    )
                parts.append(sl.contiguous())
            return tuple(parts)

        saved["GatherSumGrad_forward"] = Cls.forward
        Cls.forward = staticmethod(_fwd)  # type: ignore[assignment]

    if hasattr(gp_utils, "GatherFromModelParallelRegionGradPadded"):
        Cls2 = gp_utils.GatherFromModelParallelRegionGradPadded
        saved["GatherGrad_forward"] = Cls2.forward
        Cls2.forward = staticmethod(_fwd)  # type: ignore[assignment]  # reuse

    if hasattr(gp_utils, "ReduceFromModelParallelRegion"):
        Cls3 = gp_utils.ReduceFromModelParallelRegion

        def _rfwd(ctx, input: torch.Tensor):  # noqa: ANN001
            return torch.ops.uma_peer.all_reduce_sum(input)

        saved["Reduce_forward"] = Cls3.forward
        Cls3.forward = staticmethod(_rfwd)  # type: ignore[assignment]

    def restore() -> None:
        for name, val in saved.items():
            if name.endswith("_forward") or name in (
                "dist_nn_all_reduce",
                "escn_md_all_reduce_with_grad",
            ):
                continue
            setattr(gp_utils, name, val)
        if "GatherSumGrad_forward" in saved:
            gp_utils.GatherFromModelParallelRegionSumGradPadded.forward = saved[  # type: ignore[assignment]
                "GatherSumGrad_forward"
            ]
        if "GatherGrad_forward" in saved:
            gp_utils.GatherFromModelParallelRegionGradPadded.forward = saved[  # type: ignore[assignment]
                "GatherGrad_forward"
            ]
        if "Reduce_forward" in saved:
            gp_utils.ReduceFromModelParallelRegion.forward = saved["Reduce_forward"]  # type: ignore[assignment]
        if "dist_nn_all_reduce" in saved:
            import torch.distributed.nn.functional as dist_nn

            dist_nn.all_reduce = saved["dist_nn_all_reduce"]  # type: ignore[assignment]
        if "escn_md_all_reduce_with_grad" in saved:
            import fairchem.core.models.uma.escn_md as escn_md

            escn_md.all_reduce_with_grad = saved["escn_md_all_reduce_with_grad"]

    return restore
