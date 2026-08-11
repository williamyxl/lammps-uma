"""Thread-local FairChem GP collectives via CUDA peer copies (no Ray / no c10d).

Installs replacements for ``fairchem.core.common.gp_utils`` so N worker threads
can run sharded UMA forwards concurrently; gather/reduce use ``tensor.to(device)``
peer copies (same semantics as ``uma::kokkos_peer``).

Important: FairChem Autograd Function classes call ``dist.all_reduce`` /
``dist.all_gather`` directly. Patching only the wrapper functions is not enough
if anything still invokes those classes — we replace Function.forward/backward too.
"""

from __future__ import annotations

import threading
from typing import Callable

import torch

_tls = threading.local()
_world_size = 1
_installed = False
_orig: dict[str, Callable] = {}
_orig_methods: dict[str, tuple] = {}


class _Barrier:
    def __init__(self, n: int) -> None:
        self._n = n
        self._c = threading.Condition()
        self._count = 0
        self._gen = 0

    def wait(self) -> None:
        with self._c:
            gen = self._gen
            self._count += 1
            if self._count == self._n:
                self._gen += 1
                self._count = 0
                self._c.notify_all()
            else:
                while gen == self._gen:
                    self._c.wait()


class _GatherSlot:
    def __init__(self, world: int) -> None:
        self.world = world
        self.barrier = _Barrier(world)
        self.bufs: list[torch.Tensor | None] = [None] * world
        self.lock = threading.Lock()

    def all_gather(self, rank: int, local: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with self.lock:
            self.bufs[rank] = local.contiguous()
        self.barrier.wait()
        srcs = list(self.bufs)
        self.barrier.wait()
        if rank == 0:
            with self.lock:
                self.bufs = [None] * self.world
        self.barrier.wait()
        out = []
        dev = local.device
        for t in srcs:
            assert t is not None
            out.append(t.to(dev, non_blocking=False).contiguous())
        return tuple(out)


class SharedGatherSlot:
    """Host-staged all_gather for multiprocessing (no c10d / no Ray).

    Tensors cross process boundaries via Manager-pickled numpy; each rank
    copies the result back to its CUDA device.

    Important: a single ``Barrier`` must not gate multiple phases of consecutive
    collectives — a fast rank's phase-1 wait can pair with a slow rank's phase-3
    wait and silently release with missing payloads. Use a generation-matched
    Condition instead (double-buffered slots).
    """

    def __init__(self, world: int, manager) -> None:
        self.world = world
        self.lock = manager.Lock()
        self.cond = manager.Condition(self.lock)
        self.buf_a = manager.list([None] * world)
        self.buf_b = manager.list([None] * world)
        self.gen = manager.Value("i", 0)
        self.nwrite = manager.Value("i", 0)
        self.nread = manager.Value("i", 0)

    def all_gather(self, rank: int, local: torch.Tensor) -> tuple[torch.Tensor, ...]:
        import numpy as np

        payload = local.detach().cpu().contiguous().numpy().copy()
        with self.cond:
            gen = self.gen.value
            slot = self.buf_a if (gen % 2 == 0) else self.buf_b
            slot[rank] = payload
            self.nwrite.value += 1
            if self.nwrite.value == self.world:
                self.cond.notify_all()
            else:
                while self.nwrite.value < self.world and self.gen.value == gen:
                    self.cond.wait(timeout=300)
                    if self.gen.value != gen:
                        break
            if self.gen.value != gen and self.nwrite.value < self.world:
                raise RuntimeError(
                    f"SharedGatherSlot: rank {rank} woke on gen advance without "
                    f"full write quorum (nwrite={self.nwrite.value})"
                )
            srcs = []
            for r in range(self.world):
                raw = slot[r]
                if raw is None:
                    raise RuntimeError(
                        f"SharedGatherSlot missing rank {r} payload at gen={gen}"
                    )
                srcs.append(np.array(raw, copy=True))
            self.nread.value += 1
            if self.nread.value == self.world:
                self.nwrite.value = 0
                self.nread.value = 0
                self.gen.value = gen + 1
                self.cond.notify_all()
            else:
                while self.gen.value == gen:
                    self.cond.wait(timeout=300)

        out = []
        for arr in srcs:
            t = torch.from_numpy(arr).to(
                device=local.device, dtype=local.dtype, non_blocking=False
            )
            out.append(t.contiguous())
        return tuple(out)


_gather_slot: _GatherSlot | SharedGatherSlot | None = None
_reduce_slot: _GatherSlot | SharedGatherSlot | None = None


def size_list_fn(size: int, parts: int) -> list[int]:
    return [size // parts + (1 if idx < size % parts else 0) for idx in range(parts)]


def pad_input(inp: torch.Tensor, padded_size: int) -> torch.Tensor:
    if inp.shape[0] == padded_size:
        return inp
    return torch.cat(
        [
            inp,
            torch.zeros(
                (padded_size - inp.shape[0], *inp.shape[1:]),
                device=inp.device,
                dtype=inp.dtype,
            ),
        ],
        dim=0,
    )


def configure(
    world_size: int,
    gather_slot: _GatherSlot | SharedGatherSlot | None = None,
    reduce_slot: _GatherSlot | SharedGatherSlot | None = None,
) -> None:
    global _world_size, _gather_slot, _reduce_slot
    if world_size < 1:
        raise ValueError(world_size)
    _world_size = world_size
    if gather_slot is None:
        _gather_slot = _GatherSlot(world_size)
        _reduce_slot = _GatherSlot(world_size)
    else:
        _gather_slot = gather_slot
        _reduce_slot = reduce_slot if reduce_slot is not None else gather_slot



def set_rank(rank: int) -> None:
    _tls.rank = rank


def get_rank() -> int:
    return int(getattr(_tls, "rank", 0))


def initialized() -> bool:
    return _world_size > 1 and _gather_slot is not None


def get_gp_world_size() -> int:
    return _world_size if initialized() else 1


def get_gp_rank() -> int:
    return get_rank() if initialized() else 0


def get_gp_group():
    """Sentinel so any leftover FairChem code does not touch c10d default group."""
    return "kokkos_peer"


def peer_all_reduce_sum(inp: torch.Tensor) -> torch.Tensor:
    assert _reduce_slot is not None
    rank = get_gp_rank()
    gathered = _reduce_slot.all_gather(rank, inp.contiguous())
    acc = gathered[0].clone()
    for t in gathered[1:]:
        acc = acc + t
    return acc


def peer_all_gather_padded(inp: torch.Tensor) -> tuple[torch.Tensor, ...]:
    assert _gather_slot is not None
    return _gather_slot.all_gather(get_gp_rank(), inp.contiguous())


class _ReduceFromMP(torch.autograd.Function):
    """FairChem ReduceFromModelParallelRegion: all_reduce fwd, identity bwd.

    Used for energy/force GP reductions where each rank already holds a correct
    partial and the backward must not double-count.
    """

    @staticmethod
    def forward(ctx, inp: torch.Tensor) -> torch.Tensor:
        return peer_all_reduce_sum(inp)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class _AllReduceWithGrad(torch.autograd.Function):
    """Peer stand-in for escn_md's ``all_reduce_with_grad``.

    Default: identity backward (safe with concurrent per-rank autograd.grad).

    Set ``UMA_ALLREDUCE_WITH_GRAD_BWD=1`` to all_reduce grads in backward
    (true ``nn.functional.all_reduce`` semantics). Pair with
    ``UMA_SKIP_FORCE_GP_REDUCE=1`` when testing whether each rank already holds
    full dE/dpos after that backward (force all_reduce would then over-count).
    """

    @staticmethod
    def forward(ctx, inp: torch.Tensor) -> torch.Tensor:
        return peer_all_reduce_sum(inp)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import os

        if os.environ.get("UMA_ALLREDUCE_WITH_GRAD_BWD", "0") == "1":
            return peer_all_reduce_sum(grad_output.contiguous())
        return grad_output


class _GatherSumGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, natoms: int) -> torch.Tensor:
        world = get_gp_world_size()
        rank = get_gp_rank()
        sizes = size_list_fn(natoms, world)
        pad = natoms // world + (1 if natoms % world else 0)
        padded = pad_input(inp, pad)
        tensor_list = peer_all_gather_padded(padded)
        ctx.rank = rank
        ctx.pad = pad
        ctx.sizes = sizes
        ctx.natoms = natoms
        parts = [
            t.narrow(0, 0, s) if t.shape[0] != s else t
            for t, s in zip(tensor_list, sizes)
        ]
        return torch.cat(parts, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # FairChem gloo path: all_reduce(cat shards) then take this rank's slice.
        # Padding is applied *inside* this Function, so return unpadded local
        # grad matching the Function input ``inp`` (not the padded temp).
        assert _gather_slot is not None
        gathered = _gather_slot.all_gather(ctx.rank, grad_output.contiguous())
        summed = gathered[0].clone()
        for t in gathered[1:]:
            summed = summed + t
        start = sum(ctx.sizes[: ctx.rank])
        local = summed.narrow(0, start, ctx.sizes[ctx.rank])
        return local, None


def reduce_from_model_parallel_region(inp: torch.Tensor) -> torch.Tensor:
    if not initialized():
        return inp
    return _ReduceFromMP.apply(inp)


def gather_from_model_parallel_region(inp: torch.Tensor, natoms: int) -> torch.Tensor:
    if not initialized():
        return inp
    return _GatherSumGrad.apply(inp, natoms)


def gather_from_model_parallel_region_sum_grad(
    inp: torch.Tensor, natoms: int
) -> torch.Tensor:
    if not initialized():
        return inp
    return _GatherSumGrad.apply(inp, natoms)


def scatter_to_model_parallel_region(inp: torch.Tensor) -> torch.Tensor:
    if not initialized():
        return inp
    sizes = size_list_fn(inp.shape[0], get_gp_world_size())
    return inp.split(sizes)[get_gp_rank()]


def _patch_autograd_functions(gp) -> None:
    """Replace FairChem Function forwards that call dist.*."""

    def _reduce_fwd(ctx, inp: torch.Tensor) -> torch.Tensor:
        return peer_all_reduce_sum(inp)

    def _reduce_bwd(ctx, grad_output: torch.Tensor):
        return grad_output

    def _gather_fwd(ctx, inp: torch.Tensor):
        ctx.rank = get_gp_rank()
        ctx.world = get_gp_world_size()
        return peer_all_gather_padded(inp)

    def _gather_bwd_keep(ctx, *grad_outputs):
        return grad_outputs[ctx.rank]

    def _gather_bwd_sum(ctx, *grad_outputs):
        # FairChem NCCL reduce_scatter / gloo all_reduce(cat):
        # output[r] = sum_k grad_outputs_on_rank_k[r]
        # Equivalent: all_reduce(cat(grad_outputs)) then take padded slice r.
        assert _gather_slot is not None
        full = torch.cat([g.contiguous() for g in grad_outputs], dim=0)
        summed = peer_all_reduce_sum(full)
        pad = grad_outputs[0].shape[0]
        return summed.narrow(0, pad * ctx.rank, pad)

    if "ReduceFromModelParallelRegion" not in _orig_methods:
        _orig_methods["ReduceFromModelParallelRegion"] = (
            gp.ReduceFromModelParallelRegion.forward,
            gp.ReduceFromModelParallelRegion.backward,
        )
        _orig_methods["GatherFromModelParallelRegionGradPadded"] = (
            gp.GatherFromModelParallelRegionGradPadded.forward,
            gp.GatherFromModelParallelRegionGradPadded.backward,
        )
        _orig_methods["GatherFromModelParallelRegionSumGradPadded"] = (
            gp.GatherFromModelParallelRegionSumGradPadded.forward,
            gp.GatherFromModelParallelRegionSumGradPadded.backward,
        )

    gp.ReduceFromModelParallelRegion.forward = staticmethod(_reduce_fwd)
    gp.ReduceFromModelParallelRegion.backward = staticmethod(_reduce_bwd)
    gp.GatherFromModelParallelRegionGradPadded.forward = staticmethod(_gather_fwd)
    gp.GatherFromModelParallelRegionGradPadded.backward = staticmethod(_gather_bwd_keep)
    gp.GatherFromModelParallelRegionSumGradPadded.forward = staticmethod(_gather_fwd)
    gp.GatherFromModelParallelRegionSumGradPadded.backward = staticmethod(_gather_bwd_sum)


_orig_dist_nn: dict[str, Callable] = {}


def _patch_dist_nn_collectives() -> None:
    """escn_md imports ``all_reduce`` from torch.distributed.nn.functional directly.

    Must preserve nn.functional semantics: all_reduce in forward *and* backward.
    Do **not** reuse ReduceFromModelParallelRegion (identity backward).
    """
    import torch.distributed.nn.functional as dist_nn

    if "all_reduce" not in _orig_dist_nn:
        _orig_dist_nn["all_reduce"] = dist_nn.all_reduce

    def _all_reduce_peer(tensor, op=None, group=None):  # noqa: ANN001
        if initialized() and (group is None or group == "kokkos_peer"):
            return _AllReduceWithGrad.apply(tensor)
        return _orig_dist_nn["all_reduce"](tensor, op=op, group=group)

    dist_nn.all_reduce = _all_reduce_peer

    # Also patch module-level binding in escn_md if already imported.
    try:
        import fairchem.core.models.uma.escn_md as escn_md

        escn_md.all_reduce_with_grad = _all_reduce_peer
    except Exception:
        pass


def install_patches(
    world_size: int,
    gather_slot: _GatherSlot | SharedGatherSlot | None = None,
    reduce_slot: _GatherSlot | SharedGatherSlot | None = None,
) -> None:
    """Monkeypatch fairchem.core.common.gp_utils for Kokkos-peer GP."""
    global _installed, _orig
    configure(world_size, gather_slot=gather_slot, reduce_slot=reduce_slot)
    if world_size <= 1:
        return
    import fairchem.core.common.gp_utils as gp

    if not _installed:
        for name in (
            "initialized",
            "get_gp_world_size",
            "get_gp_rank",
            "get_gp_group",
            "reduce_from_model_parallel_region",
            "gather_from_model_parallel_region",
            "gather_from_model_parallel_region_sum_grad",
            "scatter_to_model_parallel_region",
            "pad_input",
            "size_list_fn",
        ):
            if hasattr(gp, name):
                _orig[name] = getattr(gp, name)
        _installed = True

    gp.initialized = initialized
    gp.get_gp_world_size = get_gp_world_size
    gp.get_gp_rank = get_gp_rank
    gp.get_gp_group = get_gp_group
    gp.reduce_from_model_parallel_region = reduce_from_model_parallel_region
    gp.gather_from_model_parallel_region = gather_from_model_parallel_region
    gp.gather_from_model_parallel_region_sum_grad = (
        gather_from_model_parallel_region_sum_grad
    )
    gp.scatter_to_model_parallel_region = scatter_to_model_parallel_region
    gp.pad_input = pad_input
    gp.size_list_fn = size_list_fn
    gp._GRAPH_PARALLEL_GROUP = "kokkos_peer"
    gp._DATA_PARALLEL_GROUP = "kokkos_peer_dp"
    _patch_autograd_functions(gp)
    _patch_dist_nn_collectives()
    _maybe_skip_force_gp_reduce()


def _maybe_skip_force_gp_reduce() -> None:
    """If UMA_SKIP_FORCE_GP_REDUCE=1, do not all_reduce forces after autograd.

    Unit tests show gather sum_grad backward can already yield total-energy
    embedding grads; an extra force all_reduce then over-counts on some paths.
    Toggle to measure which regime UMA inference is in.

    Must patch both ``outputs`` and ``escn_md`` (the latter binds the symbols at
    import time).
    """
    import os

    if os.environ.get("UMA_SKIP_FORCE_GP_REDUCE", "0") != "1":
        return
    import fairchem.core.models.uma.outputs as outputs

    def compute_forces(energy_part, pos, training=True):
        (grad,) = torch.autograd.grad(energy_part.sum(), pos, create_graph=training)
        return torch.neg(grad)

    def compute_forces_and_stress(energy_part, pos, cell, batch, training=True):
        grads = torch.autograd.grad(
            [energy_part.sum()], [pos, cell], create_graph=training
        )
        num_systems = cell.shape[0]
        forces = torch.neg(grads[0])
        pos_virial_per_atom = grads[0].unsqueeze(2) * pos.unsqueeze(1)
        pos_virial, _ = outputs.reduce_node_to_system(
            pos_virial_per_atom, batch, num_systems
        )
        cell_virial = cell.mT @ grads[1]
        virial = (pos_virial + pos_virial.mT + cell_virial + cell_virial.mT) / 2
        volume = torch.det(cell).abs().unsqueeze(-1)
        stress = (virial / volume.view(-1, 1, 1)).view(-1, 9)
        return forces, stress

    outputs.compute_forces = compute_forces
    outputs.compute_forces_and_stress = compute_forces_and_stress
    try:
        import fairchem.core.models.uma.escn_md as escn_md

        escn_md.compute_forces = compute_forces
        escn_md.compute_forces_and_stress = compute_forces_and_stress
    except Exception:
        pass


def uninstall_patches() -> None:
    global _installed, _world_size, _gather_slot, _reduce_slot
    if not _installed:
        return
    import fairchem.core.common.gp_utils as gp

    for name, fn in _orig.items():
        setattr(gp, name, fn)
    for cls_name, (fwd, bwd) in _orig_methods.items():
        cls = getattr(gp, cls_name)
        cls.forward = fwd
        cls.backward = bwd
    if "all_reduce" in _orig_dist_nn:
        import torch.distributed.nn.functional as dist_nn

        dist_nn.all_reduce = _orig_dist_nn["all_reduce"]
        try:
            import fairchem.core.models.uma.escn_md as escn_md

            escn_md.all_reduce_with_grad = _orig_dist_nn["all_reduce"]
        except Exception:
            pass
    gp._GRAPH_PARALLEL_GROUP = None
    gp._DATA_PARALLEL_GROUP = None
    _installed = False
    _world_size = 1
    _gather_slot = None
    _reduce_slot = None
    _orig_methods.clear()
    _orig_dist_nn.clear()
