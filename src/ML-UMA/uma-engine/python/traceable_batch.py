"""
Trace/export-friendly batch container mimicking AtomicData without validation.

AtomicData.validate() and num_graphs (Python int from batch.max()) block
torch.export and torch.jit.trace. This class provides the subset of the
AtomicData interface used by UMA inference with single-system batches.
"""

from __future__ import annotations

from typing import Iterator

import torch


class TraceableBatch:
    """Minimal AtomicData stand-in for TorchScript export (single graph only)."""

    def __init__(
        self,
        pos: torch.Tensor,
        atomic_numbers: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        natoms: torch.Tensor,
        edge_index: torch.Tensor,
        cell_offsets: torch.Tensor,
        nedges: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
        fixed: torch.Tensor,
        tags: torch.Tensor,
        batch: torch.Tensor,
        dataset: list[str],
        sid: list[str] | None = None,
    ) -> None:
        self.pos = pos
        self.atomic_numbers = atomic_numbers
        self.cell = cell
        self.pbc = pbc
        self.natoms = natoms
        self.edge_index = edge_index
        self.cell_offsets = cell_offsets
        self.nedges = nedges
        self.charge = charge
        self.spin = spin
        self.fixed = fixed
        self.tags = tags
        self.batch = batch
        self.dataset = dataset
        self.sid = sid if sid is not None else [""]
        self.__keys__ = {
            "pos",
            "atomic_numbers",
            "cell",
            "pbc",
            "natoms",
            "edge_index",
            "cell_offsets",
            "nedges",
            "charge",
            "spin",
            "fixed",
            "tags",
            "batch",
            "dataset",
            "sid",
        }

    @property
    def num_nodes(self) -> int:
        return self.pos.size(0)

    @property
    def num_graphs(self) -> int:
        return 1

    def get(self, key: str, default=None):
        if key in self:
            return self[key]
        return default

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        setattr(self, key, value)
        self.__keys__.add(key)

    def __contains__(self, key: str) -> bool:
        return key in self.__keys__

    def keys(self) -> set[str]:
        return set(self.__keys__)

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor | list[str]]]:
        for key in sorted(self.__keys__):
            yield key, self[key]

    def values(self) -> list:
        return [item for _, item in self]

    def clone(self) -> TraceableBatch:
        return TraceableBatch(
            pos=self.pos.clone(),
            atomic_numbers=self.atomic_numbers.clone(),
            cell=self.cell.clone(),
            pbc=self.pbc.clone(),
            natoms=self.natoms.clone(),
            edge_index=self.edge_index.clone(),
            cell_offsets=self.cell_offsets.clone(),
            nedges=self.nedges.clone(),
            charge=self.charge.clone(),
            spin=self.spin.clone(),
            fixed=self.fixed.clone(),
            tags=self.tags.clone(),
            batch=self.batch.clone(),
            dataset=list(self.dataset),
            sid=list(self.sid),
        )

    def to(self, device, **kwargs) -> TraceableBatch:
        out = self.clone()
        for key in out.__keys__:
            val = out[key]
            if torch.is_tensor(val):
                out[key] = val.to(device, **kwargs)
        return out
