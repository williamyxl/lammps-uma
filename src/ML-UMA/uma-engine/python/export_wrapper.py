"""
Differentiable energy export wrapper for UMA (OMAT).

Forces are NOT computed inside the traced module. The C++ engine runs
torch::autograd::grad on the energy w.r.t. positions (FairChem's
compute_forces pattern: forces = -dE/dpos).

Unlike the energy-only prior-art wrapper, forward() does NOT wrap the
inner model in torch.no_grad(), so gradients can flow at inference time.
Internal force/stress heads remain disabled via disable_derivative_regression.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.base import HydraModel

from common import disable_derivative_regression, find_energy_task
from traceable_batch import TraceableBatch


class EnergyExportWrapper(nn.Module):
    """TorchScript-friendly energy-only module (differentiable w.r.t. pos)."""

    def __init__(self, model: HydraModel, dataset_name: str, traceable: bool = False):
        super().__init__()
        self.inner = model
        self.dataset_name = dataset_name
        self.energy_task = find_energy_task(model, dataset_name)
        self.traceable = traceable
        disable_derivative_regression(self.inner)

    def _build_batch(
        self,
        pos: torch.Tensor,
        atomic_numbers: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        edge_index: torch.Tensor,
        cell_offsets: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
    ) -> AtomicData | TraceableBatch:
        device = pos.device
        # Capture-safe: no _shape_as_tensor + CPU→GPU .to() (illegal under CUDA
        # graph capture). empty+fill_ stays on-device; size() is host int at
        # trace time and is fine for fixed-shape EDGE_PAD graphs (W17).
        natoms = torch.empty(1, dtype=torch.long, device=device)
        natoms.fill_(pos.size(0))
        nedges = torch.empty(1, dtype=torch.long, device=device)
        nedges.fill_(edge_index.size(1))
        fixed = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
        tags = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)

        if cell.dim() == 2:
            cell = cell.unsqueeze(0)
        if pbc.dim() == 1:
            pbc = pbc.unsqueeze(0)
        if charge.dim() == 0:
            charge = charge.unsqueeze(0)
        if spin.dim() == 0:
            spin = spin.unsqueeze(0)

        kwargs = dict(
            pos=pos,
            atomic_numbers=atomic_numbers.long().to(device),
            cell=cell.to(device=device, dtype=pos.dtype),
            pbc=pbc.bool().to(device),
            natoms=natoms,
            edge_index=edge_index.long().to(device),
            cell_offsets=cell_offsets.to(device=device, dtype=pos.dtype),
            nedges=nedges,
            charge=charge.long().to(device),
            spin=spin.long().to(device),
            fixed=fixed,
            tags=tags,
            batch=batch,
            dataset=[self.dataset_name],
        )
        if self.traceable:
            return TraceableBatch(**kwargs)
        return AtomicData(**kwargs)

    def forward(
        self,
        pos: torch.Tensor,
        atomic_numbers: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        edge_index: torch.Tensor,
        cell_offsets: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized energy logit (requires_grad-friendly)."""
        device = pos.device
        pos = pos.to(device)
        atomic_numbers = atomic_numbers.to(device)
        cell = cell.to(device)
        pbc = pbc.to(device)
        edge_index = edge_index.to(device)
        cell_offsets = cell_offsets.to(device)
        charge = charge.to(device)
        spin = spin.to(device)
        data = self._build_batch(
            pos, atomic_numbers, cell, pbc, edge_index, cell_offsets, charge, spin
        )
        # Intentionally no torch.no_grad — C++ autograd needs a live graph.
        output = self.inner(data)
        value = output[self.energy_task.name]
        if isinstance(value, dict):
            return value[self.energy_task.property]
        return value

    def example_inputs_from_data(self, data: AtomicData) -> tuple[torch.Tensor, ...]:
        device = data.pos.device
        charge = data.charge.to(device)
        spin = data.spin.to(device)
        if charge.numel() == 1 and charge.dim() == 1:
            charge = charge.squeeze(0)
        if spin.numel() == 1 and spin.dim() == 1:
            spin = spin.squeeze(0)
        return (
            data.pos,
            data.atomic_numbers.to(device),
            data.cell.squeeze(0).to(device),
            data.pbc.squeeze(0).to(device),
            data.edge_index.to(device),
            data.cell_offsets.to(device),
            charge,
            spin,
        )


def clone_prepared_model(model: HydraModel) -> HydraModel:
    cloned = copy.deepcopy(model)
    cloned.eval()
    disable_derivative_regression(cloned)
    return cloned


def make_traced_export_wrapper(model: HydraModel, dataset_name: str) -> EnergyExportWrapper:
    return EnergyExportWrapper(clone_prepared_model(model), dataset_name, traceable=True)


class NodeEnergyExportWrapper(EnergyExportWrapper):
    """M1 (multi-node DD): return per-atom energies alongside the total.

    Spatial decomposition needs ``E_local = sum_{i in local} e_i``. FairChem
    computes per-node energies internally (``compute_energy`` in
    ``escn_md.py``: ``node_energy = energy_block(scalar_embedding)``) but
    ``reduce_node_to_system`` collapses them before the head returns, and the
    energy-only wrapper exposes only that scalar.

    Rather than fork the model, capture the energy block's output with a
    forward hook. The hook sees exactly the tensor that is about to be reduced,
    so ``sum(node_energy) == total`` holds by construction and stays inside the
    autograd graph -- required, because forces come from
    ``grad(E, pos)`` and a detached copy would silently zero them.

    Post-processing is per-atom linear (``denorm_energy`` is ``E*rmsd + mean``
    with ``mean = 0``; ``undo_element_references`` is a per-atom scatter_add),
    so the identity survives it. See plan sec 4.2.
    """

    def __init__(self, model: HydraModel, dataset_name: str, traceable: bool = False):
        super().__init__(model, dataset_name, traceable)
        self._node_energy: torch.Tensor | None = None
        self._hook_handle = None
        self._register_energy_hook()

    def _find_energy_block(self) -> nn.Module | None:
        """Locate the head's energy_block, which maps [N, C] -> [N, 1]."""
        for module in self.inner.modules():
            blk = getattr(module, "energy_block", None)
            if blk is not None and isinstance(blk, nn.Module):
                return blk
        return None

    def _register_energy_hook(self) -> None:
        blk = self._find_energy_block()
        if blk is None:
            raise RuntimeError(
                "NodeEnergyExportWrapper: no energy_block found; cannot expose "
                "per-atom energies (model layout changed?)")

        def _capture(_module, _inputs, output):
            # Keep the graph: forces are grad(E, pos).
            self._node_energy = output.view(-1)

        self._hook_handle = blk.register_forward_hook(_capture)

    def forward(
        self,
        pos: torch.Tensor,
        atomic_numbers: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        edge_index: torch.Tensor,
        cell_offsets: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(node_energy[N], total_energy)``."""
        self._node_energy = None
        total = super().forward(
            pos, atomic_numbers, cell, pbc, edge_index, cell_offsets, charge, spin
        )
        node_e = self._node_energy
        if node_e is None:
            raise RuntimeError(
                "NodeEnergyExportWrapper: energy_block hook did not fire")
        return node_e, total


def make_node_energy_export_wrapper(
    model: HydraModel, dataset_name: str
) -> NodeEnergyExportWrapper:
    """M1 factory: DD export wrapper returning (node_energy, total_energy)."""
    return NodeEnergyExportWrapper(
        clone_prepared_model(model), dataset_name, traceable=True
    )
