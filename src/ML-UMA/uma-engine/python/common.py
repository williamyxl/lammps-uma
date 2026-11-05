"""
Shared utilities for libTorch export and parity checking.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from ase import Atoms

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.base import HydraModel
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    inference_settings_default,
)
from fairchem.core.units.mlip_unit.mlip_unit import Task

GRAPH_RADIUS = 6.0
GRAPH_MAX_NEIGHBORS = 300

# Active libTorch deployment target (OMAT forces+energy for LAMMPS).
DEFAULT_MODEL = "uma-s-1p2"
DEFAULT_TASK = "omat"
# Positions FP32; forces/energy accum FP64 handled in C++ engine / autograd path.
DEFAULT_DTYPE = "float32"
DEFAULT_DEVICE = "cuda"
FALLBACK_DEVICE = "cpu"


def resolve_device(requested: str | None = None) -> str:
    """Prefer CUDA; fall back to CPU when GPU is unavailable."""
    import logging

    device = requested or DEFAULT_DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        logging.getLogger(__name__).warning(
            "CUDA requested but unavailable; falling back to CPU"
        )
        return FALLBACK_DEVICE
    return device


def phase0_inference_settings() -> InferenceSettings:
    """Inference settings shared by oracle and export (Phase 0)."""
    settings = inference_settings_default()
    settings.external_graph_gen = True
    settings.execution_mode = "general"
    return settings


def export_inference_settings() -> InferenceSettings:
    """
    Settings for JIT/libTorch export.

    activation_checkpointing=False avoids torch.utils.checkpoint in the traced
    graph (not TorchScript-exportable).

    merge_mole=True fuses MOLE experts for fixed-composition turbo deployments
    only; default export keeps merge_mole=False for multi-composition parity.
    """
    settings = phase0_inference_settings()
    settings.activation_checkpointing = False
    return settings


def inference_settings_with_dtype(dtype: str) -> InferenceSettings:
    """Export/oracle settings with the requested compute precision."""
    settings = export_inference_settings()
    settings.base_precision_dtype = getattr(torch, dtype)
    return settings


def artifact_name_suffix(dtype: str) -> str:
    return "-f64" if dtype == "float64" else ""


def artifact_dir_name(
    model: str, task: str, dtype: str = DEFAULT_DTYPE
) -> str:
    return f"{model}-{task}{artifact_name_suffix(dtype)}"


def default_artifact_dir(artifact_root: Path | None = None) -> Path:
    root = artifact_root or Path("libtorch/artifacts")
    return root / artifact_dir_name(DEFAULT_MODEL, DEFAULT_TASK, DEFAULT_DTYPE)


def parse_dtype_from_metadata(metadata_path) -> torch.dtype:
    import json
    from pathlib import Path

    raw = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    dtype_name = raw.get("inference_settings", {}).get("base_precision_dtype", "float32")
    if isinstance(dtype_name, str):
        dtype_name = dtype_name.replace("torch.", "")
    return getattr(torch, dtype_name, torch.float32)


def _cell_list_edges(atoms: Atoms, radius: float):
    """O(N*neighbors) periodic neighbor list via a cell/bin list.

    Replaces AtomicData.from_ase's radius_graph (effectively O(N^2): 27s @ N=18,
    hangs @ N>=24) for LARGE trace exports. Returns (edge_index [2,E] int64
    row0=neighbor row1=center, cell_offsets [E,3] float) matching FairChem's
    convention: edge_distance_vec = pos[nbr] + offset@cell - pos[center].
    Orthorhombic cells only (the NaCl test); falls back to from_ase otherwise.
    """
    import itertools
    import numpy as np
    import torch
    cell = np.asarray(atoms.get_cell(), dtype=np.float64)
    if not (np.allclose(cell, np.diag(np.diag(cell))) and np.all(atoms.get_pbc())):
        return None  # non-orthorhombic or non-periodic -> caller uses from_ase
    L = np.diag(cell).astype(np.float64)
    pos = atoms.get_positions().astype(np.float64)
    pos -= np.floor(pos / L) * L  # wrap into cell
    n = len(atoms)
    ncell = np.maximum((L / radius).astype(np.int64), 1)  # bins per axis
    nx, ny, nz = int(ncell[0]), int(ncell[1]), int(ncell[2])
    binsz = L / ncell
    bi = np.minimum((pos / binsz).astype(np.int64), ncell - 1)  # [n,3] bin idx
    # Flat bin id per atom, and atoms grouped by bin (sorted).
    binid = (bi[:, 0] * ny + bi[:, 1]) * nz + bi[:, 2]          # [n]
    order = np.argsort(binid, kind="stable")
    binid_sorted = binid[order]
    # start offset of each occupied bin in the sorted order
    uniq, starts = np.unique(binid_sorted, return_index=True)
    ends = np.append(starts[1:], n)
    bin_start = {int(b): (int(s), int(e)) for b, s, e in zip(uniq, starts, ends)}
    r2 = radius * radius
    rows_nbr = []; rows_ctr = []; offs = []
    # For each of 27 neighbor-bin directions, vectorize over all atoms at once.
    for dcx, dcy, dcz in itertools.product((-1, 0, 1), repeat=3):
        # center-atom neighbor bin (may wrap): image = floor(nb/ncell)
        nbx = bi[:, 0] + dcx; wx = np.floor(nbx / nx).astype(np.int64); nbx -= wx * nx
        nby = bi[:, 1] + dcy; wy = np.floor(nby / ny).astype(np.int64); nby -= wy * ny
        nbz = bi[:, 2] + dcz; wz = np.floor(nbz / nz).astype(np.int64); nbz -= wz * nz
        tgt = (nbx * ny + nby) * nz + nbz                       # [n] target bin id
        # group centers by their target bin so we batch per occupied target bin
        c_order = np.argsort(tgt, kind="stable")
        tgt_s = tgt[c_order]
        u2, s2 = np.unique(tgt_s, return_index=True)
        e2 = np.append(s2[1:], n)
        for tb, cs, ce in zip(u2, s2, e2):
            se = bin_start.get(int(tb))
            if se is None:
                continue
            js = order[se[0]:se[1]]                              # neighbor atoms in target bin
            cs_atoms = c_order[cs:ce]                            # centers pointing here
            wrap = np.array([int(wx[cs_atoms[0]]), int(wy[cs_atoms[0]]),
                             int(wz[cs_atoms[0]])], dtype=np.int64)
            # NOTE wrap is per-center; but centers in this group share (dc) and
            # their own bin, so wrap differs only if they span the boundary. To be
            # exact, compute per (center,neighbor) below.
            # pairwise: centers cs_atoms x neighbors js
            pc = pos[cs_atoms]                                   # [C,3]
            pj = pos[js]                                         # [J,3]
            # per-center wrap:
            wcc = np.stack([wx[cs_atoms], wy[cs_atoms], wz[cs_atoms]], 1).astype(np.float64)  # [C,3]
            # d = pj + wrap*L - pc  (broadcast [C,1,3] vs [1,J,3])
            d = (pj[None, :, :] + (wcc[:, None, :] * L)) - pc[:, None, :]   # [C,J,3]
            r2m = np.einsum("cjk,cjk->cj", d, d)                # [C,J]
            mask = r2m <= r2
            ci_idx, jj_idx = np.nonzero(mask)
            if ci_idx.size == 0:
                continue
            ci_at = cs_atoms[ci_idx]; jj_at = js[jj_idx]
            # drop self (same atom, zero image)
            self_zero = (ci_at == jj_at) & (wx[ci_at] == 0) & (wy[ci_at] == 0) & (wz[ci_at] == 0)
            keep = ~self_zero
            ci_at = ci_at[keep]; jj_at = jj_at[keep]
            rows_ctr.append(ci_at); rows_nbr.append(jj_at)
            offs.append(np.stack([wx[ci_at], wy[ci_at], wz[ci_at]], 1))
    if not rows_nbr:
        return (torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3)))
    nbr = np.concatenate(rows_nbr); ctr = np.concatenate(rows_ctr)
    off = np.concatenate(offs).astype(np.float64)
    ei = torch.from_numpy(np.stack([nbr, ctr]).astype(np.int64))
    co = torch.from_numpy(off)
    return ei, co


def atoms_to_atomic_data(
    atoms: Atoms,
    task_name: str,
    settings: InferenceSettings,
) -> AtomicData:
    """Build AtomicData the same way FAIRChemCalculator does for external graphs.

    For large systems set UMA_EXPORT_CELL_LIST=1 to use an O(N) cell-list edge
    build (from_ase's radius_graph is O(N^2) and hangs at N>=24). The cell-list
    path builds the AtomicData with r_edges=False then injects edge_index +
    cell_offsets, matching FairChem's convention exactly.
    """
    r_edges = settings.external_graph_gen
    use_cell_list = (r_edges and
                     os.environ.get("UMA_EXPORT_CELL_LIST", "0").strip()
                     in ("1", "true", "yes"))
    if use_cell_list:
        edges = _cell_list_edges(atoms, GRAPH_RADIUS)
        if edges is not None:
            data = AtomicData.from_ase(
                atoms, task_name=task_name, r_edges=False, radius=GRAPH_RADIUS,
                max_neigh=None, r_data_keys=["spin", "charge"],
                target_dtype=settings.base_precision_dtype)
            ei, co = edges
            data.edge_index = ei
            data.cell_offsets = co.to(settings.base_precision_dtype)
            # some AtomicData variants expect neighbors count / nedges
            try:
                data.neighbors = torch.tensor([ei.shape[1]], dtype=torch.long)
            except Exception:
                pass
            return data
    max_neigh = GRAPH_MAX_NEIGHBORS if r_edges else None
    return AtomicData.from_ase(
        atoms,
        task_name=task_name,
        r_edges=r_edges,
        radius=GRAPH_RADIUS,
        max_neigh=max_neigh,
        r_data_keys=["spin", "charge"],
        target_dtype=settings.base_precision_dtype,
    )


def find_energy_task(model: HydraModel, dataset_name: str) -> Task:
    for task in model.dataset_to_tasks[dataset_name]:
        if task.property == "energy":
            return task
    raise KeyError(f"No energy task found for dataset '{dataset_name}'")


def extract_normed_energy(raw_output: dict, task: Task) -> torch.Tensor:
    value = raw_output[task.name]
    if isinstance(value, dict):
        return value[task.property]
    return value


def ensure_batch_full_fields(data: AtomicData) -> None:
    """Element reference undo expects fields set by the UMA backbone forward pass."""
    if "batch_full" not in data:
        data["batch_full"] = data.batch
    if "atomic_numbers_full" not in data:
        data["atomic_numbers_full"] = data.atomic_numbers


def postprocess_energy(
    model: HydraModel,
    data: AtomicData,
    normed_energy: torch.Tensor,
    dataset_name: str,
    undo_refs: bool = True,
) -> torch.Tensor:
    task = find_energy_task(model, dataset_name)
    device = normed_energy.device
    mean = task.normalizer.mean.to(device)
    rmsd = task.normalizer.rmsd.to(device)
    energy = normed_energy * rmsd + mean
    if undo_refs and task.element_references is not None:
        ensure_batch_full_fields(data)
        elem_refs = task.element_references.element_references.to(device)
        refs_sum = torch.zeros(energy.shape, dtype=energy.dtype, device=device).scatter_reduce(
            0,
            data["batch_full"].long(),
            elem_refs[data["atomic_numbers_full"].long()],
            reduce="sum",
        )
        energy = energy + refs_sum
    return energy


def energy_parity_tolerance(reference: float, dtype: torch.dtype) -> float:
    if dtype == torch.float64:
        return 1e-10
    return max(1e-6, 1e-6 * abs(reference))


def energies_match(
    predicted: float,
    reference: float,
    dtype: torch.dtype = torch.float32,
) -> bool:
    return abs(predicted - reference) <= energy_parity_tolerance(reference, dtype)


def _disable_regress_config(cfg) -> None:
    if cfg is None:
        return
    cfg.forces = False
    cfg.stress = False
    cfg.hessian = False


def disable_derivative_regression(model: HydraModel) -> None:
    """Energy-only export: skip autograd force/stress paths."""
    if hasattr(model.backbone, "regress_config"):
        _disable_regress_config(model.backbone.regress_config)
    for head in model.output_heads.values():
        if hasattr(head, "regress_config"):
            _disable_regress_config(head.regress_config)
        if hasattr(head, "head") and hasattr(head.head, "regress_config"):
            _disable_regress_config(head.head.regress_config)
