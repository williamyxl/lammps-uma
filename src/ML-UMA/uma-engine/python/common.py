"""
Shared utilities for libTorch export and parity checking.
"""

from __future__ import annotations

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


def atoms_to_atomic_data(
    atoms: Atoms,
    task_name: str,
    settings: InferenceSettings,
) -> AtomicData:
    """Build AtomicData the same way FAIRChemCalculator does for external graphs."""
    r_edges = settings.external_graph_gen
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
