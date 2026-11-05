"""
Sidecar metadata schema for exported libTorch UMA energy artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from fairchem.core.models.base import HydraModel
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

from common import (
    GRAPH_MAX_NEIGHBORS,
    GRAPH_RADIUS,
    find_energy_task,
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(obj, DictConfig):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _to_plain_json(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=_json_default))


@dataclass
class TaskMetadata:
    name: str
    dataset: str
    normalizer_mean: float
    normalizer_rmsd: float
    element_references: list[float] | None = None


@dataclass
class ExportMetadata:
    model_name: str
    task_name: str
    export_format: str
    inference_settings: dict[str, Any]
    cutoff: float
    max_neighbors: int
    graph_mode: str
    energy_task: TaskMetadata
    atom_refs: dict[str, dict[str, float]] | None = None
    checkpoint_path: str | None = None
    checkpoint_revision: str | None = None
    export_notes: list[str] = field(default_factory=list)
    # P4'.1 schema version + provenance. The C++ loader requires
    # metadata_version >= 2 (else UMA_ALLOW_LEGACY_METADATA=1). Provenance is
    # filled by fill_provenance() at export time.
    metadata_version: int = 2
    fairchem_version: str | None = None
    torch_version: str | None = None
    exporter_git_sha: str | None = None
    checkpoint_sha256: str | None = None

    def fill_provenance(self, checkpoint_file: str | None = None) -> None:
        """Populate fairchem/torch versions, exporter git sha, and (optionally) the
        checkpoint sha256. Best-effort: never fails the export on a lookup error."""
        import hashlib
        import subprocess
        try:
            import torch as _t
            self.torch_version = str(_t.__version__)
        except Exception:
            pass
        try:
            import fairchem.core as _fc
            self.fairchem_version = getattr(_fc, "__version__", None)
            if self.fairchem_version is None:
                from importlib.metadata import version as _v
                self.fairchem_version = _v("fairchem-core")
        except Exception:
            pass
        try:
            here = Path(__file__).resolve().parent
            self.exporter_git_sha = subprocess.check_output(
                ["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            pass
        if checkpoint_file:
            try:
                h = hashlib.sha256()
                with open(checkpoint_file, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                self.checkpoint_sha256 = h.hexdigest()
            except Exception:
                pass

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=_json_default),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> ExportMetadata:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["energy_task"] = TaskMetadata(**raw["energy_task"])
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)


def _sanitize_settings(settings: InferenceSettings) -> dict[str, Any]:
    return _to_plain_json(settings.to_omegaconf())


def _element_references_list(task) -> list[float] | None:
    if task.element_references is None:
        return None
    refs = task.element_references.element_references.detach().cpu().tolist()
    if isinstance(refs[0], list):
        return refs[0]
    return refs


def build_export_metadata(
    model: HydraModel,
    model_name: str,
    task_name: str,
    settings: InferenceSettings,
    export_format: str,
    checkpoint_path: str | None = None,
    checkpoint_revision: str | None = None,
    atom_refs: dict | None = None,
    export_notes: list[str] | None = None,
) -> ExportMetadata:
    energy_task = find_energy_task(model, task_name)
    backbone = model.backbone
    cutoff = getattr(backbone, "cutoff", GRAPH_RADIUS)
    max_neighbors = getattr(backbone, "max_neighbors", GRAPH_MAX_NEIGHBORS)

    return ExportMetadata(
        model_name=model_name,
        task_name=task_name,
        export_format=export_format,
        inference_settings=_sanitize_settings(settings),
        cutoff=float(cutoff),
        max_neighbors=int(max_neighbors),
        graph_mode="external" if settings.external_graph_gen else "internal",
        energy_task=TaskMetadata(
            name=energy_task.name,
            dataset=task_name,
            normalizer_mean=float(energy_task.normalizer.mean.item()),
            normalizer_rmsd=float(energy_task.normalizer.rmsd.item()),
            element_references=_element_references_list(energy_task),
        ),
        atom_refs=_to_plain_json(atom_refs) if atom_refs is not None else None,
        checkpoint_path=checkpoint_path,
        checkpoint_revision=checkpoint_revision,
        export_notes=export_notes or [],
    )


def denorm_from_metadata(normed: torch.Tensor, metadata: ExportMetadata) -> torch.Tensor:
    task = metadata.energy_task
    return normed * task.normalizer_rmsd + task.normalizer_mean


def undo_refs_from_metadata(
    metadata: ExportMetadata,
    atomic_numbers: torch.Tensor,
    batch: torch.Tensor,
    energy: torch.Tensor,
) -> torch.Tensor:
    refs = metadata.energy_task.element_references
    if refs is None:
        return energy
    elem_refs = torch.tensor(refs, dtype=energy.dtype, device=energy.device)
    per_atom = elem_refs[atomic_numbers.long()]
    total = torch.zeros_like(energy).scatter_add_(0, batch.long(), per_atom)
    return energy + total
