#!/usr/bin/env python3
"""
Export uma-s-1p2 / omat TorchScript artifact with differentiable energy.

Forces are produced at C++ runtime via torch::autograd::grad (see Predictor).

Usage:
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate uma312
  cd /home/xyan11/workdir/uma-lmp
  ENG=lammps/src/ML-UMA/uma-engine
  PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_omat.py \\
      --checkpoint /mnt/d/workdir/uma-cache/uma-s-1p2.pt \\
      --output $ENG/artifacts/uma-s-1p2-omat
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

from checkpoints import resolve_checkpoint
from common import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_MODEL,
    atoms_to_atomic_data,
    inference_settings_with_dtype,
    resolve_device,
)
from export_wrapper import (
    EnergyExportWrapper,
    clone_prepared_model,
    make_traced_export_wrapper,
)
from metadata import build_export_metadata, denorm_from_metadata, undo_refs_from_metadata
from model_loader import get_atom_refs, load_prepared_hydra_model
from trace_patch import apply_trace_patches, restore_trace_patches

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _nacl_perturbed(seed: int = 0, noise: float = 0.05) -> Atoms:
    """2x2x2 rocksalt NaCl with Gaussian position noise (matches ASE refs)."""
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    rng = np.random.default_rng(seed)
    atoms.positions = atoms.positions + rng.normal(0.0, noise, size=atoms.positions.shape)
    atoms.pbc = True
    return atoms


def _attempt_trace(
    trace_wrapper: EnergyExportWrapper,
    example_inputs: tuple[torch.Tensor, ...],
    output_dir: Path,
) -> tuple[str, torch.nn.Module | None, list[str]]:
    notes: list[str] = []
    try:
        trace_wrapper.eval()
        apply_trace_patches()
        try:
            # Warmup under no_grad (does not embed no_grad into the module).
            with torch.no_grad():
                trace_wrapper(*example_inputs)
            traced = torch.jit.trace(trace_wrapper, example_inputs, strict=False)
        finally:
            restore_trace_patches()
        path = output_dir / "model_traced.pt"
        traced.save(str(path))
        loaded = torch.jit.load(str(path), map_location=example_inputs[0].device)
        notes.append(f"torch.jit.trace succeeded -> {path}")
        return "torch.jit.trace", loaded, notes
    except Exception as exc:
        notes.append(f"torch.jit.trace failed: {exc}")
        return "none", None, notes


def _physical_energy(
    normed: torch.Tensor,
    metadata,
    atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    energy = denorm_from_metadata(normed, metadata)
    batch = torch.zeros(atomic_numbers.shape[0], dtype=torch.long, device=normed.device)
    return undo_refs_from_metadata(metadata, atomic_numbers, batch, energy)


def export_omat(
    model_name: str,
    output_dir: Path,
    device: str,
    checkpoint_path: str,
    dtype: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = inference_settings_with_dtype(dtype)

    sample = bulk("Fe", "bcc", a=2.87, cubic=True)
    sample_data = atoms_to_atomic_data(sample, "omat", settings)

    logger.info("Loading checkpoint: %s", checkpoint_path)
    model, _, _ = load_prepared_hydra_model(
        checkpoint_path, sample_data, settings=settings, device=device
    )
    wrapper = EnergyExportWrapper(clone_prepared_model(model), "omat", traceable=False)
    trace_wrapper = make_traced_export_wrapper(model, "omat")
    wrapper.eval().to(device)
    trace_wrapper.to(device)

    example_data = atoms_to_atomic_data(sample, "omat", settings).to(device)
    example_inputs = wrapper.example_inputs_from_data(example_data)

    export_format, loaded, notes = _attempt_trace(trace_wrapper, example_inputs, output_dir)

    metadata = build_export_metadata(
        model=model,
        model_name=model_name,
        task_name="omat",
        settings=settings,
        export_format=export_format,
        checkpoint_path=checkpoint_path,
        atom_refs=get_atom_refs(model_name),
        export_notes=notes
        + [
            f"forces via C++ torch::autograd::grad on differentiable energy module",
            f"positions {dtype}; energy {dtype}; forces float64",
        ],
    )
    # Extend metadata with forces contract (C++ loader ignores unknown keys).
    meta_dict = metadata.to_dict()
    meta_dict["outputs"] = ["energy", "forces"]
    meta_dict["forces_via"] = "autograd"
    meta_dict["position_dtype"] = dtype
    meta_dict["energy_dtype"] = dtype
    meta_dict["force_dtype"] = "float64"
    (output_dir / "metadata.json").write_text(
        json.dumps(meta_dict, indent=2, default=str), encoding="utf-8"
    )

    report: dict = {
        "model_name": model_name,
        "task_name": "omat",
        "dtype": dtype,
        "export_format": export_format,
        "export_notes": notes,
        "output_dir": str(output_dir),
    }

    if loaded is None:
        report["error"] = "export failed"
        (output_dir / "export_report.json").write_text(json.dumps(report, indent=2))
        return report

    # Parity vs in-process wrapper + autograd forces on NaCl.
    logger.info("NaCl energy/force parity (exported module + autograd)...")
    atoms = _nacl_perturbed()
    data = atoms_to_atomic_data(atoms, "omat", settings).to(device)
    inputs = wrapper.example_inputs_from_data(data)

    # Reference: differentiable wrapper
    pos = inputs[0].detach().requires_grad_(True)
    ref_args = (pos, *inputs[1:])
    ref_normed = wrapper(*ref_args)
    ref_energy = _physical_energy(ref_normed, metadata, inputs[1])
    (ref_grad,) = torch.autograd.grad(ref_energy.sum(), pos)
    ref_forces = (-ref_grad).detach().to(torch.float64).cpu().numpy()
    ref_e = float(ref_energy.detach().cpu())

    # Exported
    pos2 = inputs[0].detach().requires_grad_(True)
    exp_args = (pos2, *inputs[1:])
    exp_normed = loaded(*exp_args)
    exp_energy = _physical_energy(exp_normed, metadata, inputs[1])
    (exp_grad,) = torch.autograd.grad(exp_energy.sum(), pos2)
    exp_forces = (-exp_grad).detach().to(torch.float64).cpu().numpy()
    exp_e = float(exp_energy.detach().cpu())

    de = abs(exp_e - ref_e)
    df = float(np.max(np.abs(exp_forces - ref_forces)))
    report["nacl_parity"] = {
        "ref_energy": ref_e,
        "exported_energy": exp_e,
        "abs_energy_error": de,
        "max_force_error": df,
        "passed": bool(de < 1e-4 and df < 1e-3),
    }
    logger.info(
        "NaCl E_ref=%.6f E_exp=%.6f dE=%.3e dFmax=%.3e pass=%s",
        ref_e,
        exp_e,
        de,
        df,
        report["nacl_parity"]["passed"],
    )

    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export UMA omat artifact for uma-engine")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=["float32", "float64"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "uma-s-1p2-omat",
    )
    parser.add_argument(
        "--checkpoint",
        default="/mnt/d/workdir/uma-cache/uma-s-1p2.pt",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda"])
    args = parser.parse_args(argv)
    args.device = resolve_device(args.device)

    try:
        checkpoint = resolve_checkpoint(args.model, args.checkpoint)
        report = export_omat(
            model_name=args.model,
            output_dir=args.output,
            device=args.device,
            checkpoint_path=checkpoint,
            dtype=args.dtype,
        )
    except Exception:
        traceback.print_exc()
        return 1

    if report.get("export_format") == "none":
        return 1
    if report.get("nacl_parity") and not report["nacl_parity"]["passed"]:
        logger.error("NaCl parity failed")
        return 1
    logger.info("Export complete: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
