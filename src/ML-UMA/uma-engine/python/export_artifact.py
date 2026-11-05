#!/usr/bin/env python3
"""
Export a UMA task head as a TorchScript energy artifact (forces via C++ autograd).

Usage:
  PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_artifact.py \\
      --checkpoint $UMA_CHECKPOINT \\
      --dtype float64 --task omat \\
      --output $ENG/artifacts/uma-s-1p2-omat-f64

  # All energy tasks (FP64):
  PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_artifact.py \\
      --checkpoint ... --dtype float64 --all-tasks \\
      --artifacts-root $ENG/artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
    DEFAULT_TASK,
    artifact_dir_name,
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

# UMA-S-1.2 multihead datasets (FairChem); energy head required for export.
DEFAULT_ALL_TASKS = ("oc20", "oc22", "oc25", "omat", "odac", "omc", "omol")


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


def list_energy_tasks(model) -> list[str]:
    names: list[str] = []
    for dataset_name, tasks in model.dataset_to_tasks.items():
        if any(t.property == "energy" for t in tasks):
            names.append(dataset_name)
    return sorted(names)


def export_task(
    model_name: str,
    task_name: str,
    output_dir: Path,
    device: str,
    checkpoint_path: str,
    dtype: str,
    *,
    skip_if_present: bool = False,
    execution_mode: str = "general",
    merge_mole: bool = False,
    activation_checkpointing: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if skip_if_present and (output_dir / "model_traced.pt").is_file():
        logger.info("skip existing: %s", output_dir / "model_traced.pt")
        return {
            "model_name": model_name,
            "task_name": task_name,
            "dtype": dtype,
            "export_format": "skipped",
            "output_dir": str(output_dir),
            "skipped": True,
        }

    settings = inference_settings_with_dtype(dtype)
    # W15 / Tier1: bake umas_fast_pytorch + merge_mole into devices=1 model_traced.pt
    # (MP shards already use these via export_mp_artifact.py).
    settings.execution_mode = execution_mode
    settings.merge_mole = bool(merge_mole)
    settings.activation_checkpointing = bool(activation_checkpointing)
    settings.external_graph_gen = True

    sample = bulk("Fe", "bcc", a=2.87, cubic=True)
    sample_data = atoms_to_atomic_data(sample, task_name, settings)

    logger.info("Loading checkpoint: %s (task=%s dtype=%s)", checkpoint_path, task_name, dtype)
    model, _, _ = load_prepared_hydra_model(
        checkpoint_path, sample_data, settings=settings, device=device
    )
    wrapper = EnergyExportWrapper(clone_prepared_model(model), task_name, traceable=False)
    trace_wrapper = make_traced_export_wrapper(model, task_name)
    wrapper.eval().to(device)
    trace_wrapper.to(device)

    example_data = atoms_to_atomic_data(sample, task_name, settings).to(device)
    example_inputs = wrapper.example_inputs_from_data(example_data)

    export_format, loaded, notes = _attempt_trace(trace_wrapper, example_inputs, output_dir)

    metadata = build_export_metadata(
        model=model,
        model_name=model_name,
        task_name=task_name,
        settings=settings,
        export_format=export_format,
        checkpoint_path=checkpoint_path,
        atom_refs=get_atom_refs(model_name),
        export_notes=notes
        + [
            "forces via C++ torch::autograd::grad on differentiable energy module",
            f"positions {dtype}; energy {dtype}; forces float64",
        ],
    )
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
        "task_name": task_name,
        "dtype": dtype,
        "export_format": export_format,
        "export_notes": notes,
        "output_dir": str(output_dir),
    }

    if loaded is None:
        report["error"] = "export failed"
        (output_dir / "export_report.json").write_text(json.dumps(report, indent=2))
        return report

    logger.info("NaCl energy/force parity (exported module + autograd, task=%s)...", task_name)
    atoms = _nacl_perturbed()
    data = atoms_to_atomic_data(atoms, task_name, settings).to(device)
    inputs = wrapper.example_inputs_from_data(data)

    pos = inputs[0].detach().requires_grad_(True)
    ref_args = (pos, *inputs[1:])
    ref_normed = wrapper(*ref_args)
    ref_energy = _physical_energy(ref_normed, metadata, inputs[1])
    (ref_grad,) = torch.autograd.grad(ref_energy.sum(), pos)
    ref_forces = (-ref_grad).detach().to(torch.float64).cpu().numpy()
    ref_e = float(ref_energy.detach().cpu())

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


# Backward-compatible alias
def export_omat(
    model_name: str,
    output_dir: Path,
    device: str,
    checkpoint_path: str,
    dtype: str,
) -> dict:
    return export_task(
        model_name=model_name,
        task_name="omat",
        output_dir=output_dir,
        device=device,
        checkpoint_path=checkpoint_path,
        dtype=dtype,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export UMA TorchScript energy artifact(s) for uma-engine"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=["float32", "float64"])
    parser.add_argument("--task", default=DEFAULT_TASK, help="Single dataset/task head")
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Export every energy task in the checkpoint (FP64 recommended)",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task list (overrides --all-tasks default list)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir for a single --task export",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
        help="Root for --all-tasks / --tasks outputs (dirs named model-task[-f64])",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("UMA_CHECKPOINT"),
        help="UMA checkpoint (.pt); defaults to $UMA_CHECKPOINT",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda"])
    parser.add_argument(
        "--activation-checkpointing", action="store_true",
        help="Bake activation checkpointing into the traced model (recompute in "
             "backward -> ~3x less activation memory for ~1.3x compute; capacity).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a task if model_traced.pt already exists",
    )
    parser.add_argument(
        "--execution-mode",
        default="general",
        choices=("general", "umas_fast_pytorch"),
        help="FairChem execution backend baked into model_traced.pt "
        "(W15: umas_fast_pytorch for devices=1 product).",
    )
    parser.add_argument(
        "--merge-mole",
        action="store_true",
        help="Fuse MOLE experts (fixed composition — required with umas_fast_pytorch).",
    )
    parser.add_argument(
        "--discover-tasks",
        action="store_true",
        help="Load checkpoint once and print energy task names, then exit",
    )
    args = parser.parse_args(argv)
    args.device = resolve_device(args.device)

    try:
        checkpoint = resolve_checkpoint(args.model, args.checkpoint)

        if args.discover_tasks or args.all_tasks or args.tasks:
            # Probe available energy tasks from the prepared model.
            settings = inference_settings_with_dtype(args.dtype)
            sample = bulk("Fe", "bcc", a=2.87, cubic=True)
            probe_task = args.task if not args.all_tasks and not args.tasks else "omat"
            sample_data = atoms_to_atomic_data(sample, probe_task, settings)
            model, _, _ = load_prepared_hydra_model(
                checkpoint, sample_data, settings=settings, device=args.device
            )
            available = list_energy_tasks(model)
            logger.info("Energy tasks in checkpoint: %s", available)
            if args.discover_tasks:
                print("\n".join(available))
                return 0
            del model

        if args.all_tasks or args.tasks:
            if args.tasks.strip():
                tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
            else:
                # Prefer tasks actually present in the checkpoint
                tasks = available if available else list(DEFAULT_ALL_TASKS)
            missing = [t for t in tasks if t not in available]
            if missing:
                logger.warning("Requested tasks missing energy head (skipped): %s", missing)
                tasks = [t for t in tasks if t in available]
            if not tasks:
                logger.error("No energy tasks to export")
                return 1
            root = args.artifacts_root
            summaries = []
            failed = False
            for task in tasks:
                out = root / artifact_dir_name(args.model, task, args.dtype)
                report = export_task(
                    model_name=args.model,
                    task_name=task,
                    output_dir=out,
                    device=args.device,
                    checkpoint_path=checkpoint,
                    dtype=args.dtype,
                    skip_if_present=args.skip_existing,
                    execution_mode=args.execution_mode,
                    merge_mole=bool(args.merge_mole),
                    activation_checkpointing=bool(args.activation_checkpointing),
                )
                summaries.append(report)
                if report.get("export_format") == "none":
                    failed = True
                if report.get("nacl_parity") and not report["nacl_parity"]["passed"]:
                    failed = True
            summary_path = root / f"export_all_{args.dtype}_summary.json"
            summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            logger.info("Wrote %s", summary_path)
            return 1 if failed else 0

        output = args.output
        if output is None:
            output = args.artifacts_root / artifact_dir_name(
                args.model, args.task, args.dtype
            )
        report = export_task(
            model_name=args.model,
            task_name=args.task,
            output_dir=output,
            device=args.device,
            checkpoint_path=checkpoint,
            dtype=args.dtype,
            skip_if_present=args.skip_existing,
            execution_mode=args.execution_mode,
            merge_mole=bool(args.merge_mole),
            activation_checkpointing=bool(args.activation_checkpointing),
        )
    except Exception:
        traceback.print_exc()
        return 1

    if report.get("export_format") == "none":
        return 1
    if report.get("nacl_parity") and not report["nacl_parity"]["passed"]:
        logger.error("NaCl parity failed")
        return 1
    logger.info("Export complete: %s", report.get("output_dir", args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
