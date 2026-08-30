#!/usr/bin/env python3
"""Dump eager FP64 UMA weights + arch sidecar for native multi-GPU (no Ray).

Writes into an artifact directory (alongside model_traced.pt when present):
  model_state.pt       — state_dict of prepared HydraModel (CPU, FP64 tensors)
  eager_arch.json      — task / cutoff / max_neighbors / dtype / module type
  eager_export_report.json — optional single-GPU E vs wrapper sanity

Usage:
  PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_eager_weights.py \\
      --checkpoint $UMA_CHECKPOINT \\
      --dtype float64 --task omat \\
      --output $ENG/artifacts/uma-s-1p2-omat-f64
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk

from checkpoints import resolve_checkpoint
from common import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_TASK,
    atoms_to_atomic_data,
    inference_settings_with_dtype,
    resolve_device,
)
from model_loader import load_prepared_hydra_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _nacl_perturbed(seed: int = 0, noise: float = 0.05):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    rng = np.random.default_rng(seed)
    atoms.positions = atoms.positions + rng.normal(0.0, noise, size=atoms.positions.shape)
    atoms.pbc = True
    return atoms


def export_eager_weights(
    checkpoint: Path,
    output_dir: Path,
    *,
    dtype: str,
    task: str,
    device: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = getattr(torch, dtype)
    settings = inference_settings_with_dtype(dtype)
    settings.external_graph_gen = True
    settings.activation_checkpointing = False
    settings.execution_mode = "general"

    atoms = _nacl_perturbed()
    sample = atoms_to_atomic_data(atoms, task_name=task, settings=settings)
    model, ckpt, _ = load_prepared_hydra_model(
        str(checkpoint), sample, settings=settings, device="cpu"
    )
    model = model.to(dtype=torch_dtype)

    state_path = output_dir / "model_state.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dtype": dtype,
            "task": task,
            "checkpoint_path": str(checkpoint.resolve()),
        },
        state_path,
    )

    arch = {
        "model_type": type(model).__module__ + "." + type(model).__name__,
        "dtype": dtype,
        "task": task,
        "checkpoint_path": str(checkpoint.resolve()),
        "cutoff": float(getattr(settings, "radius", 6.0) or 6.0),
        "max_neighbors": int(getattr(settings, "max_neighbors", 300) or 300),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "state_dict_keys": sorted(model.state_dict().keys())[:32],
        "state_dict_key_count": len(model.state_dict()),
        "notes": [
            "Eager weights for native Kokkos multi-GPU path (not TorchScript).",
            "Full UMA; do not use with Ray GP worker.",
        ],
    }
    # Prefer concrete NL knobs from common/export metadata if present on model.
    try:
        from common import GRAPH_MAX_NEIGHBORS, GRAPH_RADIUS

        arch["cutoff"] = float(GRAPH_RADIUS)
        arch["max_neighbors"] = int(GRAPH_MAX_NEIGHBORS)
    except Exception:
        pass

    arch_path = output_dir / "eager_arch.json"
    arch_path.write_text(json.dumps(arch, indent=2) + "\n", encoding="utf-8")

    # Sanity: one forward energy logit on CPU (cheap; not force parity).
    report: dict = {"ok": True, "state_path": str(state_path), "arch_path": str(arch_path)}
    try:
        model.eval()
        with torch.no_grad():
            data = atoms_to_atomic_data(atoms, task_name=task, settings=settings)
            if hasattr(data, "pos") and data.pos.dtype != torch_dtype:
                data.pos = data.pos.to(torch_dtype)
            out = model(data)
            # HydraModel output shape varies; record type only.
            report["forward_out_type"] = type(out).__name__
            report["device_used"] = device
    except Exception as exc:
        report["ok"] = False
        report["forward_error"] = str(exc)
        logger.warning("eager forward sanity failed: %s", exc)

    report_path = output_dir / "eager_export_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d params)", state_path, arch["n_parameters"])
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", default=DEFAULT_DTYPE, choices=("float64", "float32"))
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    args = p.parse_args(argv)

    if args.dtype != "float64":
        logger.warning("Policy is FP64-only for UMA production; got dtype=%s", args.dtype)

    # Accept a filesystem path or a FairChem model name (e.g. uma-s-1p2).
    ckpt_arg = Path(args.checkpoint).expanduser()
    if ckpt_arg.is_file():
        ckpt = str(ckpt_arg.resolve())
    else:
        ckpt = resolve_checkpoint(str(args.checkpoint))
    device = resolve_device(args.device)
    report = export_eager_weights(
        Path(ckpt), args.output, dtype=args.dtype, task=args.task, device=device
    )
    return 0 if report.get("ok", False) else 1


if __name__ == "__main__":
    sys.exit(main())
