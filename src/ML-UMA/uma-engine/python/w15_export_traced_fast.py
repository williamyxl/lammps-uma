#!/usr/bin/env python3
"""W15: export devices=1 model_traced.pt with umas_fast_pytorch + merge_mole.

Mirrors export_mp_artifact.py (checkpoint-prepared weights, system atoms for
prepare_for_inference). Does NOT use export_artifact.py's Fe sample + HF
atom_refs path (gated repo 401; Fe+merge_mole fuses the wrong composition).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import torch

# uma-engine/python on path when launched from slurm
from common import (
    DEFAULT_DEVICE,
    DEFAULT_TASK,
    atoms_to_atomic_data,
    inference_settings_with_dtype,
)
from export_wrapper import make_traced_export_wrapper
from model_loader import load_prepared_hydra_model
from trace_patch import apply_trace_patches, restore_trace_patches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("w15_export_traced")


def _load_atoms(path: Path):
    from ase.io import read

    atoms = read(str(path))
    atoms.pbc = True
    return atoms


def export_traced(
    checkpoint: Path,
    output_dir: Path,
    atoms_path: Path,
    dtype: str,
    task: str,
    device: str,
    out_name: str = "model_traced.pt",
) -> dict:
    torch_dtype = getattr(torch, dtype)
    settings = inference_settings_with_dtype(dtype)
    settings.external_graph_gen = True
    settings.activation_checkpointing = False
    settings.execution_mode = "umas_fast_pytorch"
    settings.merge_mole = True

    atoms = _load_atoms(atoms_path)
    n_atoms = len(atoms)
    sample = atoms_to_atomic_data(atoms, task_name=task, settings=settings)
    model, _ckpt, _ = load_prepared_hydra_model(
        str(checkpoint), sample, settings=settings, device="cpu"
    )
    model = model.to(dtype=torch_dtype)
    # Do not load general model_state.pt into umas_fast/merge_mole (key mismatch).

    wrapper = make_traced_export_wrapper(model, task)
    wrapper.eval()

    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)

    data = atoms_to_atomic_data(atoms, task_name=task, settings=settings)
    example = list(wrapper.example_inputs_from_data(data))
    example = [t.to(dev) if torch.is_tensor(t) else t for t in example]
    example[0] = example[0].to(torch_dtype)
    example[2] = example[2].to(torch_dtype)
    example[5] = example[5].to(torch_dtype)
    wrapper = wrapper.to(dev)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / out_name
    apply_trace_patches()
    try:
        with torch.no_grad():
            _ = wrapper(*example)
        traced = torch.jit.trace(wrapper, tuple(example), strict=False)
    finally:
        restore_trace_patches()
    traced.save(str(out_path))
    loaded = torch.jit.load(str(out_path), map_location="cpu")

    # Update metadata inference_settings in place (keep MP fields).
    meta_path = output_dir / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    inf = dict(meta.get("inference_settings") or {})
    inf["execution_mode"] = "umas_fast_pytorch"
    inf["merge_mole"] = True
    inf["activation_checkpointing"] = False
    inf["external_graph_gen"] = True
    inf["base_precision_dtype"] = "float64"
    meta["inference_settings"] = inf
    meta["w15_devices1_traced"] = {
        "atoms": str(atoms_path),
        "n_atoms": n_atoms,
        "out": out_name,
        "execution_mode": "umas_fast_pytorch",
        "merge_mole": True,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    h = hashlib.md5()
    with open(out_path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return {
        "ok": True,
        "path": str(out_path),
        "n_atoms": n_atoms,
        "size": out_path.stat().st_size,
        "md5_1MB": h.hexdigest(),
        "loaded": type(loaded).__name__,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--atoms", type=Path, required=True)
    p.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--out-name", default="model_traced.pt")
    p.add_argument(
        "--backup-existing",
        action="store_true",
        help="Rename existing out-name to *.bak before write",
    )
    args = p.parse_args(argv)

    out = args.output
    target = out / args.out_name
    if args.backup_existing and target.is_file():
        bak = target.with_suffix(target.suffix + ".bak_w15")
        shutil.move(str(target), str(bak))
        logger.info("backed up %s -> %s", target, bak)

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    report = export_traced(
        checkpoint=args.checkpoint,
        output_dir=out,
        atoms_path=args.atoms,
        dtype=args.dtype,
        task=args.task,
        device=args.device,
        out_name=args.out_name,
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
