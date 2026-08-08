#!/usr/bin/env python3
"""Export per-rank TorchScript UMA modules with uma_peer collective ops.

Build-time only (FairChem Python). Runtime is C++ LibTorch + kokkos_peer.

Writes into an artifact dir (e.g. artifacts/uma-s-1p2-omat-f64/):
  model_mp_w{N}_r{R}.pt   — traced EnergyExportWrapper with GP path baked for rank R
  model_mp_export.json    — report

Usage:
  PYTHONPATH=$ENG/python:$PYTHONPATH python $ENG/python/export_mp_artifact.py \\
      --checkpoint /work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt \\
      --output $ENG/artifacts/uma-s-1p2-omat-f64 \\
      --world 2 --dtype float64 --task omat --device cuda
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
from export_wrapper import make_traced_export_wrapper
from model_loader import load_prepared_hydra_model
from trace_patch import apply_trace_patches, restore_trace_patches
from uma_peer_ops import install_export_ops, patch_fairchem_gp_utils, set_export_rank

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _nacl_perturbed(seed: int = 0, noise: float = 0.05):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    rng = np.random.default_rng(seed)
    atoms.positions = atoms.positions + rng.normal(0.0, noise, size=atoms.positions.shape)
    atoms.pbc = True
    return atoms


def _load_atoms(path: Path | None):
    """Load ASE atoms from extxyz/any ASE-readable file, or default NaCl64."""
    if path is None:
        return _nacl_perturbed()
    from ase.io import read

    atoms = read(str(path))
    atoms.pbc = True
    return atoms


def _load_state_dict_into(model: torch.nn.Module, state_path: Path) -> None:
    obj = torch.load(state_path, map_location="cpu", weights_only=False)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("state_dict missing %d keys (first: %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning("state_dict unexpected %d keys (first: %s)", len(unexpected), unexpected[:3])


def _shard_example_edges(
    example: list,
    world: int,
    rank: int,
) -> list:
    """Filter edge_index / cell_offsets to FairChem center-partition for ``rank``.

    example layout matches EnergyExportWrapper.forward args:
      pos, atomic_numbers, cell, pbc, edge_index, cell_offsets, charge, spin
    """
    pos, z, cell, pbc, edge_index, cell_offsets, charge, spin = example
    n_atoms = int(pos.shape[0])
    nodes = torch.arange(n_atoms, dtype=torch.long, device=edge_index.device)
    parts = torch.tensor_split(nodes, world)
    node_ids = parts[rank].contiguous()
    keep = torch.isin(edge_index[1], node_ids)
    idx = keep.nonzero().squeeze(-1)
    edge_index = edge_index.index_select(1, idx).contiguous()
    cell_offsets = cell_offsets.index_select(0, idx).contiguous()
    return [pos, z, cell, pbc, edge_index, cell_offsets, charge, spin]


def export_mp_rank(
    *,
    checkpoint: Path,
    output_dir: Path,
    world: int,
    rank: int,
    dtype: str,
    task: str,
    device: str,
    state_path: Path | None,
    atoms_path: Path | None = None,
) -> dict:
    import os

    install_export_ops()
    set_export_rank(rank, world)
    restore_gp = patch_fairchem_gp_utils(world, rank)

    # Isolate one GPU so traced constants say cuda:0 (== this rank's device).
    # Runtime workers must also set CUDA_VISIBLE_DEVICES=<rank>.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    if "cuda" in str(device):
        device = "cuda"

    torch_dtype = getattr(torch, dtype)
    settings = inference_settings_with_dtype(dtype)
    # External graph: vesin will supply edges at C++ runtime; match devices=1 contract.
    settings.external_graph_gen = True
    settings.activation_checkpointing = False
    settings.execution_mode = "general"

    atoms = _load_atoms(atoms_path)
    n_atoms = len(atoms)
    sample = atoms_to_atomic_data(atoms, task_name=task, settings=settings)
    model, _ckpt, _ = load_prepared_hydra_model(
        str(checkpoint), sample, settings=settings, device="cpu"
    )
    model = model.to(dtype=torch_dtype)
    if state_path is not None and state_path.is_file():
        _load_state_dict_into(model, state_path)
        logger.info("Loaded eager weights from %s", state_path)

    wrapper = make_traced_export_wrapper(model, task)
    wrapper.eval()

    # Move example tensors to target device for tracing.
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    data = atoms_to_atomic_data(atoms, task_name=task, settings=settings)
    # Build example inputs via wrapper helper, then move.
    example = list(wrapper.example_inputs_from_data(data))
    example = [t.to(dev) if torch.is_tensor(t) else t for t in example]
    # Ensure FP64 positions/cell/offsets.
    example[0] = example[0].to(torch_dtype)
    example[2] = example[2].to(torch_dtype)
    example[5] = example[5].to(torch_dtype)
    # FairChem otf_graph=False does not filter edges; pre-shard like graph_shard.h.
    example = _shard_example_edges(example, world, rank)
    wrapper = wrapper.to(dev)

    notes: list[str] = []
    # Partition offsets are baked into TorchScript at trace time → n-specific.
    out_path = output_dir / f"model_mp_w{world}_n{n_atoms}_r{rank}.pt"
    if atoms_path is None and n_atoms == 64:
        # Keep legacy filename for the default NaCl64 export.
        out_path = output_dir / f"model_mp_w{world}_r{rank}.pt"
    try:
        apply_trace_patches()
        try:
            with torch.no_grad():
                _ = wrapper(*example)
            traced = torch.jit.trace(wrapper, tuple(example), strict=False)
        finally:
            restore_trace_patches()
        traced.save(str(out_path))
        # Reload sanity
        loaded = torch.jit.load(str(out_path), map_location="cpu")
        notes.append(f"saved {out_path}")
        notes.append(f"reloaded type={type(loaded).__name__}")
        ok = True
        err = None
    except Exception as exc:  # noqa: BLE001
        ok = False
        err = f"{type(exc).__name__}: {exc}"
        notes.append(err)
        logger.exception("MP trace failed for rank %d", rank)
    finally:
        restore_gp()

    return {
        "ok": ok,
        "world": world,
        "rank": rank,
        "n_atoms": n_atoms,
        "path": str(out_path) if ok else None,
        "device": str(dev),
        "error": err,
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--ranks", type=str, default="", help="comma ranks (default: all)")
    p.add_argument("--dtype", default=DEFAULT_DTYPE, choices=("float64", "float32"))
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument(
        "--state-dict",
        type=Path,
        default=None,
        help="optional model_state.pt to load after prepare_for_inference",
    )
    p.add_argument(
        "--atoms",
        type=Path,
        default=None,
        help="ASE-readable structure for trace (default: perturbed NaCl 2x2x2 = 64). "
        "MP TS bakes partition offsets → re-export per n_atoms.",
    )
    args = p.parse_args(argv)

    if args.dtype != "float64":
        logger.warning("Policy is FP64-only; got dtype=%s", args.dtype)
    if args.world < 2:
        logger.error("--world must be >= 2 for MP export")
        return 2

    ckpt_arg = Path(args.checkpoint).expanduser()
    if ckpt_arg.is_file():
        ckpt = ckpt_arg.resolve()
    else:
        ckpt = Path(resolve_checkpoint(str(args.checkpoint)))

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dict
    if state_path is None:
        cand = output_dir / "model_state.pt"
        if cand.is_file():
            state_path = cand

    device = resolve_device(args.device)
    if args.ranks.strip():
        ranks = [int(x) for x in args.ranks.split(",")]
    else:
        ranks = list(range(args.world))

    results = []
    for r in ranks:
        logger.info("Exporting MP artifact world=%d rank=%d on %s", args.world, r, device)
        results.append(
            export_mp_rank(
                checkpoint=ckpt,
                output_dir=output_dir,
                world=args.world,
                rank=r,
                dtype=args.dtype,
                task=args.task,
                device=device,
                state_path=state_path,
                atoms_path=args.atoms.expanduser().resolve() if args.atoms else None,
            )
        )

    report = {
        "ok": all(r.get("ok") for r in results),
        "world": args.world,
        "dtype": args.dtype,
        "task": args.task,
        "checkpoint": str(ckpt),
        "state_dict": str(state_path) if state_path else None,
        "ranks": results,
        "backend": "Kokkos+LibTorch+vesin",
        "notes": [
            "Per-rank TorchScript with uma_peer ops; C++ runtime owns collectives.",
            "Not Ray / not FairChem process-GP.",
        ],
    }
    report_path = output_dir / "model_mp_export.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s ok=%s", report_path, report["ok"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
