#!/usr/bin/env python3
"""Parity: LibTorch artifact (+ autograd forces) vs ASE FAIRChemCalculator.

Supports float32 and float64. For FP64, ASE uses InferenceSettings.base_precision_dtype=float64
and the artifact must be the matching uma-s-1p2-omat-f64 export.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from common import atoms_to_atomic_data, inference_settings_with_dtype, resolve_device
from metadata import ExportMetadata, denorm_from_metadata, undo_refs_from_metadata


def nacl_perturbed(seed: int = 0, noise: float = 0.05) -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    rng = np.random.default_rng(seed)
    atoms.positions = atoms.positions + rng.normal(0.0, noise, size=atoms.positions.shape)
    atoms.pbc = True
    return atoms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Compute precision for ASE oracle and exported module",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Artifact dir (default: uma-engine/artifacts/uma-s-1p2-omat[-f64])",
    )
    parser.add_argument("--checkpoint", default=os.environ.get("UMA_CHECKPOINT"),
                        help="UMA checkpoint (.pt); defaults to $UMA_CHECKPOINT")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument(
        "--energy-tol",
        type=float,
        default=None,
        help="Absolute energy tolerance (default: 1e-2 f32, 1e-6 f64)",
    )
    parser.add_argument(
        "--force-tol",
        type=float,
        default=None,
        help="Max |ΔF| tolerance (default: 5e-2 f32, 1e-5 f64)",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    dtype = args.dtype
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32

    root = Path(__file__).resolve().parents[1]
    if args.artifact is None:
        suffix = "-f64" if dtype == "float64" else ""
        args.artifact = root / "artifacts" / f"uma-s-1p2-omat{suffix}"

    energy_tol = args.energy_tol if args.energy_tol is not None else (
        1e-6 if dtype == "float64" else 1e-2
    )
    force_tol = args.force_tol if args.force_tol is not None else (
        1e-5 if dtype == "float64" else 5e-2
    )

    atoms = nacl_perturbed()
    if args.npz and args.npz.is_file():
        data = np.load(args.npz)
        atoms = Atoms(
            numbers=data["numbers"],
            positions=data["positions"],
            cell=data["cell"],
            pbc=True,
        )

    settings = inference_settings_with_dtype(dtype)

    # ASE oracle at requested precision
    predictor = load_predict_unit(
        args.checkpoint, device=device, inference_settings=settings
    )
    calc = FAIRChemCalculator(predictor, task_name="omat")
    atoms_ase = atoms.copy()
    atoms_ase.calc = calc
    e_ase = float(atoms_ase.get_potential_energy())
    f_ase = np.asarray(atoms_ase.get_forces(), dtype=np.float64)

    # Exported module + autograd (mirrors C++ Predictor)
    meta = ExportMetadata.load(args.artifact / "metadata.json")
    data = atoms_to_atomic_data(atoms, "omat", settings).to(device)
    module = torch.jit.load(str(args.artifact / "model_traced.pt"), map_location=device)
    module.eval()

    pos = data.pos.detach().to(torch_dtype).requires_grad_(True)
    charge = data.charge.to(device)
    spin = data.spin.to(device)
    if charge.numel() == 1 and charge.dim() == 1:
        charge = charge.squeeze(0)
    if spin.numel() == 1 and spin.dim() == 1:
        spin = spin.squeeze(0)

    cell = data.cell.squeeze(0).to(device=device, dtype=torch_dtype)
    cell_offsets = data.cell_offsets.to(device=device, dtype=torch_dtype)

    normed = module(
        pos,
        data.atomic_numbers.to(device),
        cell,
        data.pbc.squeeze(0).to(device),
        data.edge_index.to(device),
        cell_offsets,
        charge,
        spin,
    )
    energy = denorm_from_metadata(normed.to(torch_dtype), meta)
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
    energy = undo_refs_from_metadata(meta, data.atomic_numbers.to(device), batch, energy)
    (grad,) = torch.autograd.grad(energy.sum(), pos)
    forces = (-grad).detach().to(torch.float64).cpu().numpy()
    e_exp = float(energy.detach().double().cpu())

    de = abs(e_exp - e_ase)
    df = float(np.max(np.abs(forces - f_ase)))
    out = {
        "dtype": dtype,
        "artifact": str(args.artifact),
        "ase_energy": e_ase,
        "exported_energy": e_exp,
        "abs_energy_error": de,
        "max_force_error": df,
        "ase_fmax": float(np.max(np.abs(f_ase))),
        "energy_tol": energy_tol,
        "force_tol": force_tol,
        "passed": bool(de < energy_tol and df < force_tol),
    }
    print(json.dumps(out, indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
