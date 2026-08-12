#!/usr/bin/env python3
"""Phase-P0 TENSOR-level parity (no LAMMPS, no dump, no %g round-trip).

For each system, on the identical geometry and the identical AtomicData graph:
  A. FairChem calculator  -> E, F               (the ASE oracle path)
  B. traced artifact + torch.autograd.grad -> E, F   (mirrors the C++ engine
     uma::Predictor exactly: same module, same denorm/element-ref postprocess)

Both are computed in one process from tensors, so the reported dE/dF isolate
floating-point reduction-order between the two implementations from any output
formatting effect. This is the definitive check that LAMMPS %.17g output is not
the parity limiter.

Runs both recipes:
  general   -> compare against the traced (devices-1) model contract
  fastmerge -> umas_fast_pytorch + merge_mole (the multi-GPU MP-shard contract)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from p0_common import ART_F64, DEFAULT_CKPT, ENGINE, SYSTEMS, TASK, atoms_from_data, results_dir

sys.path.insert(0, str(ENGINE / "python"))
from common import atoms_to_atomic_data, inference_settings_with_dtype, resolve_device  # noqa: E402
from metadata import ExportMetadata, denorm_from_metadata, undo_refs_from_metadata  # noqa: E402


def settings_for(recipe: str):
    s = inference_settings_with_dtype("float64")
    s.external_graph_gen = True
    s.activation_checkpointing = False
    if recipe == "fastmerge":
        s.execution_mode = "umas_fast_pytorch"
        s.merge_mole = True
    else:
        s.execution_mode = "general"
        s.merge_mole = False
    return s


def run_system(sysname: str, recipe: str, ckpt: str, device, artifact: Path) -> dict:
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    # Authoritative full-precision geometry from the LAMMPS .data.
    atoms = atoms_from_data(sysname)
    settings = settings_for(recipe)

    # ---- A. FairChem calculator (oracle) --------------------------------
    predictor = load_predict_unit(ckpt, device=device, inference_settings=settings)
    calc = FAIRChemCalculator(predictor, task_name=TASK)
    a = atoms.copy()
    a.calc = calc
    e_ase = float(a.get_potential_energy())
    f_ase = np.asarray(a.get_forces(), dtype=np.float64)

    # ---- B. traced artifact + autograd (mirrors C++ uma::Predictor) -----
    meta = ExportMetadata.load(artifact / "metadata.json")
    data = atoms_to_atomic_data(atoms, TASK, settings).to(device)
    module = torch.jit.load(str(artifact / "model_traced.pt"), map_location=device)
    module.eval()

    pos = data.pos.detach().to(torch.float64).requires_grad_(True)
    charge = data.charge.to(device)
    spin = data.spin.to(device)
    if charge.numel() == 1 and charge.dim() == 1:
        charge = charge.squeeze(0)
    if spin.numel() == 1 and spin.dim() == 1:
        spin = spin.squeeze(0)
    cell = data.cell.squeeze(0).to(device=device, dtype=torch.float64)
    cell_offsets = data.cell_offsets.to(device=device, dtype=torch.float64)

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
    energy = denorm_from_metadata(normed.to(torch.float64), meta)
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
    energy = undo_refs_from_metadata(meta, data.atomic_numbers.to(device), batch, energy)
    (grad,) = torch.autograd.grad(energy.sum(), pos)
    f_eng = (-grad).detach().to(torch.float64).cpu().numpy()
    e_eng = float(energy.detach().double().cpu())

    dE = abs(e_eng - e_ase)
    mag = np.linalg.norm(f_eng - f_ase, axis=1)
    return {
        "system": sysname,
        "recipe": recipe,
        "natoms": len(atoms),
        "energy_ase_eV": e_ase,
        "energy_engine_eV": e_eng,
        "abs_dE": dE,
        "dE_per_atom": dE / len(atoms),
        "force_max_per_atom": float(mag.max()),
        "force_mean_per_atom": float(mag.mean()),
        "ase_fmax": float(np.abs(f_ase).max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="*", default=sorted(SYSTEMS))
    ap.add_argument("--recipes", nargs="*", default=["general", "fastmerge"])
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = resolve_device(args.device)
    out = results_dir()
    recs = []
    for s in args.systems:
        for rcp in args.recipes:
            rec = run_system(s, rcp, args.ckpt, device, ART_F64)
            rec["geom"] = "data"
            recs.append(rec)
            print(json.dumps(rec, indent=2))

    (out / "P0_TENSOR_PARITY.json").write_text(json.dumps(recs, indent=2) + "\n")

    lines = [
        "# Phase P0 - TENSOR-level parity (in-process, no LAMMPS / no %g)",
        "",
        "FairChem calculator vs traced artifact + autograd (mirrors the C++ engine),",
        "same geometry, same AtomicData graph, compared from tensors. Residual is",
        "pure FP64 reduction-order between the two implementations.",
        "",
        "| system | N | recipe | E_engine (eV) | E_ase (eV) | |dE| | dE/atom | max|dF| | mean|dF| |",
        "|---|---:|:--:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in recs:
        lines.append(
            f"| {r['system']} | {r['natoms']} | {r['recipe']} | "
            f"{r['energy_engine_eV']:.9f} | {r['energy_ase_eV']:.9f} | "
            f"{r['abs_dE']:.3e} | {r['dE_per_atom']:.3e} | "
            f"{r['force_max_per_atom']:.3e} | {r['force_mean_per_atom']:.3e} |"
        )
    lines += ["", "Compare these dE against the LAMMPS P0_REPORT.md values: if they",
              "match, output formatting is not the limiter (reduction-order is).", ""]
    (out / "P0_TENSOR_PARITY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out / 'P0_TENSOR_PARITY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
