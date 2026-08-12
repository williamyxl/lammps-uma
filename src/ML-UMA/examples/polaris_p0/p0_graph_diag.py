#!/usr/bin/env python3
"""Diagnose the LAMMPS-vs-tensor energy gap: is it the neighbor graph?

Builds, for each system:
  - FairChem's atoms_to_atomic_data graph (used by the tensor oracle), and
  - the engine's vesin cell_list graph (used by the LAMMPS pair style path),
then compares edge counts / undirected edge sets, and evaluates the traced
module energy on BOTH graphs to attribute the dE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from p0_common import ART_F64, ENGINE, SYSTEMS, TASK, atoms_from_data, results_dir

sys.path.insert(0, str(ENGINE / "python"))
from common import atoms_to_atomic_data, inference_settings_with_dtype  # noqa: E402
from metadata import ExportMetadata, denorm_from_metadata, undo_refs_from_metadata  # noqa: E402


def eval_energy(module, meta, z, pos, cell, pbc, edge_index, cell_offsets, charge, spin):
    p = pos.detach().to(torch.float64).requires_grad_(True)
    normed = module(p, z, cell, pbc, edge_index, cell_offsets, charge, spin)
    e = denorm_from_metadata(normed.to(torch.float64), meta)
    batch = torch.zeros(p.shape[0], dtype=torch.long, device=p.device)
    e = undo_refs_from_metadata(meta, z, batch, e)
    (grad,) = torch.autograd.grad(e.sum(), p)
    return float(e.detach().double().cpu()), (-grad).detach().double().cpu().numpy()


def undirected(a, b):
    return set(map(tuple, np.sort(np.stack([a, b], 1), axis=1)))


def run(sysname: str, device) -> dict:
    art = ART_F64
    meta = ExportMetadata.load(art / "metadata.json")
    module = torch.jit.load(str(art / "model_traced.pt"), map_location=device)
    module.eval()

    atoms = atoms_from_data(sysname)
    s = inference_settings_with_dtype("float64")
    s.external_graph_gen = True
    s.execution_mode = "general"
    s.merge_mole = False
    data = atoms_to_atomic_data(atoms, TASK, s).to(device)

    z = data.atomic_numbers.to(device)
    cell = data.cell.squeeze(0).to(device=device, dtype=torch.float64)
    pbc = data.pbc.squeeze(0).to(device)
    charge = data.charge.to(device)
    spin = data.spin.to(device)
    if charge.numel() == 1 and charge.dim() == 1:
        charge = charge.squeeze(0)
    if spin.numel() == 1 and spin.dim() == 1:
        spin = spin.squeeze(0)
    pos = data.pos.to(device=device, dtype=torch.float64)

    # --- graph A: FairChem ---
    ei_fc = data.edge_index.to(device)
    co_fc = data.cell_offsets.to(device=device, dtype=torch.float64)
    e_fc, f_fc = eval_energy(module, meta, z, pos, cell, pbc, ei_fc, co_fc, charge, spin)

    # --- graph B: engine vesin cell_list (LAMMPS path) ---
    torch.ops.load_library(str(ENGINE / "third_party/vesin/lib/libvesin_torch.so"))
    NL = torch.classes.vesin._NeighborList(float(meta.cutoff), True, False, "cell_list")
    out = NL.compute(pos.contiguous(), cell.contiguous(), pbc.contiguous(), "ijS", True)
    vi = out[0].to(torch.int64)
    vj = out[1].to(torch.int64)
    vS = out[2].to(torch.float64)
    # FairChem convention: row0=neighbor(src), row1=center(tgt); vesin ij = center i, neighbor j
    ei_ve = torch.stack([vj, vi], 0).contiguous()
    # cell_offsets in cartesian: S @ cell
    co_ve = (vS @ cell).contiguous()
    e_ve, f_ve = eval_energy(module, meta, z, pos, cell, pbc, ei_ve, co_ve, charge, spin)

    fc_set = undirected(ei_fc[0].cpu().numpy(), ei_fc[1].cpu().numpy())
    ve_set = undirected(vi.cpu().numpy(), vj.cpu().numpy())

    fmag = np.linalg.norm(f_fc - f_ve, axis=1)
    return {
        "system": sysname,
        "natoms": len(atoms),
        "cutoff": float(meta.cutoff),
        "max_neighbors": int(getattr(meta, "max_neighbors", -1)),
        "edges_fairchem": int(ei_fc.shape[1]),
        "edges_vesin": int(ei_ve.shape[1]),
        "undirected_fairchem": len(fc_set),
        "undirected_vesin": len(ve_set),
        "in_fc_not_vesin": len(fc_set - ve_set),
        "in_vesin_not_fc": len(ve_set - fc_set),
        "E_graph_fairchem": e_fc,
        "E_graph_vesin": e_ve,
        "dE_graph_fc_vs_vesin": abs(e_fc - e_ve),
        "force_max_fc_vs_vesin": float(fmag.max()),
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = results_dir()
    recs = [run(s, device) for s in sorted(SYSTEMS)]
    (out / "P0_GRAPH_DIAG.json").write_text(json.dumps(recs, indent=2) + "\n")
    for r in recs:
        print(json.dumps(r, indent=2))
    print(f"\nwrote {out / 'P0_GRAPH_DIAG.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
