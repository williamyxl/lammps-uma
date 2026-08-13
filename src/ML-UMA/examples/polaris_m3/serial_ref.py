#!/usr/bin/env python3
"""Serial single-GPU reference E+F for a LAMMPS .data (checkpointing on, no GP).
Writes <out>.npz. For validating GP force correctness at small N."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np, torch
from ase.io import read
from ase.data import atomic_numbers as AN
ENG = Path(__file__).resolve().parents[1].parent / "uma-engine"
sys.path.insert(0, str(ENG / "python"))
from common import inference_settings_with_dtype  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default=os.environ.get("UMA_CHECKPOINT"))
a_ = ap.parse_args()
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
atoms = read(a_.data, format="lammps-data", atom_style="atomic", units="metal")
if atoms.has("type"):
    atoms.set_atomic_numbers([AN[["Na","Cl"][t-1]] for t in atoms.get_array("type")])
atoms.pbc = True
s = inference_settings_with_dtype("float64")
s.external_graph_gen = True; s.activation_checkpointing = True
s.execution_mode = "general"; s.merge_mole = False
pred = load_predict_unit(a_.ckpt, device="cuda", inference_settings=s, workers=1)
atoms.calc = FAIRChemCalculator(pred, task_name="omat")
e = float(atoms.get_potential_energy()); f = np.asarray(atoms.get_forces(), float)
np.savez(a_.out, forces=f, energy_eV=np.array(e), natoms=len(atoms))
print(f"serial ref N={len(atoms)} E={e:.9f} fmax={np.abs(f).max():.4f}")
