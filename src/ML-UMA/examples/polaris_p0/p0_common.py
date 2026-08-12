#!/usr/bin/env python3
"""Shared config for Polaris Phase-P0 single-node validation.

The LAMMPS `.data` file is the SINGLE authoritative geometry source for every
test (LAMMPS-UMA, ASE-FC oracle, tensor parity, MP export). The reduced-precision
`.extxyz` is intentionally NOT used and is git-untracked: comparing a 16-digit
`.data` against an 8-digit `.extxyz` inflated water888 |dE| from ~0 to 1.4e-7 eV.
Any tool that needs an ASE/extxyz input derives it on the fly from the `.data`
at full precision via `atoms_from_data()` / `write_full_precision_extxyz()`.
"""
from __future__ import annotations

import os
from pathlib import Path

EX = Path(__file__).resolve().parent
ML_UMA = EX.parents[1]          # .../src/ML-UMA
ENGINE = ML_UMA / "uma-engine"
ROOT = EX.parents[3]            # .../lammps-uma

DEFAULT_CKPT = Path(
    os.environ.get(
        "UMA_CHECKPOINT",
        "/lus/eagle/projects/RAPINS/xiaoliyan/polaris/uma-s-1p2.pt",
    )
)
ART_F64 = Path(
    os.environ.get("UMA_ARTIFACT_DIR", str(ENGINE / "artifacts" / "uma-s-1p2-omat-f64"))
)

# System registry. elements = LAMMPS type order (type 1, type 2, ...).
# `data` (full precision) is authoritative; there is deliberately no `extxyz` key.
SYSTEMS = {
    "nacl666": {
        "natoms": 1728,
        "elements": ["Na", "Cl"],
        "data": ML_UMA / "examples" / "multi_gpu_nacl6" / "structures"
        / "nacl6_nvt_300K_atomic_metal.data",
    },
    "water888": {
        "natoms": 648,
        "elements": ["O", "H"],
        "data": ML_UMA / "examples" / "water888" / "water_nvt_300K_atomic_metal.data",
    },
}


def atoms_from_data(sysname: str):
    """Read the authoritative full-precision LAMMPS .data as an ASE Atoms with
    correct species (LAMMPS type order) and PBC set."""
    from ase.io import read
    from ase.data import atomic_numbers as _AN

    info = SYSTEMS[sysname]
    atoms = read(str(info["data"]), format="lammps-data", atom_style="atomic", units="metal")
    if atoms.has("type"):
        elems = info["elements"]
        atoms.set_atomic_numbers([_AN[elems[t - 1]] for t in atoms.get_array("type")])
    atoms.pbc = True
    return atoms


def write_full_precision_extxyz(sysname: str, out_path) -> Path:
    """Write a full-precision extxyz derived from the .data for tools that require
    an extxyz input (e.g. MP export). Uses maximal float formatting so no geometry
    precision is lost."""
    from ase.io import write

    out_path = Path(out_path)
    atoms = atoms_from_data(sysname)
    write(str(out_path), atoms, format="extxyz")  # ASE writes ~%.8f by default...
    # ...so re-emit positions at full precision explicitly.
    _rewrite_extxyz_full_precision(out_path, atoms)
    return out_path


def _rewrite_extxyz_full_precision(path: Path, atoms) -> None:
    import numpy as np

    cell = np.array(atoms.cell, dtype=float)
    lat = " ".join(repr(float(v)) for v in cell.reshape(-1))
    lines = [str(len(atoms)),
             f'Lattice="{lat}" Properties=species:S:1:pos:R:3 pbc="T T T"']
    syms = atoms.get_chemical_symbols()
    for s, p in zip(syms, atoms.get_positions()):
        lines.append(f"{s} {repr(float(p[0]))} {repr(float(p[1]))} {repr(float(p[2]))}")
    Path(path).write_text("\n".join(lines) + "\n")

TASK = "omat"
NVT_STEPS = int(os.environ.get("P0_NVT_STEPS", "10"))
NVT_TEMP_K = 300.0
TIMESTEP_PS = 0.001  # metal units, 1 fs

# Parity gates (Phase P0).
GATE_ABS_DE = 1e-6       # eV, absolute energy tolerance
GATE_REL_DE = 1e-9       # relative energy tolerance (either passes)
GATE_MAX_DF = 1e-5       # eV/Angstrom, per-atom max |dF|


def results_dir() -> Path:
    jobid = os.environ.get("PBS_JOBID", os.environ.get("SLURM_JOB_ID", "manual"))
    d = EX / "results" / jobid
    d.mkdir(parents=True, exist_ok=True)
    return d
