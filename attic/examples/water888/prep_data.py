#!/usr/bin/env python3
"""Write LAMMPS atomic data from water888_2000step.extxyz (untimed prep)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from ase.io import read

NATOMS_EXPECTED = 1536  # 512 H2O


def write_data(atoms: Atoms, path: Path, title: str) -> list[str]:
    Z = atoms.get_atomic_numbers()
    uniq = sorted(set(int(z) for z in Z))
    z_to_type = {z: i + 1 for i, z in enumerate(uniq)}
    types = np.array([z_to_type[int(z)] for z in Z], dtype=np.int32)
    symbols = [chemical_symbols[z] for z in uniq]
    cell = atoms.cell.array
    lx, ly, lz = float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])
    pos = atoms.get_positions()
    lines = [
        title,
        "",
        f"{len(atoms)} atoms",
        f"{len(uniq)} atom types",
        "",
        f"0.0 {lx:.16f} xlo xhi",
        f"0.0 {ly:.16f} ylo yhi",
        f"0.0 {lz:.16f} zlo zhi",
        "",
        "Masses",
        "",
    ]
    for i, z in enumerate(uniq, 1):
        lines.append(f"{i} {atomic_masses[z]:.8f}")
    lines += ["", "Atoms # atomic", ""]
    for i, (t, p) in enumerate(zip(types, pos), 1):
        lines.append(f"{i} {t} {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}")
    path.write_text("\n".join(lines) + "\n")
    return symbols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--extxyz",
        type=Path,
        default=Path(__file__).resolve().parent / "water888_2000step.extxyz",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    atoms = read(str(args.extxyz), index=0)
    assert isinstance(atoms, Atoms)
    n = len(atoms)
    if n != NATOMS_EXPECTED:
        raise SystemExit(f"expected {NATOMS_EXPECTED} atoms (512 H2O), got {n}")
    symbols = write_data(atoms, args.out, "H2O 8x8x8 from water888_2000step.extxyz")
    print(f"wrote {args.out} natoms={n} types={symbols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
