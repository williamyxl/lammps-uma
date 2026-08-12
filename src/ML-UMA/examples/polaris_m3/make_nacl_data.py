#!/usr/bin/env python3
"""Emit a full-precision LAMMPS data file for NaCl N^3 (conventional cells) with
per-atom isotropic random displacement |d| in [0.05,0.10] A (seed 0). Reuses the
canonical builder's scheme; parametric N for capacity tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk

A_LATTICE = 5.64
MASS = {"Na": 22.98976928, "Cl": 35.45}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="conventional cells per side")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    atoms = bulk("NaCl", "rocksalt", a=A_LATTICE, cubic=True).repeat((args.n,) * 3)
    atoms.pbc = True
    n = len(atoms)
    rng = np.random.default_rng(args.seed)
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    mag = rng.uniform(0.05, 0.10, size=(n, 1))
    atoms.set_positions(atoms.get_positions() + v * mag)
    atoms.wrap()

    cell = np.array(atoms.cell, dtype=float)
    lx, ly, lz = float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])
    sym_to_type = {"Na": 1, "Cl": 2}
    syms = atoms.get_chemical_symbols()
    pos = np.asarray(atoms.get_positions(), dtype=float)
    L = [f"NaCl {args.n}^3 = {n} atoms, |d| in [0.05,0.10] A, seed={args.seed}", "",
         f"{n} atoms", "2 atom types", "",
         f"0.0 {lx!r} xlo xhi", f"0.0 {ly!r} ylo yhi", f"0.0 {lz!r} zlo zhi", "",
         "Masses", "", f"1 {MASS['Na']!r} # Na", f"2 {MASS['Cl']!r} # Cl", "",
         "Atoms # atomic", ""]
    for i, (s, p) in enumerate(zip(syms, pos), start=1):
        L.append(f"{i} {sym_to_type[s]} {float(p[0])!r} {float(p[1])!r} {float(p[2])!r}")
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}  N={n} box={lx:.1f}A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
