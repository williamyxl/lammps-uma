#!/usr/bin/env python3
"""Build the canonical NaCl 8x8x8 (4096-atom) test geometry.

Each atom is displaced by a random vector whose MAGNITUDE is uniform in
[0.05, 0.10] Angstrom and whose DIRECTION is isotropic on the unit sphere (so
no atom sits at a near-perfect-crystal site with vanishing force). Written to a
full-precision LAMMPS data file (repr() floats, ~17 significant digits) that is
the single authoritative geometry for all NaCl-8x8x8 tests.

Deterministic (seed=0) so the geometry is reproducible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk

A_LATTICE = 5.64          # NaCl conventional lattice constant (A), matches campaign
NREP = 8                  # 8x8x8 conventional cells -> 4096 atoms
DMIN, DMAX = 0.05, 0.10   # perturbation magnitude range (A)
SEED = 0
MASS = {"Na": 22.98976928, "Cl": 35.45}


def build():
    atoms = bulk("NaCl", "rocksalt", a=A_LATTICE, cubic=True).repeat((NREP, NREP, NREP))
    atoms.pbc = True
    n = len(atoms)
    rng = np.random.default_rng(SEED)
    # isotropic directions
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # magnitudes uniform in [DMIN, DMAX]
    mag = rng.uniform(DMIN, DMAX, size=(n, 1))
    disp = v * mag
    atoms.set_positions(atoms.get_positions() + disp)
    # wrap back into the cell so all coords are in [0, L)
    atoms.wrap()
    return atoms, disp


def write_lammps_data(atoms, path: Path):
    """Full-precision LAMMPS 'atomic' data file (metal units).
    Types: 1=Na, 2=Cl (matches the campaign convention)."""
    sym_to_type = {"Na": 1, "Cl": 2}
    cell = np.array(atoms.cell, dtype=float)
    # orthorhombic box assumed (rocksalt supercell is cubic)
    lx, ly, lz = float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])
    syms = atoms.get_chemical_symbols()
    pos = np.asarray(atoms.get_positions(), dtype=float)
    lines = []
    lines.append("NaCl 8x8x8 (4096) rocksalt a=5.64, per-atom rand disp |d| in [0.05,0.10] A, seed=0")
    lines.append("")
    lines.append(f"{len(atoms)} atoms")
    lines.append("2 atom types")
    lines.append("")
    lines.append(f"0.0 {lx!r} xlo xhi")
    lines.append(f"0.0 {ly!r} ylo yhi")
    lines.append(f"0.0 {lz!r} zlo zhi")
    lines.append("")
    lines.append("Masses")
    lines.append("")
    lines.append(f"1 {MASS['Na']!r} # Na")
    lines.append(f"2 {MASS['Cl']!r} # Cl")
    lines.append("")
    lines.append("Atoms # atomic")
    lines.append("")
    for i, (s, p) in enumerate(zip(syms, pos), start=1):
        t = sym_to_type[s]
        lines.append(f"{i} {t} {float(p[0])!r} {float(p[1])!r} {float(p[2])!r}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "structures"
                    / "nacl8x8x8_perturbed.data")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    atoms, disp = build()
    write_lammps_data(atoms, args.out)

    dmag = np.linalg.norm(disp, axis=1)
    print(f"natoms          = {len(atoms)}")
    print(f"cell (A)        = {np.diag(np.array(atoms.cell))}")
    print(f"disp |d| min/max= {dmag.min():.6f} / {dmag.max():.6f} A  (target 0.05-0.10)")
    print(f"disp |d| mean   = {dmag.mean():.6f} A")
    print(f"wrote           = {args.out}")

    # round-trip precision check
    from ase.io import read
    a2 = read(str(args.out), format="lammps-data", atom_style="atomic", units="metal")
    dmax = float(np.abs(a2.get_positions() - atoms.get_positions()).max())
    print(f"data round-trip max|dx| = {dmax:.3e} A  (should be ~0 / <1e-12)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
