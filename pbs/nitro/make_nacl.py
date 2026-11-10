#!/usr/bin/env python3
"""Write data.nacl and structure.xyz for the NaCl rocksalt NxNxN gate cell.

Identical construction to the Wave E gate (a = 5.64 A, 0.05 A gaussian rattle,
seed 0) so energies are comparable. Usage: make_nacl.py N OUTDIR
"""
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    n = int(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    a = 5.64
    na = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], float)
    cl = na + 0.5
    syms, sc = [], []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                o = np.array([ix, iy, iz], float)
                for f in na:
                    syms.append("Na")
                    sc.append((f + o) / n)
                for f in cl:
                    syms.append("Cl")
                    sc.append((f + o) / n)
    cell = np.eye(3) * (a * n)
    pos = np.array(sc) @ cell
    pos = pos + np.random.default_rng(0).normal(0, 0.05, pos.shape)
    lx, ly, lz = float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])
    nat = len(syms)
    if nat != 8 * n ** 3:
        raise SystemExit(f"FAIL: nat={nat} expected {8 * n ** 3}")
    lines = [
        f"NaCl NVT/NPT N={n} a=5.64 rattle0.05 seed0", "", f"{nat} atoms",
        "2 atom types", "", f"0.0 {lx:.16e} xlo xhi", f"0.0 {ly:.16e} ylo yhi",
        f"0.0 {lz:.16e} zlo zhi", "", "Masses", "", "1 22.989769", "2 35.453",
        "", "Atoms # atomic", "",
    ]
    type_of = {"Na": 1, "Cl": 2}
    for i, (s, p) in enumerate(zip(syms, pos), 1):
        lines.append(f"{i} {type_of[s]} {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}")
    (out / "data.nacl").write_text("\n".join(lines) + "\n")
    lat = f"{lx:.16e} 0 0 0 {ly:.16e} 0 0 0 {lz:.16e}"
    xyz = [str(nat), f'Lattice="{lat}" Properties=species:S:1:pos:R:3 pbc="T T T"']
    for s, p in zip(syms, pos):
        xyz.append(f"{s} {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}")
    (out / "structure.xyz").write_text("\n".join(xyz) + "\n")
    print(f"CELL_OK nat={nat}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
