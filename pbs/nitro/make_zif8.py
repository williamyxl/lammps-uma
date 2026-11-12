#!/usr/bin/env python3
"""Write data.zif8 and structure.xyz for the ZIF-8 NxNxN gate cell.

Same construction idea as make_nacl.py (gaussian rattle 0.05 A, seed 0), applied
to ZIF-8. The structure comes from the P1 4x4x4 CIF in the read-only
rosen_uma_ZIF project: reduce to the 276-atom conventional cell (a = 16.99 A),
replicate N x N x N. N = 3 gives 7452 atoms and a 50.97 A cube.

Usage: make_zif8.py N OUTDIR [RATTLE_SIGMA]
  RATTLE_SIGMA default 0.05 (A). Use 0 for the unrattled lattice.
Env: ZIF8_CIF overrides the CIF path. Needs numpy only (no torch).
Type order and masses match the rosen runs: 1 H, 2 C, 3 N, 4 Zn.
"""
import os
import sys
from pathlib import Path

import numpy as np

CIF_DEFAULT = (
    "/lus/flare/projects/MatSciAI/xiaoliyan/workdir/rosen_uma_ZIF/"
    "ZIF_pyrolysis/ZIF-8_4x4x4.cif"
)
TYPE_ORDER = ["H", "C", "N", "Zn"]
MASSES = {"H": 1.00784, "C": 12.011, "N": 14.0067, "Zn": 65.38}
ATOMS_PER_CELL = 276
CELL_A = 16.99  # conventional cell edge, used only to infer the 4x4x4 repeat


def _f(tok):
    return float(tok.split("(")[0])


def read_cif(path):
    a = b = c = None
    angles = {"alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    in_atoms, fields, rows = False, [], []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            low = line.lower()
            if low.startswith("_cell_length_a"):
                a = _f(line.split()[1])
            elif low.startswith("_cell_length_b"):
                b = _f(line.split()[1])
            elif low.startswith("_cell_length_c"):
                c = _f(line.split()[1])
            elif low.startswith("_cell_angle_alpha"):
                angles["alpha"] = _f(line.split()[1])
            elif low.startswith("_cell_angle_beta"):
                angles["beta"] = _f(line.split()[1])
            elif low.startswith("_cell_angle_gamma"):
                angles["gamma"] = _f(line.split()[1])
            elif line.startswith("loop_"):
                in_atoms, fields = False, []
            elif line.startswith("_atom_site_"):
                in_atoms = True
                fields.append(low.split()[0])
            elif in_atoms and line.startswith("_"):
                in_atoms = False
            elif in_atoms:
                rows.append(line.split())
    if a is None or any(abs(v - 90.0) > 1e-6 for v in angles.values()):
        raise SystemExit("FAIL: CIF missing or non-orthogonal cell")
    i_el = fields.index("_atom_site_type_symbol")
    i_x = fields.index("_atom_site_fract_x")
    i_y = fields.index("_atom_site_fract_y")
    i_z = fields.index("_atom_site_fract_z")
    syms, pos = [], []
    for p in rows:
        syms.append(p[i_el])
        pos.append([_f(p[i_x]) * a, _f(p[i_y]) * b, _f(p[i_z]) * c])
    return syms, pos, (a, b, c)


def wrap(x, length):
    t = x / length
    t -= int(t)
    if t < 0:
        t += 1.0
    if t >= 1.0 - 1e-12:
        t = 0.0
    return t * length


def conventional_cell(syms, pos, box, nrep):
    a, b, c = box[0] / nrep, box[1] / nrep, box[2] / nrep
    seen = {}
    for s, p in zip(syms, pos):
        x, y, z = wrap(p[0], a), wrap(p[1], b), wrap(p[2], c)
        seen[(s, round(x * 1e6), round(y * 1e6), round(z * 1e6))] = (s, [x, y, z])
    return [v[0] for v in seen.values()], [v[1] for v in seen.values()], (a, b, c)


def main() -> int:
    n = int(sys.argv[1])
    out = Path(sys.argv[2])
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    out.mkdir(parents=True, exist_ok=True)
    cif = os.environ.get("ZIF8_CIF", CIF_DEFAULT)
    if not os.path.isfile(cif):
        raise SystemExit(f"FAIL: CIF missing: {cif}")
    syms, pos, box = read_cif(cif)
    nrep = max(1, int(round(box[0] / CELL_A)))
    syms, pos, cell = conventional_cell(syms, pos, box, nrep)
    if len(syms) != ATOMS_PER_CELL:
        raise SystemExit(f"FAIL: conventional cell has {len(syms)} atoms, expected {ATOMS_PER_CELL}")
    unknown = sorted(set(syms) - set(TYPE_ORDER))
    if unknown:
        raise SystemExit(f"FAIL: unexpected elements {unknown}")
    rs, rp = [], []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                off = (ix * cell[0], iy * cell[1], iz * cell[2])
                for s, p in zip(syms, pos):
                    rs.append(s)
                    rp.append([p[0] + off[0], p[1] + off[1], p[2] + off[2]])
    syms = rs
    pos = np.array(rp, float)
    if sigma > 0:
        pos = pos + np.random.default_rng(0).normal(0, sigma, pos.shape)
    lx, ly, lz = n * cell[0], n * cell[1], n * cell[2]
    nat = len(syms)
    if nat != ATOMS_PER_CELL * n ** 3:
        raise SystemExit(f"FAIL: nat={nat} expected {ATOMS_PER_CELL * n ** 3}")
    tmap = {s: i + 1 for i, s in enumerate(TYPE_ORDER)}
    lines = [
        f"ZIF-8 NVT/NPT N={n} cif={os.path.basename(cif)} rattle{sigma:g} seed0",
        "", f"{nat} atoms", f"{len(TYPE_ORDER)} atom types", "",
        f"0.0 {lx:.16e} xlo xhi", f"0.0 {ly:.16e} ylo yhi", f"0.0 {lz:.16e} zlo zhi",
        "", "Masses", "",
    ]
    for i, s in enumerate(TYPE_ORDER, 1):
        lines.append(f"{i} {MASSES[s]:.8f}")
    lines += ["", "Atoms # atomic", ""]
    for i, (s, p) in enumerate(zip(syms, pos), 1):
        lines.append(f"{i} {tmap[s]} {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}")
    (out / "data.zif8").write_text("\n".join(lines) + "\n")
    lat = f"{lx:.16e} 0 0 0 {ly:.16e} 0 0 0 {lz:.16e}"
    xyz = [str(nat), f'Lattice="{lat}" Properties=species:S:1:pos:R:3 pbc="T T T"']
    for s, p in zip(syms, pos):
        xyz.append(f"{s} {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}")
    (out / "structure.xyz").write_text("\n".join(xyz) + "\n")
    counts = {s: syms.count(s) for s in TYPE_ORDER}
    print(f"CELL_OK nat={nat} lx={lx:.6f} counts={counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
