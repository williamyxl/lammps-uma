#!/usr/bin/env python3
"""Write a LAMMPS data file for perturbed rocksalt NaCl NxNxN.

BIT-IDENTICAL geometry + atom ordering to the ASE-GP oracle builder
(hen/scripts/fxpu_1node_ef_atoms.py:build_nacl_rocksalt): nested ix,iy,iz loops,
Na (4) then Cl (4) per conventional cell, a=5.64, rattle 0.05, seed 0. Atom id
i+1 corresponds to oracle row i, so parity_vs_asegp.py (sorts LAMMPS by id) lines
up atom-for-atom with forces_w12.npy.

Usage:  python make_data.py --n 32 --out data.nacl_n32
"""
import argparse
import numpy as np

MASS = {"Na": 22.989769, "Cl": 35.453}


def build(n, a=5.64, rattle=0.05, seed=0):
    na = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], float)
    cl = na + 0.5
    syms, sc = [], []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                o = np.array([ix, iy, iz], float)
                for f in na:
                    syms.append("Na"); sc.append((f + o) / n)
                for f in cl:
                    syms.append("Cl"); sc.append((f + o) / n)
    cell = np.eye(3) * (a * n)
    pos = np.array(sc) @ cell
    # oracle uses rattle * rng.standard_normal(shape); normal(0, rattle) is the
    # SAME distribution AND same draws for a given seed via default_rng.
    pos = pos + rattle * np.random.default_rng(seed).standard_normal(pos.shape)
    return syms, pos, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default="data.nacl")
    args = ap.parse_args()
    syms, pos, cell = build(args.n)
    L = cell[0, 0]
    lines = [f"NaCl {args.n}x{args.n}x{args.n} perturbed (a=5.64, rattle 0.05, seed 0)", "",
             f"{len(syms)} atoms", "2 atom types", "",
             f"0.0 {L:.16e} xlo xhi", f"0.0 {L:.16e} ylo yhi",
             f"0.0 {L:.16e} zlo zhi", "", "Masses", "",
             f"1 {MASS['Na']}", f"2 {MASS['Cl']}", "", "Atoms # atomic", ""]
    for i, s in enumerate(syms):
        t = 1 if s == "Na" else 2
        lines.append(f"{i+1} {t} {pos[i,0]:.16e} {pos[i,1]:.16e} {pos[i,2]:.16e}")
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(syms)} atoms (N={args.n})")


if __name__ == "__main__":
    main()
