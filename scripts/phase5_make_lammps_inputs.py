#!/usr/bin/env python3
"""Phase-5: write a LAMMPS data file + NVT input for perturbed NaCl NxNxN,
identical coordinates to the ASE oracle (hen build: a=5.64, rattle 0.05, seed 0).

Writes <outdir>/nacl_n<N>/{data.nacl, in.nvt}. The in.nvt does `run 0` with a
step-0 force dump (for ASE parity) then `run 10` NVT@300K.

Env: N (default 16), OUTDIR, ART (uma artifact dir for metadata/checkpoint path).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

Z = {"Na": 11, "Cl": 17}
MASS = {"Na": 22.989769, "Cl": 35.453}


def build_nacl(n, rattle=0.05, seed=0):
    a = 5.64
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
    rng = np.random.default_rng(seed)
    pos = pos + rng.normal(0, rattle, pos.shape)
    return syms, pos, cell


def write_data(path, syms, pos, cell):
    L = cell[0, 0]
    lines = ["NaCl perturbed (hen build a=5.64 rattle0.05 seed0)", "",
             f"{len(syms)} atoms", "2 atom types", "",
             f"0.0 {L:.16e} xlo xhi", f"0.0 {L:.16e} ylo yhi",
             f"0.0 {L:.16e} zlo zhi", "", "Masses", "",
             f"1 {MASS['Na']}", f"2 {MASS['Cl']}", "", "Atoms # atomic", ""]
    for i, s in enumerate(syms):
        t = 1 if s == "Na" else 2
        lines.append(f"{i+1} {t} {pos[i,0]:.16e} {pos[i,1]:.16e} {pos[i,2]:.16e}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_input(path, art, ndump):
    txt = f"""# UMA FP64 NVT@300K, perturbed NaCl. pair_style uma (eager, devices 1).
units           metal
atom_style      atomic
boundary        p p p
newton          off
read_data       data.nacl

mass            1 22.989769
mass            2 35.453

pair_style      uma precision double devices 1
pair_coeff      * * {art} Na Cl
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

thermo          1
thermo_style    custom step temp pe etotal fmax
thermo_modify   norm no format float %.16e

# --- step 0: single point + per-atom force dump for ASE parity ---
dump            f0 all custom 1 forces_step0.dump id type x y z fx fy fz
dump_modify     f0 sort id format float %.16e
run             0
undump          f0

# --- 10-step NVT @ 300 K ---
velocity        all create 300.0 4928459 mom yes rot yes dist gaussian
fix             1 all nvt temp 300.0 300.0 0.1
timestep        0.001
run             10
"""
    Path(path).write_text(txt)


def main():
    n = int(os.environ.get("N", "16"))
    outdir = Path(os.environ.get("OUTDIR", "./phase5"))
    art = os.environ.get("ART",
        "/lus/flare/projects/MatSciAI/xiaoliyan/workdir/lammps-uma/scripts/out/phase3b/traced_mergemole_xpu")
    d = outdir / f"nacl_n{n}"
    d.mkdir(parents=True, exist_ok=True)
    syms, pos, cell = build_nacl(n)
    write_data(d / "data.nacl", syms, pos, cell)
    write_input(d / "in.nvt", art, len(syms))
    # also save numpy coords for the oracle
    np.save(d / "positions.npy", pos)
    np.save(d / "cell.npy", cell)
    (d / "symbols.txt").write_text("\n".join(syms))
    print(f"wrote {d}/data.nacl in.nvt ({len(syms)} atoms, N={n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
