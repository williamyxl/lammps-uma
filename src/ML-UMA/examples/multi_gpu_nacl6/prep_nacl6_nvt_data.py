#!/usr/bin/env python3
"""Bake frozen NaCl6 positions + Maxwell–Boltzmann velocities @ 300 K (seed=0).

Writes ``structures/nacl6_nvt_300K_atomic_metal.data`` (type1=Na, type2=Cl).
Do not re-rattle positions — only velocities are (re)sampled when --force.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from load_geometry import EXPECTED_NATOMS, load_nacl6_fixed

EX = Path(__file__).resolve().parent
DEFAULT_OUT = EX / "structures" / "nacl6_nvt_300K_atomic_metal.data"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature-K", type=float, default=300.0)
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing data file",
    )
    args = ap.parse_args()
    if args.out.is_file() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite")

    atoms = load_nacl6_fixed()
    if len(atoms) != EXPECTED_NATOMS:
        raise SystemExit(f"natoms={len(atoms)} expected {EXPECTED_NATOMS}")
    np.random.seed(int(args.seed))
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=float(args.temperature_K), force_temp=True
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(
        str(args.out),
        atoms,
        format="lammps-data",
        atom_style="atomic",
        units="metal",
        velocities=True,
    )
    # ASE write uses a generic header; stamp provenance.
    text = args.out.read_text()
    text = text.replace(
        text.splitlines()[0],
        "LAMMPS data file atomic style, metal units "
        f"(NaCl6 frozen rattle + MB {args.temperature_K:g}K seed={args.seed})",
        1,
    )
    args.out.write_text(text)
    print(f"wrote {args.out} natoms={len(atoms)} T={args.temperature_K} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
