#!/usr/bin/env python3
"""Build NaCl rocksalt N×N×N (8·N³ atoms) for multi-node Phase G.

Examples:
  python build_nacl_n.py --n 8 --ideal --out /tmp/nacl8_ideal.extxyz
  python build_nacl_n.py --n 10 --rattle --out structures/nacl10_rattle_fixed.extxyz --freeze-manifest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write

A_LATTICE = 5.64
PERTURB_DELTA_A = 0.10
PERTURB_SEED = 0


def build_ideal(n: int) -> Atoms:
    if n < 1:
        raise ValueError(n)
    return bulk("NaCl", "rocksalt", a=A_LATTICE, cubic=True) * (n, n, n)


def rattle(atoms: Atoms, *, delta: float = PERTURB_DELTA_A, seed: int = PERTURB_SEED) -> dict:
    rng = np.random.Generator(np.random.PCG64(seed))
    disp = rng.uniform(-delta, delta, size=atoms.positions.shape)
    atoms.positions = atoms.positions + disp
    atoms.wrap()
    return {
        "mode": "uniform_box",
        "delta_A": float(delta),
        "seed": int(seed),
        "disp_max_abs_A": float(np.max(np.abs(disp))),
        "disp_rms_A": float(np.sqrt(np.mean(disp**2))),
        "note": "Frozen after write; do not regenerate.",
    }


def write_extxyz(atoms: Atoms, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Match nacl6_rattle_fixed: high-precision coords
    write(path, atoms, format="extxyz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="supercell repeat N (natoms=8*N^3)")
    ap.add_argument("--out", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ideal", action="store_true")
    g.add_argument("--rattle", action="store_true")
    ap.add_argument("--freeze-manifest", action="store_true")
    ap.add_argument("--delta", type=float, default=PERTURB_DELTA_A)
    ap.add_argument("--seed", type=int, default=PERTURB_SEED)
    args = ap.parse_args()

    atoms = build_ideal(args.n)
    expected = 8 * (args.n**3)
    if len(atoms) != expected:
        raise SystemExit(f"natoms={len(atoms)} != 8*N^3={expected}")

    pert = None
    if args.rattle:
        pert = rattle(atoms, delta=args.delta, seed=args.seed)

    write_extxyz(atoms, args.out)
    print(f"wrote {args.out}  N={args.n}  natoms={len(atoms)}  ideal={args.ideal}")

    if args.freeze_manifest:
        man = {
            "n": args.n,
            "natoms": len(atoms),
            "path": str(args.out),
            "cell": atoms.cell.array.tolist(),
            "a_lattice_A": A_LATTICE,
            "perturbation": pert,
            "coordinate_format": "ase.io.write extxyz",
            "note": "Immutable for multi-node campaign; never re-rattle.",
        }
        mpath = args.out.with_suffix(".manifest.json")
        if args.out.suffix == ".extxyz":
            mpath = Path(str(args.out)[: -len(".extxyz")] + ".manifest.json")
        mpath.write_text(json.dumps(man, indent=2) + "\n")
        print(f"wrote {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
