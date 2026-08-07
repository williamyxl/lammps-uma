#!/usr/bin/env python3
"""Freeze rattled geometries for the Delta 4-path parity suite (CPU-only).

Systems
-------
- nacl / nacl4 / nacl5 / nacl6 / nacl7 / nacl8 : NaCl rocksalt 3/4/5/6/7/8
- al   / al5 / al8             : Al FCC 3/5/8
- si   / si4 / si7 / si8       : Si diamond 3/4/7/8
- (large ASE-safe ensemble: nacl4 + al5 + si4 ≈ 500 atoms)

Protocol (shared by all evaluators)
-----------------------------------
1. Build perfect crystal (ASE ``bulk``) at equilibrium lattice constant.
2. Uniform-box rattle: each Cartesian ~ Unif[-0.10, +0.10] Å, PCG64(seed=0).
3. Wrap into cell; write ``structures/structure_<name>_rattle.npz``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk

OUT = Path(__file__).resolve().parent
STRUCT = OUT / "structures"

PERTURB_DELTA_A = 0.10
PERTURB_SEED = 0
PERTURB_MODE = "uniform_box"

SYSTEMS = (
    "nacl",
    "al",
    "si",
    "si4",
    "nacl4",
    "nacl5",
    "nacl6",
    "al5",
    "nacl7",
    "si7",
    "nacl8",
    "al8",
    "si8",
)

_BUILDERS = {
    "nacl": ("NaCl", "rocksalt", 5.64, (3, 3, 3)),
    "nacl4": ("NaCl", "rocksalt", 5.64, (4, 4, 4)),
    "nacl5": ("NaCl", "rocksalt", 5.64, (5, 5, 5)),
    "nacl6": ("NaCl", "rocksalt", 5.64, (6, 6, 6)),
    "nacl7": ("NaCl", "rocksalt", 5.64, (7, 7, 7)),
    "nacl8": ("NaCl", "rocksalt", 5.64, (8, 8, 8)),
    "al": ("Al", "fcc", 4.05, (3, 3, 3)),
    "al5": ("Al", "fcc", 4.05, (5, 5, 5)),
    "al8": ("Al", "fcc", 4.05, (8, 8, 8)),
    "si": ("Si", "diamond", 5.43, (3, 3, 3)),
    "si4": ("Si", "diamond", 5.43, (4, 4, 4)),
    "si7": ("Si", "diamond", 5.43, (7, 7, 7)),
    "si8": ("Si", "diamond", 5.43, (8, 8, 8)),
}


def build_ideal(name: str) -> Atoms:
    if name not in _BUILDERS:
        raise ValueError(name)
    element, crystal, a, repeat = _BUILDERS[name]
    return bulk(element, crystal, a=a, cubic=True) * repeat


def perturb(atoms: Atoms) -> dict:
    rng = np.random.Generator(np.random.PCG64(PERTURB_SEED))
    disp = rng.uniform(-PERTURB_DELTA_A, PERTURB_DELTA_A, size=atoms.positions.shape)
    atoms.positions = atoms.positions + disp
    atoms.wrap()
    return {
        "mode": PERTURB_MODE,
        "delta_A": float(PERTURB_DELTA_A),
        "seed": int(PERTURB_SEED),
        "rng": "numpy.random.Generator(PCG64(seed))",
        "distribution": f"Unif[-{PERTURB_DELTA_A}, +{PERTURB_DELTA_A}] Å per Cartesian",
        "wrap": True,
        "disp_max_abs_A": float(np.max(np.abs(disp))),
        "disp_rms_A": float(np.sqrt(np.mean(disp**2))),
    }


def freeze(name: str) -> dict:
    STRUCT.mkdir(parents=True, exist_ok=True)
    atoms = build_ideal(name)
    meta = perturb(atoms)
    npz = STRUCT / f"structure_{name}_rattle.npz"
    np.savez(
        npz,
        numbers=atoms.get_atomic_numbers().astype(np.int32),
        positions=atoms.get_positions().astype(np.float64),
        cell=atoms.cell.array.astype(np.float64),
        perturb_mode=np.array(meta["mode"]),
        perturb_delta_A=np.array(meta["delta_A"]),
        perturb_seed=np.array(meta["seed"]),
    )
    info = {
        "name": name,
        "natoms": len(atoms),
        "npz": str(npz.relative_to(OUT)),
        "cell_diag_A": [float(x) for x in np.diag(atoms.cell.array)],
        "perturbation": meta,
    }
    print(
        f"froze {npz.name}: natoms={len(atoms)}  "
        f"δ={meta['delta_A']} Å  seed={meta['seed']}  "
        f"disp_rms={meta['disp_rms_A']:.4f} Å",
        flush=True,
    )
    return info


def main() -> int:
    names = sys.argv[1:] or list(SYSTEMS)
    for n in names:
        if n not in _BUILDERS:
            raise SystemExit(f"unknown system {n!r}; choose from {list(_BUILDERS)}")
    out = STRUCT / "manifest.json"
    systems = {}
    if out.is_file():
        try:
            systems.update(json.loads(out.read_text()).get("systems") or {})
        except json.JSONDecodeError:
            pass
    for n in names:
        systems[n] = freeze(n)
    out.write_text(json.dumps({"systems": systems}, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
