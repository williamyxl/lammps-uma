#!/usr/bin/env python3
"""Perturb the 8-atom NaCl unit cell, THEN replicate 9x9x9 (5832 atoms).

Note the order: the perturbation is applied to the unit cell and the *same*
displaced motif is replicated. The result is therefore exactly periodic with
the 5.64 A unit cell, not the 50.76 A supercell. Two consequences:

  1. Forces must repeat every 8 atoms (atom i and i+8k are symmetry-identical).
     That is a free, strong internal correctness check -- exploited by
     --check-periodicity and by the per-atom force gate downstream.
  2. This is NOT the campaign's nacl6/nacl9 convention, where every atom in the
     supercell is independently rattled. Do not compare energies against those
     oracles; this is its own structure.

Displacement matches the campaign convention (uniform in [-0.1, 0.1] A per
Cartesian component, seed 0) so the magnitude is familiar; --mode sphere gives
a fixed 0.1 A displacement magnitude instead.

Writes structures/nacl9rep_nvt_300K_atomic_metal.data (+ .extxyz, manifest),
with a Masses section (type1=Na type2=Cl) and 300 K Maxwell-Boltzmann
velocities, matching the existing LAMMPS data files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

EX = Path(__file__).resolve().parent
A_LATTICE = 5.64
DELTA = 0.10
SEED = 0
TEMP_K = 300.0


def perturb_unit_cell(delta: float, seed: int, mode: str):
    """Displace each of the 8 unit-cell atoms, then report the displacements."""
    u = bulk("NaCl", "rocksalt", a=A_LATTICE, cubic=True)
    if len(u) != 8:
        raise SystemExit(f"expected 8-atom cubic cell, got {len(u)}")
    rng = np.random.Generator(np.random.PCG64(seed))
    if mode == "box":
        disp = rng.uniform(-delta, delta, size=(len(u), 3))
    else:  # sphere: fixed |d| = delta, random direction
        v = rng.normal(size=(len(u), 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        disp = v * delta
    u.positions = u.positions + disp
    return u, disp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mode", choices=("box", "sphere"), default="box",
                    help="box: uniform +/-delta per component (campaign "
                         "convention); sphere: fixed |d|=delta")
    ap.add_argument("--temperature-K", type=float, default=TEMP_K)
    ap.add_argument("--outdir", type=Path, default=EX / "structures")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    n = a.n
    out = a.outdir
    out.mkdir(parents=True, exist_ok=True)
    tag = f"nacl{n}rep"
    data = out / f"{tag}_nvt_300K_atomic_metal.data"
    xyz = out / f"{tag}_rattle.extxyz"
    man = out / f"{tag}.manifest.json"
    if data.is_file() and not a.force:
        print(f"exists (use --force): {data}")
        return 0

    # 1) perturb the unit cell, 2) replicate
    unit, disp = perturb_unit_cell(a.delta, a.seed, a.mode)
    unit.wrap()
    super_cell = unit * (n, n, n)
    natoms = len(super_cell)
    if natoms != 8 * n**3:
        raise SystemExit(f"natoms={natoms} != {8 * n**3}")
    super_cell.wrap()

    write(str(xyz), super_cell, format="extxyz")

    mb = super_cell.copy()
    np.random.seed(int(a.seed))
    MaxwellBoltzmannDistribution(mb, temperature_K=float(a.temperature_K),
                                 force_temp=True)
    # masses=True + specorder pins type1=Na type2=Cl; without the Masses section
    # readers fall back to H/He. velocities=True so LAMMPS starts at 300 K
    # instead of 0 K and reheating.
    write(str(data), mb, format="lammps-data", atom_style="atomic",
          units="metal", velocities=True, masses=True, specorder=["Na", "Cl"])

    man.write_text(json.dumps({
        "tag": tag,
        "n": n,
        "natoms": natoms,
        "a_lattice_A": A_LATTICE,
        "cell_A": float(n * A_LATTICE),
        "construction": "perturb 8-atom unit cell FIRST, then replicate n^3",
        "perturbation": {
            "mode": a.mode,
            "delta_A": a.delta,
            "seed": a.seed,
            "applied_to": "unit cell (8 atoms), then replicated",
            "disp_max_abs_A": float(np.abs(disp).max()),
            "disp_rms_A": float(np.sqrt((disp**2).mean())),
            "unit_cell_displacements": disp.tolist(),
        },
        "periodicity_note": (
            "Structure is exactly periodic with the 5.64 A unit cell, so forces "
            "repeat every 8 atoms. Not comparable to the independently-rattled "
            "nacl6 oracle."),
        "velocities": f"MaxwellBoltzmann {a.temperature_K}K force_temp=True seed={a.seed}",
        "types": {"1": "Na", "2": "Cl"},
        "data": str(data),
        "extxyz": str(xyz),
    }, indent=2) + "\n")

    print(f"{tag}: natoms={natoms} cell={n * A_LATTICE:.2f} A")
    print(f"  unit-cell disp |d|max={np.abs(disp).max():.4f} A "
          f"rms={np.sqrt((disp**2).mean()):.4f} A  mode={a.mode}")
    print(f"  wrote {data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
