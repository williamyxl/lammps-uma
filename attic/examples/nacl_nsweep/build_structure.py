#!/usr/bin/env python3
"""Build a frozen NaCl NxNxN rocksalt cell (8*N^3 atoms) for the V6 N-sweep.

Reproduces the multi_gpu_nacl6 convention exactly (a=5.64, uniform rattle
delta=0.1 A, seed=0), so N=6 regenerates the frozen nacl6 geometry to ~5e-11 A.

Writes:
  structures/nacl{N}_rattle.extxyz                  (positions, for export/oracle)
  structures/nacl{N}_nvt_300K_atomic_metal.data     (+ MB velocities @300K, seed=0)
  structures/nacl{N}.manifest.json
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
PERTURB_DELTA_A = 0.10
PERTURB_SEED = 0
TEMP_K = 300.0


def build(n: int, *, delta: float = PERTURB_DELTA_A, seed: int = PERTURB_SEED):
    atoms = bulk("NaCl", "rocksalt", a=A_LATTICE, cubic=True) * (n, n, n)
    rng = np.random.Generator(np.random.PCG64(seed))
    disp = rng.uniform(-delta, delta, size=atoms.positions.shape)
    atoms.positions = atoms.positions + disp
    atoms.wrap()
    return atoms, {
        "mode": "uniform_box",
        "delta_A": float(delta),
        "seed": int(seed),
        "disp_max_abs_A": float(np.max(np.abs(disp))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--outdir", type=Path, default=EX / "structures")
    ap.add_argument("--temperature-K", type=float, default=TEMP_K)
    ap.add_argument("--seed", type=int, default=PERTURB_SEED)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    n = a.n
    out = a.outdir
    out.mkdir(parents=True, exist_ok=True)
    xyz = out / f"nacl{n}_rattle.extxyz"
    data = out / f"nacl{n}_nvt_300K_atomic_metal.data"
    man = out / f"nacl{n}.manifest.json"
    if data.is_file() and not a.force:
        print(f"exists (use --force): {data}")
        return 0

    atoms, pert = build(n, seed=a.seed)
    natoms = len(atoms)
    if natoms != 8 * n**3:
        raise SystemExit(f"natoms={natoms} != 8*N^3={8 * n**3}")
    # 12 significant digits, matching the frozen nacl6 coordinate_format; the
    # extxyz default loses precision and shifts the parity reference.
    write(str(xyz), atoms, format="extxyz", write_info=True,
          columns=["symbols", "positions"], plain=False)

    # Velocities are sampled onto a copy so the .extxyz stays position-only,
    # matching prep_nacl6_nvt_data.py (positions frozen, velocities regenerable).
    mb = atoms.copy()
    np.random.seed(int(a.seed))
    MaxwellBoltzmannDistribution(mb, temperature_K=float(a.temperature_K),
                                 force_temp=True)
    # velocities=True: without it LAMMPS starts at 0 K and NVT has to reheat.
    # masses=True + specorder pins type1=Na type2=Cl; without the Masses section
    # readers fall back to H/He and the run is silently the wrong material.
    write(str(data), mb, format="lammps-data", atom_style="atomic",
          units="metal", velocities=True, masses=True, specorder=["Na", "Cl"])

    man.write_text(json.dumps({
        "n": n,
        "natoms": natoms,
        "a_lattice_A": A_LATTICE,
        "cell_A": float(n * A_LATTICE),
        "cell": atoms.cell.array.tolist(),
        "perturbation": pert,
        "velocities": f"MaxwellBoltzmann {a.temperature_K}K force_temp=True seed={a.seed}",
        "types": {"1": "Na", "2": "Cl"},
        "extxyz": str(xyz),
        "data": str(data),
        "note": "N=6 reproduces frozen nacl6_rattle_fixed to ~5e-11 A.",
    }, indent=2) + "\n")
    print(f"N={n} natoms={natoms} cell={n * A_LATTICE:.2f} A -> {data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
