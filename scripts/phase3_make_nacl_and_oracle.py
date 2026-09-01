#!/usr/bin/env python3
"""Phase-3: write perturbed-NaCl structure files (parity_cli format) + ASE UMA
oracle (energy + per-atom forces) for a list of N.

Structure file format (matches uma-engine/tests/parity_cli.cpp load_structure):
    N
    Z x y z        (N lines)
    c00 c01 c02 c10 c11 c12 c20 c21 c22   (row-major cell)

Oracle: FAIRChemCalculator(MLIPPredictUnit(uma-s-1p2.pt, device=xpu, fp64))
        task omat, on the IDENTICAL perturbed coordinates (hen build_nacl:
        a=5.64, rattle 0.05, rng seed 0). Energy (eV) + forces [N,3] saved.

Env: UMA_CKPT, N_LIST="2,3,4,6,8,10", OUTDIR, ZE_AFFINITY_MASK (1 tile).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HEN_ROOT = Path("/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen")
for p in (HEN_ROOT / "shim", HEN_ROOT / "patches", HEN_ROOT):
    if p.is_dir():
        sys.path.insert(0, str(p))

Z_OF = {"Na": 11, "Cl": 17}


def build_nacl(n: int, rattle: float = 0.05, seed: int = 0):
    from ase import Atoms

    a = 5.64
    na_frac = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    )
    cl_frac = na_frac + 0.5
    symbols, scaled = [], []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                off = np.array([ix, iy, iz], dtype=float)
                for f in na_frac:
                    symbols.append("Na")
                    scaled.append((f + off) / n)
                for f in cl_frac:
                    symbols.append("Cl")
                    scaled.append((f + off) / n)
    cell = np.eye(3) * (a * n)
    atoms = Atoms(symbols=symbols, scaled_positions=scaled, cell=cell, pbc=True)
    if rattle:
        rng = np.random.default_rng(seed)
        atoms.positions += rng.normal(0.0, rattle, size=atoms.positions.shape)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0
    return atoms


def write_structure(atoms, path: Path):
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    z = [Z_OF[s] for s in atoms.get_chemical_symbols()]
    lines = [str(len(atoms))]
    for i in range(len(atoms)):
        lines.append(f"{z[i]} {pos[i,0]:.16e} {pos[i,1]:.16e} {pos[i,2]:.16e}")
    lines.append(" ".join(f"{cell[r,c]:.16e}" for r in range(3) for c in range(3)))
    path.write_text("\n".join(lines) + "\n")


def make_calc(ckpt: Path):
    import torch
    from dataclasses import replace
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
    from fairchem_xpu_parallel import patch_fairchem_xpu_device

    patch_fairchem_xpu_device()
    settings = guess_inference_settings("default")
    settings = replace(settings, base_precision_dtype=torch.float64, tf32=False, compile=False)
    unit = MLIPPredictUnit(str(ckpt), device="xpu", inference_settings=settings)
    for attr in ("model", "module", "_module"):
        mod = getattr(unit, attr, None)
        if mod is not None:
            mod.double()
            break
    return FAIRChemCalculator(unit, task_name=os.environ.get("UMA_TASK", "omat"))


def main() -> int:
    import torch

    ckpt = Path(os.environ.get("UMA_CKPT", str(HEN_ROOT / "uma-cache" / "uma-s-1p2.pt")))
    n_list = [int(x) for x in os.environ.get("N_LIST", "2,3,4,6,8,10").split(",") if x.strip()]
    outdir = Path(os.environ.get("OUTDIR", "./phase3"))
    outdir.mkdir(parents=True, exist_ok=True)

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit("XPU unavailable")

    t0 = time.perf_counter()
    calc = make_calc(ckpt)
    print(f"oracle calc loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    manifest = []
    for n in n_list:
        atoms = build_nacl(n)
        nat = len(atoms)
        sdir = outdir / f"nacl_n{n}"
        sdir.mkdir(parents=True, exist_ok=True)
        write_structure(atoms, sdir / "structure.txt")

        atoms.calc = calc
        t = time.perf_counter()
        try:
            e = float(atoms.get_potential_energy())
            f = np.asarray(atoms.get_forces(), dtype=np.float64)
            torch.xpu.synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"N={n} nat={nat} ORACLE FAIL: {type(exc).__name__}: {exc}", flush=True)
            manifest.append({"n": n, "natoms": nat, "oracle_ok": False,
                             "error": f"{type(exc).__name__}: {exc}"[:200]})
            (outdir / "oracle_manifest.json").write_text(json.dumps(manifest, indent=2))
            continue
        np.save(sdir / "oracle_forces.npy", f)
        (sdir / "oracle_energy.txt").write_text(f"{e:.12f}\n")
        dt = time.perf_counter() - t
        fmax = float(np.abs(f).max())
        print(f"N={n} nat={nat} E={e:.10f} fmax={fmax:.6f} ({dt:.1f}s)", flush=True)
        manifest.append({"n": n, "natoms": nat, "oracle_ok": True,
                         "energy_eV": e, "fmax": fmax, "elapsed_s": dt,
                         "structure": str(sdir / "structure.txt"),
                         "forces_npy": str(sdir / "oracle_forces.npy")})
        (outdir / "oracle_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"wrote {outdir}/oracle_manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
