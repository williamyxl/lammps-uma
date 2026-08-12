#!/usr/bin/env python3
"""Phase-P0 ASE + FairChem FP64 ground truth (single point E+F, NVT timing).

Runs one system (nacl666 | water888) with the FairChem UMA calculator in FP64,
task=omat, on the exact geometry LAMMPS reads. Writes <sys>_ase.json (energy,
timing) and <sys>_ase.npz (per-atom forces + energy). This is the oracle P0
parity is measured against.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.io import read
from ase.md.nose_hoover_chain import NoseHooverChainNVT

from p0_common import (
    ART_F64,
    DEFAULT_CKPT,
    ENGINE,
    NVT_STEPS,
    NVT_TEMP_K,
    SYSTEMS,
    TASK,
    atoms_from_data,
    results_dir,
)

sys.path.insert(0, str(ENGINE / "python"))
from common import inference_settings_with_dtype  # noqa: E402


def fp64_settings():
    settings = inference_settings_with_dtype("float64")
    settings.external_graph_gen = True
    settings.activation_checkpointing = False
    # Multi-composition parity default: general + no MOLE merge.
    settings.execution_mode = os.environ.get("FAIRCHEM_EXECUTION_MODE", "general")
    raw = os.environ.get("FAIRCHEM_MERGE_MOLE", "0").strip().lower()
    settings.merge_mole = raw in ("1", "true", "yes", "on")
    return settings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("system", choices=sorted(SYSTEMS))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--nsteps", type=int, default=NVT_STEPS)
    ap.add_argument("--recipe", choices=["general", "fastmerge"], default="general",
                    help="general = task-parity export; fastmerge = umas_fast_pytorch+merge_mole "
                         "(matches the multi-GPU MP shards)")
    args = ap.parse_args()

    # Recipe selects the FairChem execution mode / MOLE merge, tagged into the
    # output name so each LAMMPS path compares to its matching-recipe oracle.
    if args.recipe == "fastmerge":
        os.environ["FAIRCHEM_EXECUTION_MODE"] = "umas_fast_pytorch"
        os.environ["FAIRCHEM_MERGE_MOLE"] = "1"
    else:
        os.environ["FAIRCHEM_EXECUTION_MODE"] = "general"
        os.environ["FAIRCHEM_MERGE_MOLE"] = "0"

    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    sysinfo = SYSTEMS[args.system]
    # Full-precision LAMMPS .data is the single authoritative geometry (see
    # p0_common); reduced-precision .extxyz is not used.
    atoms = atoms_from_data(args.system)

    predictor = load_predict_unit(
        args.ckpt, device="cuda", inference_settings=fp64_settings(), workers=1
    )
    atoms.calc = FAIRChemCalculator(predictor, task_name=TASK)

    # untimed warmup
    _ = float(atoms.get_potential_energy())
    _ = atoms.get_forces()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # timed single point (parity target)
    if hasattr(atoms.calc, "results"):
        atoms.calc.results.clear()
    atoms.positions = atoms.positions.copy()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    e = float(atoms.get_potential_energy())
    f = np.asarray(atoms.get_forces(), dtype=np.float64)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sp_s = time.perf_counter() - t0

    # NVT 300 K timing (no velocities needed; Maxwell-Boltzmann init inside NHC)
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    if atoms.get_velocities() is None or not atoms.get_velocities().any():
        MaxwellBoltzmannDistribution(atoms, temperature_K=NVT_TEMP_K)
    dyn = NoseHooverChainNVT(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=NVT_TEMP_K,
        tdamp=100.0 * units.fs,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    dyn.run(args.nsteps)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    nvt_s = time.perf_counter() - t1

    out = results_dir()
    rec = {
        "path": "ase_fairchem_fp64",
        "system": args.system,
        "natoms": len(atoms),
        "task": TASK,
        "dtype": "float64",
        "execution_mode": fp64_settings().execution_mode,
        "merge_mole": bool(fp64_settings().merge_mole),
        "energy_eV": e,
        "sp_ms": sp_s * 1e3,
        "nvt_steps": args.nsteps,
        "nvt_ms_total": nvt_s * 1e3,
        "nvt_ms_per_step": (nvt_s / args.nsteps) * 1e3,
        "force_abs_max": float(np.abs(f).max()),
        "force_net": [float(x) for x in f.sum(axis=0)],
    }
    rec["recipe"] = args.recipe
    stem = f"{args.system}_ase_{args.recipe}"
    (out / f"{stem}.json").write_text(json.dumps(rec, indent=2) + "\n")
    np.savez(out / f"{stem}.npz", forces=f, energy_eV=np.array(e, dtype=np.float64))
    print(json.dumps(rec, indent=2))
    print(f"wrote {out / (stem + '.npz')} forces{f.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
