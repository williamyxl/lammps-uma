#!/usr/bin/env python3
"""ASE FairChem FP64 — first-frame E+F parity + post-warmup NVT timing (default 100 steps)."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from ase import units
from ase.io import read
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from path_common import (
    DEFAULT_CKPT,
    DEFAULT_DATA,
    fairchem_knobs_from_env,
    fp64_settings,
    out_dir,
    write_timing,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    ap.add_argument("--nsteps", type=int, default=int(os.environ.get("NSTEPS", "100")))
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("FAIRCHEM_WORKERS", "1")),
    )
    args = ap.parse_args()
    workers = max(1, args.workers)
    nsteps = max(1, args.nsteps)
    out = out_dir("ase")

    atoms = read(args.data, format="lammps-data", atom_style="atomic", units="metal")
    if atoms.get_velocities() is None:
        raise RuntimeError("missing Velocities")

    predictor = load_predict_unit(
        args.ckpt,
        device="cuda",
        inference_settings=fp64_settings(workers=workers, external_graph=(workers <= 1)),
        workers=workers,
    )
    atoms.calc = FAIRChemCalculator(predictor, task_name="omat")
    # untimed warmup
    _ = float(atoms.get_potential_energy())
    _ = atoms.get_forces()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # timed SP = first-frame E+F (parity)
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

    dyn = NoseHooverChainNVT(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=300.0,
        tdamp=100.0 * units.fs,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    dyn.run(nsteps)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    nvt_s = time.perf_counter() - t1

    mode, merge = fairchem_knobs_from_env()
    write_timing(
        out,
        {
            "path": "ase_fairchem_fp64",
            "key": "ase",
            "jobid": os.environ.get("SLURM_JOB_ID"),
            "natoms": len(atoms),
            "nsteps": nsteps,
            "temperature_K": 300.0,
            "dtype": "float64",
            "workers": workers,
            "ngpus": workers,
            "execution_mode": mode,
            "merge_mole": merge,
            "energy_eV": e,
            "sp_ms": sp_s * 1e3,
            "nvt_ms_total": nvt_s * 1e3,
            "nvt_ms_per_step": (nvt_s / nsteps) * 1e3,
            "timing_source_sp": "post_warmup_ase_first_frame_ef",
            "timing_source_nvt": "post_warmup_ase_nhc_run",
            "cold_start_excluded": True,
            "warmup": True,
            "parity_frame": "first",
            "nvt_frame_dumps": False,
        },
        forces=f,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
