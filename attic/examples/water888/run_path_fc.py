#!/usr/bin/env python3
"""FairChem LAMMPS — first-frame E+F parity + post-warmup NVT timing (default 100 steps)."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.lammps.lammps_fc import run_lammps_with_fairchem

from path_common import (
    DEFAULT_CKPT,
    DEFAULT_DATA,
    fairchem_knobs_from_env,
    find_fc_lmp,
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
    out = out_dir("fc")
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    data = work / "water_nvt_300K_atomic_metal.data"
    shutil.copy2(args.data, data)

    fc_lmp = find_fc_lmp()
    os.environ["PATH"] = f"{fc_lmp.parent}:{os.environ.get('PATH', '')}"

    predictor = load_predict_unit(
        args.ckpt,
        device="cuda",
        inference_settings=fp64_settings(workers=workers, external_graph=False),
        workers=workers,
    )

    inp = work / "in.fc"
    inp.write_text(
        f"""units metal
atom_style atomic
boundary p p p
read_data {data.name}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
fix 1 all nvt temp 300.0 300.0 0.1
timestep 0.001
thermo 10
thermo_style custom step temp pe ke etotal
thermo_modify norm no
run 0
"""
    )

    cwd = Path.cwd()
    os.chdir(work)
    try:
        lmp = run_lammps_with_fairchem(predictor, str(inp.name), "omat")
        nlocal = lmp.extract_global("nlocal")
        tags = np.array(lmp.numpy.extract_atom("id")[:nlocal]).copy()
        order = np.argsort(tags)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # timed first-frame E+F
        t0 = time.perf_counter()
        lmp.command("run 0")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sp_s = time.perf_counter() - t0
        e = float(lmp.get_thermo("pe"))
        f = np.array(lmp.numpy.extract_atom("f")[:nlocal], dtype=np.float64).copy()[
            order
        ]

        # timed NVT (single run N — no per-frame dumps)
        t1 = time.perf_counter()
        lmp.command(f"run {nsteps}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        nvt_s = time.perf_counter() - t1

        if hasattr(lmp, "_predictor"):
            del lmp._predictor
        lmp.close()
        del lmp
    finally:
        os.chdir(cwd)

    mode, merge = fairchem_knobs_from_env()
    write_timing(
        out,
        {
            "path": "fairchem_lammps_fix_external",
            "key": "fc",
            "jobid": os.environ.get("SLURM_JOB_ID"),
            "natoms": int(f.shape[0]),
            "nsteps": nsteps,
            "temperature_K": 300.0,
            "dtype": "float64",
            "workers": workers,
            "ngpus": workers,
            "execution_mode": mode,
            "merge_mole": merge,
            "fairchem_lmp": str(fc_lmp),
            "note": "lammps_fc builds cell in FP32; first-frame E+F only + NVT timing",
            "energy_eV": e,
            "sp_ms": sp_s * 1e3,
            "nvt_ms_total": nvt_s * 1e3,
            "nvt_ms_per_step": (nvt_s / nsteps) * 1e3,
            "timing_source_sp": "post_warmup_fc_first_frame_run0",
            "timing_source_nvt": "post_warmup_fc_run_nsteps",
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
