#!/usr/bin/env python3
"""FC LAMMPS probe on the 5832-atom NaCl box.

Records, as evidence rather than assertion:
  * whether the installed lmp actually has MPI (it is built with MPI STUBS,
    so it cannot span nodes), and
  * whether FC + FP64 + merge_mole runs or reproduces the merge_MOLE
    Float/Double crash that blocked the matching-settings FC bar.

Exits 0 either way: a reproduced blocker is a measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EX = Path(__file__).resolve().parent
ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")
ENGINE = ROOT / "src/ML-UMA/uma-engine"
LMP = ROOT / "build-uma/lmp"
CKPT = Path(os.environ.get(
    "UMA_CHECKPOINT", "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"))
XYZ = EX / "structures/nacl9rep_rattle.extxyz"
ORACLE_NPZ = EX / "oracle_ase_nacl9rep_merge.npz"
DE_TOL, DF_TOL = 1e-6, 1e-5


def mpi_capability() -> dict:
    try:
        out = subprocess.run([str(LMP), "-h"], capture_output=True, text=True,
                             timeout=120).stdout
    except Exception as exc:  # noqa: BLE001
        return {"lmp_mpi": f"probe failed: {exc}"[:200]}
    stubs = "MPI STUBS" in out
    line = next((ln.strip() for ln in out.splitlines() if "MPI v" in ln), "")
    return {
        "lmp_path": str(LMP),
        "lmp_mpi_line": line,
        "lmp_has_real_mpi": not stubs,
        "multinode_possible": not stubs,
        "blocker_1": ("build -DBUILD_MPI=OFF (MPI STUBS): a single lmp process "
                      "cannot span nodes") if stubs else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--task", default="omat")
    a = ap.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "manual")

    rec: dict = {
        "path": "fc_lammps", "sys": "nacl9rep", "job": job,
        "workers": a.workers, "ngpu": a.workers,
        "nnodes": int(os.environ.get("SLURM_JOB_NUM_NODES", "1")),
        "dtype": "float64", "merge_mole": True,
    }
    rec.update(mpi_capability())

    # Blocker 2: does FC FP64 + merge_mole even load on this box?
    try:
        import torch
        from ase.io import read
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit

        sys.path.insert(0, str(ENGINE / "python"))
        from common import inference_settings_with_dtype

        os.environ["FAIRCHEM_WORKERS"] = str(max(1, a.workers))
        s = inference_settings_with_dtype("float64")
        s.external_graph_gen = False
        s.activation_checkpointing = False
        s.execution_mode = "umas_fast_pytorch"
        s.merge_mole = True

        atoms = read(str(XYZ))
        rec["natoms"] = len(atoms)
        pred = load_predict_unit(str(CKPT), device="cuda",
                                 inference_settings=s, workers=max(1, a.workers))
        atoms.calc = FAIRChemCalculator(pred, task_name=a.task)
        e = float(atoms.get_potential_energy())
        f = np.asarray(atoms.get_forces(), dtype=np.float64)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        atoms.calc.results.clear()
        atoms.positions = atoms.positions.copy()
        e = float(atoms.get_potential_energy())
        f = np.asarray(atoms.get_forces(), dtype=np.float64)
        torch.cuda.synchronize()
        rec["ms_per_eval"] = (time.perf_counter() - t0) * 1e3
        rec["energy_eV"] = e
        rec["force_absmax"] = float(np.abs(f).max())
        rec["blocker_2"] = None
        rec["status"] = "OK"
        if ORACLE_NPZ.is_file():
            o = np.load(ORACLE_NPZ)
            if o["forces"].shape == f.shape:
                mag = np.linalg.norm(f - o["forces"], axis=1)
                rec["dE_vs_oracle"] = abs(e - float(o["energy_eV"]))
                rec["force_max_per_atom"] = float(mag.max())
                rec["n_atoms_over_tol"] = int((mag > DF_TOL).sum())
                rec["ef_pass"] = bool(rec["dE_vs_oracle"] <= DE_TOL
                                      and rec["force_max_per_atom"] <= DF_TOL)
        d = EX / "results" / f"fclammps_w{a.workers}_{job}"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "forces.npz", forces=f, energy_eV=np.array(e))
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "FAIL"
        rec["blocker_2"] = f"{type(exc).__name__}: {exc}"[:600]
        d = EX / "results" / f"fclammps_w{a.workers}_{job}"
        d.mkdir(parents=True, exist_ok=True)

    (d / "timing.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    print(f"FCLAMMPS_RECORD {d / 'timing.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
