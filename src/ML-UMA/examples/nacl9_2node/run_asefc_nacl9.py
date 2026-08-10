#!/usr/bin/env python3
"""ASE FairChem FP64 on the 5832-atom NaCl box, N workers (Ray when N>1).

Reports E, per-atom F vs the dedicated oracle, the 8-atom periodicity check,
and per-eval timing with warmup excluded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

EX = Path(__file__).resolve().parent
ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")
ENGINE = ROOT / "src/ML-UMA/uma-engine"
CKPT = Path(os.environ.get(
    "UMA_CHECKPOINT", "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"))
XYZ = EX / "structures/nacl9rep_rattle.extxyz"
ORACLE_NPZ = EX / "oracle_ase_nacl9rep_merge.npz"
DE_TOL, DF_TOL = 1e-6, 1e-5


def periodicity_report(f: np.ndarray, motif: int = 8) -> dict:
    if f.shape[0] % motif:
        return {"motif_check": "N_NOT_DIVISIBLE"}
    g = f.reshape(-1, motif, 3)
    dev = np.linalg.norm(g - g[0][None, :, :], axis=2)
    return {"motif": motif, "n_cells": int(g.shape[0]),
            "motif_max_dev": float(dev.max()),
            "motif_mean_dev": float(dev.mean()),
            "motif_pass": bool(dev.max() <= 1e-5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--task", default="omat")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    import torch
    from ase.io import read
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    sys.path.insert(0, str(ENGINE / "python"))
    from common import inference_settings_with_dtype

    job = os.environ.get("SLURM_JOB_ID", "manual")
    workers = max(1, a.workers)
    os.environ["FAIRCHEM_WORKERS"] = str(workers)
    settings = inference_settings_with_dtype("float64")
    settings.external_graph_gen = workers <= 1
    settings.activation_checkpointing = False
    settings.execution_mode = "umas_fast_pytorch"
    settings.merge_mole = True

    atoms = read(str(XYZ))
    rec: dict = {
        "path": "ase_fc_fp64", "sys": "nacl9rep", "natoms": len(atoms),
        "workers": workers, "ngpu": workers,
        "nnodes": int(os.environ.get("SLURM_JOB_NUM_NODES", "1")),
        "dtype": "float64", "task": a.task, "job": job,
        "merge_mole": True, "transport": "ray" if workers > 1 else "single",
        "cold_start_excluded": True,
        "note": ("workers>1 uses Ray, which the campaign forbids for locked "
                 "bars; multi-node capability probe only"),
    }

    try:
        predictor = load_predict_unit(str(CKPT), device="cuda",
                                      inference_settings=settings,
                                      workers=workers)
        calc = FAIRChemCalculator(predictor, task_name=a.task)
        atoms.calc = calc
        e = float(atoms.get_potential_energy())          # warmup
        f = np.asarray(atoms.get_forces(), dtype=np.float64)
        torch.cuda.synchronize()

        times = []
        for _ in range(max(1, a.repeats)):
            atoms.calc.results.clear()
            atoms.positions = atoms.positions.copy()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            e = float(atoms.get_potential_energy())
            f = np.asarray(atoms.get_forces(), dtype=np.float64)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)

        rec.update({
            "energy_eV": e, "energy_per_atom_eV": e / len(atoms),
            "ms_per_eval": float(np.median(times)),
            "ms_all": [round(x, 2) for x in times],
            "force_absmax": float(np.abs(f).max()),
            "force_sum_abs": float(np.abs(f.sum(axis=0)).max()),
            "vram_peak_GiB": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "status": "OK",
        })
        rec.update(periodicity_report(f))

        if ORACLE_NPZ.is_file():
            o = np.load(ORACLE_NPZ)
            fr, er = o["forces"], float(o["energy_eV"])
            if fr.shape == f.shape:
                mag = np.linalg.norm(f - fr, axis=1)
                rec.update({
                    "dE_vs_oracle": abs(e - er),
                    "force_max_per_atom": float(mag.max()),
                    "force_mean_per_atom": float(mag.mean()),
                    "n_atoms_over_tol": int((mag > DF_TOL).sum()),
                })
                rec["ef_pass"] = bool(rec["dE_vs_oracle"] <= DE_TOL
                                      and rec["force_max_per_atom"] <= DF_TOL)
        d = EX / "results" / f"asefc_w{workers}_{job}"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "forces.npz", forces=f, energy_eV=np.array(e))
        (d / "timing.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps(rec, indent=2))
        print(f"ASEFC_RECORD {d / 'timing.json'}")
    except Exception as exc:  # noqa: BLE001
        rec["status"] = f"FAIL: {type(exc).__name__}: {exc}"[:600]
        d = EX / "results" / f"asefc_w{workers}_{job}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "timing.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps(rec, indent=2))
        return 0   # record the failure as data, do not mask it as a crash
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
