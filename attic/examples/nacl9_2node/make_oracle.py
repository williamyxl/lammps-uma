#!/usr/bin/env python3
"""Dedicated ASE FP64 merge oracle for the perturb-then-replicate 9x9x9 box.

Single GPU, workers=1 (never ParallelMLIPPredictUnit), FP64,
execution_mode=umas_fast_pytorch + merge_mole -- the campaign's merge-oracle
settings, so every path can be gated the same way it is for nacl6.

Also checks the structure's built-in symmetry: because the perturbation was
applied to the unit cell before replication, forces must repeat every 8 atoms.
That is an oracle-independent correctness test on the reference itself.
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


def periodicity_report(f: np.ndarray, motif: int = 8) -> dict:
    """Forces must repeat every `motif` atoms for this structure."""
    n = f.shape[0]
    if n % motif:
        return {"motif_check": "N_NOT_DIVISIBLE"}
    g = f.reshape(-1, motif, 3)          # (ncells, motif, 3)
    ref = g[0]                            # first cell's motif forces
    dev = np.linalg.norm(g - ref[None, :, :], axis=2)   # (ncells, motif)
    return {
        "motif": motif,
        "n_cells": int(g.shape[0]),
        "motif_max_dev": float(dev.max()),
        "motif_mean_dev": float(dev.mean()),
        "motif_force_per_atom": [[float(x) for x in v] for v in ref],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", default=str(EX / "structures/nacl9rep_rattle.extxyz"))
    ap.add_argument("--task", default="omat")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(EX / "oracle_ase_nacl9rep_merge.npz"))
    a = ap.parse_args()

    import torch
    from ase.io import read
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    sys.path.insert(0, str(ENGINE / "python"))
    from common import inference_settings_with_dtype

    os.environ["FAIRCHEM_WORKERS"] = "1"   # never ParallelMLIPPredictUnit / Ray
    settings = inference_settings_with_dtype("float64")
    settings.external_graph_gen = False
    settings.activation_checkpointing = False
    settings.execution_mode = "umas_fast_pytorch"
    settings.merge_mole = True

    atoms = read(a.xyz)
    print(f"oracle: natoms={len(atoms)} task={a.task} FP64 umas_fast+merge workers=1",
          flush=True)

    predictor = load_predict_unit(str(CKPT), device="cuda",
                                  inference_settings=settings, workers=1)
    calc = FAIRChemCalculator(predictor, task_name=a.task)
    atoms.calc = calc

    # warmup (lazy FP64 cast + MoLE merge) then timed repeats
    e = float(atoms.get_potential_energy())
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

    rec = {
        "oracle": "ase_fp64_ufast_merge_w1",
        "structure": a.xyz,
        "natoms": int(len(atoms)),
        "task": a.task,
        "energy_eV": e,
        "energy_per_atom_eV": e / len(atoms),
        "ms_per_eval": float(np.median(times)),
        "ms_all": [round(x, 2) for x in times],
        "force_absmax": float(np.abs(f).max()),
        "force_sum_abs": float(np.abs(f.sum(axis=0)).max()),
        "vram_peak_GiB": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "checkpoint": str(CKPT),
        "cold_start_excluded": True,
    }
    rec.update(periodicity_report(f))

    np.savez(a.out, forces=f, energy_eV=np.array(e))
    Path(str(a.out).replace(".npz", ".json")).write_text(
        json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    print(f"ORACLE_OK {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
