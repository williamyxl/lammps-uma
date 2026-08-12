#!/usr/bin/env python3
"""Verify activation checkpointing keeps E/F identical, and measure its step cost.

Runs the FairChem UMA model (FP64, task=omat) on the same NaCl geometry twice --
checkpointing OFF then ON -- and compares energy + per-atom forces, plus timed
repeated single points (proxy for MD step cost).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk

ENG = Path(__file__).resolve().parents[1].parent / "uma-engine"
sys.path.insert(0, str(ENG / "python"))
from common import inference_settings_with_dtype  # noqa: E402

TASK = "omat"


def settings(ckpt: bool):
    s = inference_settings_with_dtype("float64")
    s.external_graph_gen = True
    s.activation_checkpointing = bool(ckpt)
    s.execution_mode = "general"
    s.merge_mole = False
    return s


def run(ckpt_flag: bool, ckpt_path: str, nrep: int, iters: int, save_on_cpu: bool = False):
    import contextlib
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((nrep, nrep, nrep))
    atoms.pbc = True
    pred = load_predict_unit(ckpt_path, device="cuda", inference_settings=settings(ckpt_flag))
    atoms.calc = FAIRChemCalculator(pred, task_name=TASK)
    cm = lambda: (torch.autograd.graph.save_on_cpu(pin_memory=True)
                  if save_on_cpu else contextlib.nullcontext())
    # warmup
    with cm():
        e = float(atoms.get_potential_energy())
        f = np.asarray(atoms.get_forces(), dtype=np.float64)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        atoms.positions += np.random.default_rng(0).normal(0, 1e-5, atoms.positions.shape)
        if hasattr(atoms.calc, "results"):
            atoms.calc.results.clear()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with cm():
            _ = float(atoms.get_potential_energy()); _ = atoms.get_forces()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return e, f, float(np.median(ts)), torch.cuda.max_memory_allocated() / 1024**3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nrep", type=int, default=6)   # 1728 atoms fits both
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    configs = [
        ("baseline",       False, False),
        ("ckpt",           True,  False),
        ("save_on_cpu",    False, True),
        ("soc+ckpt",       True,  True),
    ]
    base = None
    rows = []
    for name, ck, soc in configs:
        torch.cuda.reset_peak_memory_stats()
        e, f, ms, mem = run(ck, args.ckpt, args.nrep, args.iters, soc)
        if base is None:
            base = (e, f, ms, mem)
        dE = abs(e - base[0]); dF = float(np.linalg.norm(f - base[1], axis=1).max())
        rows.append((name, e, ms, mem, ms/base[2], base[3]/mem, dE, dF))
        print(f"{name:12s} E={e:.9f} ms={ms:8.1f} peak={mem:6.2f}G "
              f"time={ms/base[2]:.2f}x memsave={base[3]/mem:.2f}x |dE|={dE:.2e} max|dF|={dF:.2e}",
              flush=True)
    print("\nPARITY", "OK" if all(r[6] < 1e-6 and r[7] < 1e-5 for r in rows) else "DIFF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
