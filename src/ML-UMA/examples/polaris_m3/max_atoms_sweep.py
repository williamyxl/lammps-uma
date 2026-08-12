#!/usr/bin/env python3
"""Max-atoms capacity sweep on ONE A100-40GB: largest NaCl cube that fits a
single-point energy+force (autograd) before CUDA OOM.

Runs the FairChem UMA calculator (FP64, task=omat) directly so we can toggle
activation checkpointing via InferenceSettings and measure its capacity effect.

  --ckpt PATH          UMA checkpoint
  --checkpointing 0|1  activation_checkpointing (default 0)
  --start / --step     NaCl conventional-cell repeat sizes to try (n^3 * ... )

Prints, per size: N atoms, peak GiB, ok/OOM. Reports the max N that fit.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk

ENG = Path(__file__).resolve().parents[1].parent / "uma-engine"
sys.path.insert(0, str(ENG / "python"))
from common import inference_settings_with_dtype  # noqa: E402

A_LATTICE = 5.64
TASK = "omat"


def make_nacl(nrep: int):
    a = bulk("NaCl", "rocksalt", a=A_LATTICE, cubic=True).repeat((nrep, nrep, nrep))
    a.pbc = True
    return a


_EDGE_CHUNK = None  # set from CLI


def settings(checkpointing: bool):
    s = inference_settings_with_dtype("float64")
    s.external_graph_gen = True
    s.activation_checkpointing = bool(checkpointing)
    s.execution_mode = "general"
    s.merge_mole = False
    if _EDGE_CHUNK:
        s.edge_chunk_size = int(_EDGE_CHUNK)
    return s


import contextlib


def try_size(predictor, nrep: int, save_on_cpu: bool = False) -> dict:
    from fairchem.core import FAIRChemCalculator

    atoms = make_nacl(nrep)
    n = len(atoms)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec = {"nrep": nrep, "natoms": n, "ok": False, "peak_gib": None, "err": None}
    try:
        atoms.calc = FAIRChemCalculator(predictor, task_name=TASK)
        # A3: automatic activation offload to pinned host RAM during backward.
        cm = (torch.autograd.graph.save_on_cpu(pin_memory=True)
              if save_on_cpu else contextlib.nullcontext())
        with cm:
            e = float(atoms.get_potential_energy())
            f = atoms.get_forces()
        torch.cuda.synchronize()
        rec["ok"] = True
        rec["energy_eV"] = e
        rec["fmax"] = float(np.abs(f).max())
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
    except RuntimeError as ex:
        msg = str(ex)
        rec["err"] = "OOM" if "out of memory" in msg.lower() else msg[:200]
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
    finally:
        del atoms
        gc.collect()
        torch.cuda.empty_cache()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--checkpointing", type=int, default=0)
    ap.add_argument("--save-on-cpu", type=int, default=0,
                    help="A3: offload saved activations to pinned host RAM")
    ap.add_argument("--edge-chunk", type=int, default=0,
                    help="C1: edge_chunk_size (stream edges); 0=off")
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=[6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global _EDGE_CHUNK
    _EDGE_CHUNK = args.edge_chunk or None

    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(
        args.ckpt, device="cuda", inference_settings=settings(bool(args.checkpointing))
    )
    tag = ("ckpt_on" if args.checkpointing else "ckpt_off") + ("_soc" if args.save_on_cpu else "") + (f"_ec{args.edge_chunk}" if args.edge_chunk else "")
    print(f"# capacity sweep [{tag}] on {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB)")
    results = []
    max_ok = 0
    for nrep in args.sizes:
        rec = try_size(predictor, nrep, bool(args.save_on_cpu))
        results.append(rec)
        status = "OK " if rec["ok"] else f"FAIL({rec['err']})"
        print(f"  NaCl {nrep:2d}^3  N={rec['natoms']:7d}  peak={rec['peak_gib'] or 0:6.2f} GiB  {status}",
              flush=True)
        if rec["ok"]:
            max_ok = rec["natoms"]
        elif rec["err"] == "OOM":
            break  # stop at first OOM
    summary = {"tag": tag, "checkpointing": bool(args.checkpointing),
               "max_atoms_fit": max_ok, "results": results}
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"# MAX_ATOMS_FIT[{tag}] = {max_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
