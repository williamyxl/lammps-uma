#!/usr/bin/env python3
"""4-GPU capacity sweep via FairChem graph parallel over torch.distributed
(Ray-free). Launch with mpiexec -n <gp_size>. Each rank = one GP peer/GPU.
Sweeps NaCl N^3 with activation checkpointing to find the max that fits on the
aggregate GPU memory. Rank 0 prints results.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from ase.build import bulk

ENG = Path(__file__).resolve().parents[1].parent / "uma-engine"
sys.path.insert(0, str(ENG / "python"))
from common import inference_settings_with_dtype  # noqa: E402

TASK = "omat"


def rprint(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--checkpointing", type=int, default=1)
    ap.add_argument("--sizes", type=int, nargs="*", default=[8, 10, 12, 14, 16, 18, 20])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ---- torch.distributed from MPI env (Cray PALS sets PMI_*) ----
    rank = int(os.environ.get("PMI_RANK", os.environ.get("RANK", "0")))
    world = int(os.environ.get("PMI_SIZE", os.environ.get("WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("PMI_LOCAL_RANK", "0"))
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)

    from fairchem.core.common import gp_utils
    gp_utils.setup_graph_parallel_groups(world, "nccl")  # all ranks = one GP group

    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    s = inference_settings_with_dtype("float64")
    s.external_graph_gen = False       # GP requires internal graph gen
    s.activation_checkpointing = bool(args.checkpointing)
    s.execution_mode = "general"
    s.merge_mole = False
    predictor = load_predict_unit(args.ckpt, device="cuda", inference_settings=s, workers=1)
    calc = FAIRChemCalculator(predictor, task_name=TASK)

    rho = 4096 / 45.12**3
    rprint(rank, f"# GP sweep world={world} ckpt={args.checkpointing} "
                 f"on {torch.cuda.get_device_name(0)}")
    results = []
    max_ok = 0
    for nrep in args.sizes:
        atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((nrep, nrep, nrep))
        atoms.pbc = True
        n = len(atoms)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        rec = {"nrep": nrep, "natoms": n, "ok": False}
        try:
            atoms.calc = calc
            e = float(atoms.get_potential_energy())
            f = np.asarray(atoms.get_forces(), dtype=np.float64)
            torch.cuda.synchronize()
            rec.update(ok=True, energy_eV=e, fmax=float(np.abs(f).max()),
                       peak_gib=torch.cuda.max_memory_allocated()/1024**3)
        except RuntimeError as ex:
            rec["err"] = "OOM" if "out of memory" in str(ex).lower() else str(ex)[:160]
            try: rec["peak_gib"] = torch.cuda.max_memory_allocated()/1024**3
            except Exception: pass
        finally:
            del atoms; gc.collect(); torch.cuda.empty_cache()
        results.append(rec)
        L = (n/rho)**(1/3)
        rprint(rank, f"  NaCl {nrep:2d}^3 N={n:7d} box={L:6.1f}A "
                     f"peak/rank={rec.get('peak_gib',0):6.2f}G "
                     f"{'OK' if rec['ok'] else 'FAIL('+rec.get('err','?')+')'}")
        # all ranks agree on OOM (collective); stop together
        ok_t = torch.tensor([1 if rec["ok"] else 0], device="cuda")
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        if ok_t.item() == 1:
            max_ok = n
        else:
            break

    if rank == 0:
        L = (max_ok/rho)**(1/3) if max_ok else 0
        summary = {"world": world, "checkpointing": bool(args.checkpointing),
                   "max_atoms_fit": max_ok, "max_box_A": L, "results": results}
        if args.out:
            Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"# GP_MAX_ATOMS_FIT[world={world}] = {max_ok}  box ~{L:.1f} A", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
