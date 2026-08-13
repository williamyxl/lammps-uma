#!/usr/bin/env python3
"""ASE-FairChem FP64 + checkpointing reference (E + per-atom F) for a LAMMPS .data
geometry, on 4 GPUs via graph parallel over torch.distributed (Ray-free).
Launch with mpiexec -n 4. Rank 0 writes <out>.npz (forces, energy)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from ase.io import read
from ase.data import atomic_numbers as AN

ENG = Path(__file__).resolve().parents[1].parent / "uma-engine"
sys.path.insert(0, str(ENG / "python"))
from common import inference_settings_with_dtype  # noqa: E402

TASK = "omat"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=os.environ.get("UMA_CHECKPOINT"))
    args = ap.parse_args()

    rank = int(os.environ.get("PMI_RANK", "0"))
    world = int(os.environ.get("PMI_SIZE", "1"))
    local = int(os.environ.get("PMI_LOCAL_RANK", "0"))
    os.environ.setdefault("RANK", str(rank)); os.environ.setdefault("WORLD_SIZE", str(world))
    torch.cuda.set_device(local % max(1, torch.cuda.device_count()))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    from fairchem.core.common import gp_utils
    gp_utils.setup_graph_parallel_groups(world, "nccl")

    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    a = read(args.data, format="lammps-data", atom_style="atomic", units="metal")
    if a.has("type"):
        a.set_atomic_numbers([AN[["Na", "Cl"][t - 1]] for t in a.get_array("type")])
    a.pbc = True

    s = inference_settings_with_dtype("float64")
    s.external_graph_gen = False
    s.activation_checkpointing = True
    s.execution_mode = "general"
    s.merge_mole = False
    pred = load_predict_unit(args.ckpt, device="cuda", inference_settings=s, workers=1)
    a.calc = FAIRChemCalculator(pred, task_name=TASK)

    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(), dtype=np.float64)
    if rank == 0:
        np.savez(args.out, forces=f, energy_eV=np.array(e), natoms=len(a))
        print(f"ASE ckpt ref: N={len(a)} E={e:.9f} fmax={np.abs(f).max():.3e} -> {args.out}",
              flush=True)
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
