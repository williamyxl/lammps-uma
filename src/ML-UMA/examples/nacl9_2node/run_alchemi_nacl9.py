#!/usr/bin/env python3
"""ALCHEMI multi-node UMA: single-point E+F and NVT 300 K, fully decomposed.

Everything runs through DomainParallel so all requested GPUs share the graph.
There is deliberately NO bare ``model(batch)`` here: that call ignores the mesh
and evaluates the whole system on one GPU, which is what made an 8-GPU job OOM
as if it had one GPU. Both the single point (``DomainParallel.compute``) and MD
(``DomainParallel.run``) use the decomposition, so per-rank memory falls with
rank count.

Verification (parity vs oracle, 8-atom periodicity, per-atom force stats) is
NOT done here -- it lives in a standalone checker that reads forces.npz, the
same split used for the LibTorch paths. This file only runs and records.

  srun --mpi=pmix python run_alchemi_nacl9.py --ngpu <ntasks> --nsteps 10
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

EX = Path(__file__).resolve().parent
CKPT = Path(os.environ.get(
    "UMA_CHECKPOINT", "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"))
TAG = os.environ.get("NACL_TAG", "nacl8rep")
XYZ = EX / f"structures/{TAG}_rattle.extxyz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngpu", type=int, default=8)
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--timestep-fs", type=float, default=1.0)
    ap.add_argument("--thermostat-time-fs", type=float, default=100.0)
    ap.add_argument("--strategy", default="graph_partition",
                    choices=("graph_partition", "halo"))
    a = ap.parse_args()

    job = os.environ.get("SLURM_JOB_ID", "manual")
    from ase.io import read
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.distributed import (
        DistributedManager, DomainConfig, DomainParallel,
    )
    from nvalchemi.distributed.config import StrategyKind
    from nvalchemi.dynamics import NVTNoseHoover
    from nvalchemi.models.uma import UMAWrapper

    # ---- distributed bring-up ------------------------------------------------
    DistributedManager.initialize()
    dm = DistributedManager()
    device = torch.device(dm.device)
    world, rank = dm.world_size, dm.rank
    nnodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # ---- model (FP64 + merge_mole, same as the campaign oracle) -------------
    settings = InferenceSettings(
        tf32=False, activation_checkpointing=False, merge_mole=True,
        compile=False, external_graph_gen=False)
    settings.base_precision_dtype = getattr(torch, a.dtype)
    model = UMAWrapper.from_checkpoint(
        str(CKPT), task_name="omat", device=device.type,
        inference_settings=settings)
    if model.predict_unit.inference_settings.base_precision_dtype \
            is not getattr(torch, a.dtype):
        raise RuntimeError("precision override did not stick")
    model.model_config.active_outputs = {"energy", "forces"}

    dt_t = getattr(torch, a.dtype)
    atoms = read(str(XYZ))

    rec: dict = {
        "path": "alchemi", "sys": TAG, "natoms": len(atoms),
        "ngpu": a.ngpu, "world_size": world, "nnodes": nnodes,
        "dtype": a.dtype, "task": "omat", "nsteps": a.nsteps,
        "temperature_K": a.temperature, "timestep_fs": a.timestep_fs,
        "thermostat_time_fs": a.thermostat_time_fs, "job": job,
        "merge_mole": True, "integrator": "NVTNoseHoover",
        "strategy": a.strategy, "decomposed": True,
    }

    def fresh_batch():
        b = Batch.from_data_list(
            [AtomicData.from_atoms(atoms, dtype=dt_t)]).to(device)
        b.forces = torch.zeros_like(b.positions)
        b.energy = torch.zeros(1, 1, device=device, dtype=b.positions.dtype)
        b.velocities = torch.zeros_like(b.positions)
        return b

    strat = (StrategyKind.GRAPH_PARTITION if a.strategy == "graph_partition"
             else StrategyKind.HALO)
    mesh = DistributedManager().initialize_mesh(
        mesh_shape=(world,), mesh_dim_names=("domain",))
    cut = float(getattr(model, "cutoff", 6.0) or 6.0)
    dcfg = DomainConfig(cutoff=cut, skin=0.5, mesh=mesh, strategy=strat)

    try:
        # ---- decomposed single point: forces via DomainParallel.compute -----
        integ_sp = NVTNoseHoover(
            model=model, dt=a.timestep_fs, temperature=a.temperature,
            thermostat_time=a.thermostat_time_fs, n_steps=1)
        with DomainParallel(dynamics=integ_sp, config=dcfg, n_steps=1) as dp:
            owned = dp.partition(fresh_batch() if rank == 0 else None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = dp.compute(owned)
            if device.type == "cuda":
                torch.cuda.synchronize()
            rec["sp_ms"] = (time.perf_counter() - t0) * 1e3
            full = dp.gather(owned)   # rank 0 gets the whole system
            if rank == 0 and full is not None:
                e = float(np.asarray(full.energy.detach().cpu()).reshape(-1)[0])
                f = full.forces.detach().to(torch.float64).cpu().numpy()
                rec["energy_eV"] = e
                rec["energy_per_atom_eV"] = e / len(atoms)
                rec["force_absmax"] = float(np.abs(f).max())
                d = EX / "results" / f"alchemi_{TAG}_ngpu{a.ngpu}_{job}"
                d.mkdir(parents=True, exist_ok=True)
                np.savez(d / "forces.npz", forces=f, energy_eV=np.array(e))
        rec["sp_status"] = "OK"
    except Exception as exc:  # noqa: BLE001
        rec["sp_status"] = f"FAIL: {type(exc).__name__}: {exc}"[:400]

    try:
        # ---- decomposed NVT -------------------------------------------------
        integ = NVTNoseHoover(
            model=model, dt=a.timestep_fs, temperature=a.temperature,
            thermostat_time=a.thermostat_time_fs, n_steps=a.nsteps)
        with DomainParallel(dynamics=integ, config=dcfg,
                            n_steps=a.nsteps) as dp:
            owned = dp.partition(fresh_batch() if rank == 0 else None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            dp.run(owned)
            if device.type == "cuda":
                torch.cuda.synchronize()
            md_s = time.perf_counter() - t1
        rec["nvt_ms_per_step"] = md_s * 1e3 / max(1, a.nsteps)
        rec["nvt_total_s"] = md_s
        rec["nvt_status"] = "OK"
    except Exception as exc:  # noqa: BLE001
        rec["nvt_status"] = f"FAIL: {type(exc).__name__}: {exc}"[:400]

    if device.type == "cuda":
        rec["vram_peak_GiB"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    if rank == 0:
        d = EX / "results" / f"alchemi_{TAG}_ngpu{a.ngpu}_{job}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "timing.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps(rec, indent=2))
        print(f"ALCHEMI_RECORD {d / 'timing.json'}")

    DistributedManager.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
