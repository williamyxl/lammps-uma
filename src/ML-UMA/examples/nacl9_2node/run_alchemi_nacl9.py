#!/usr/bin/env python3
"""ALCHEMI on the 5832-atom NaCl box: E + per-atom F + NVT 300 K, multi-node.

Reuses the validated nvalchemi_path settings (FP64 both on InferenceSettings
and AtomicData, NoseHoover tau_T=100 fs matching LAMMPS `fix nvt ... 0.1`,
warmup excluded). Adds the 8-atom periodicity check, which is exact for this
perturb-then-replicate structure and independent of any oracle.
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
NVPATH = EX.parent / "nvalchemi_path"
CKPT = Path(os.environ.get(
    "UMA_CHECKPOINT", "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"))
XYZ = EX / "structures/nacl9rep_rattle.extxyz"
ORACLE_NPZ = EX / "oracle_ase_nacl9rep_merge.npz"
DE_TOL, DF_TOL = 1e-6, 1e-5


def periodicity_report(f: np.ndarray, motif: int = 8) -> dict:
    n = f.shape[0]
    if n % motif:
        return {"motif_check": "N_NOT_DIVISIBLE"}
    g = f.reshape(-1, motif, 3)
    dev = np.linalg.norm(g - g[0][None, :, :], axis=2)
    return {"motif": motif, "n_cells": int(g.shape[0]),
            "motif_max_dev": float(dev.max()),
            "motif_mean_dev": float(dev.mean()),
            "motif_pass": bool(dev.max() <= 1e-5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngpu", type=int, default=8)
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--timestep-fs", type=float, default=1.0)
    ap.add_argument("--thermostat-time-fs", type=float, default=100.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sp-repeats", type=int, default=3)
    ap.add_argument("--md-warmup-steps", type=int, default=2)
    a = ap.parse_args()

    job = os.environ.get("SLURM_JOB_ID", "manual")
    from ase.io import read
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.distributed import (
        DistributedManager, DomainConfig, DomainParallel,
    )
    from nvalchemi.dynamics import DynamicsStage, NVTNoseHoover
    from nvalchemi.hooks import HookContext
    from nvalchemi.models.uma import UMAWrapper

    distributed = a.ngpu > 1
    if distributed:
        DistributedManager.initialize()
        dm = DistributedManager()
        device = torch.device(dm.device)
        world, rank = dm.world_size, dm.rank
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world, rank = 1, 0
    nnodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))

    atoms = read(str(XYZ))
    settings = InferenceSettings(
        tf32=False, activation_checkpointing=False, merge_mole=True,
        compile=False, external_graph_gen=False)
    settings.base_precision_dtype = getattr(torch, a.dtype)
    # fairchem asserts device in ["cpu","cuda"] -- a "cuda:N" string fails.
    # torch.cuda.set_device pins the rank's GPU, so bare "cuda" is correct.
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = UMAWrapper.from_checkpoint(
        str(CKPT), task_name="omat", device=device.type,
        inference_settings=settings)
    eff = model.predict_unit.inference_settings.base_precision_dtype
    if eff is not getattr(torch, a.dtype):
        raise RuntimeError(f"precision override failed: {eff}")
    model.model_config.active_outputs = {"energy", "forces"}

    rec: dict = {
        "path": "alchemi", "sys": "nacl9rep", "natoms": len(atoms),
        "ngpu": a.ngpu, "world_size": world, "nnodes": nnodes,
        "dtype": a.dtype, "task": "omat", "nsteps": a.nsteps,
        "temperature_K": a.temperature, "timestep_fs": a.timestep_fs,
        "thermostat_time_fs": a.thermostat_time_fs, "job": job,
        "merge_mole": True, "integrator": "NVTNoseHoover",
        "strategy": "halo (DomainConfig default)",
        "cold_start_excluded": True,
    }

    dt_t = getattr(torch, a.dtype)
    data = AtomicData.from_atoms(atoms, dtype=dt_t)   # float32 default -> pass dtype
    batch = Batch.from_data_list([data]).to(device)
    nl_hooks = list(model.make_neighbor_hooks())
    ctx = HookContext(batch=batch, model=model, global_rank=rank, workflow=None)
    for h in nl_hooks:
        h(ctx, DynamicsStage.BEFORE_COMPUTE)

    for _ in range(max(0, a.warmup)):
        _w = model(batch)
        del _w
    torch.cuda.synchronize()
    sp = []
    for _ in range(max(1, a.sp_repeats)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(batch)
        torch.cuda.synchronize()
        sp.append((time.perf_counter() - t0) * 1e3)
    rec["sp_ms"] = float(np.median(sp))
    rec["sp_ms_all"] = [round(x, 2) for x in sp]

    e = float(out["energy"].reshape(-1)[0].item())
    f = out["forces"].detach().to(torch.float64).cpu().numpy()
    rec["energy_eV"] = e
    rec["energy_per_atom_eV"] = e / len(atoms)
    rec["param_dtype"] = str(next(
        p.dtype for p in model.predict_unit.model.parameters()
        if p.is_floating_point()))
    rec["precision_ok"] = rec["param_dtype"] == f"torch.{a.dtype}"
    rec["force_absmax"] = float(np.abs(f).max())
    rec["force_sum_abs"] = float(np.abs(f.sum(axis=0)).max())
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
                "force_rms_per_atom": float(np.sqrt((mag**2).mean())),
                "n_atoms_over_tol": int((mag > DF_TOL).sum()),
                "worst_atoms": [
                    {"index": int(i), "abs_err": float(mag[i]),
                     "F": [float(x) for x in f[i]],
                     "F_oracle": [float(x) for x in fr[i]]}
                    for i in np.argsort(mag)[::-1][:10]],
            })
            rec["ef_pass"] = bool(rec["dE_vs_oracle"] <= DE_TOL
                                  and rec["force_max_per_atom"] <= DF_TOL)
    else:
        rec["oracle"] = "NOT_AVAILABLE"

    # ---- NVT ----
    def mk(n):
        return NVTNoseHoover(model=model, dt=a.timestep_fs,
                             temperature=a.temperature,
                             thermostat_time=a.thermostat_time_fs, n_steps=n)

    def fresh_batch():
        b = Batch.from_data_list(
            [AtomicData.from_atoms(atoms, dtype=dt_t)]).to(device)
        b.forces = torch.zeros_like(b.positions)
        b.energy = torch.zeros(1, 1, device=device, dtype=b.positions.dtype)
        b.velocities = torch.zeros_like(b.positions)
        return b

    nwarm = max(0, a.md_warmup_steps)
    try:
        if distributed:
            mesh = DistributedManager().initialize_mesh(
                mesh_shape=(world,), mesh_dim_names=("domain",))
            cut = float(getattr(model, "cutoff", 6.0) or 6.0)
            dcfg = DomainConfig(cutoff=cut, skin=0.5, mesh=mesh)
            if nwarm:
                with DomainParallel(dynamics=mk(nwarm), config=dcfg,
                                    n_steps=nwarm) as wd:
                    wd.run(wd.partition(fresh_batch() if rank == 0 else None))
                torch.cuda.synchronize()
            with DomainParallel(dynamics=mk(a.nsteps), config=dcfg,
                                n_steps=a.nsteps) as dyn:
                owned = dyn.partition(fresh_batch() if rank == 0 else None)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                dyn.run(owned)
                torch.cuda.synchronize()
                md_s = time.perf_counter() - t1
        else:
            if nwarm:
                wi = mk(nwarm)
                for h in model.make_neighbor_hooks():
                    wi.register_hook(h, stage=DynamicsStage.BEFORE_COMPUTE)
                with wi:
                    wi.run(fresh_batch())
                torch.cuda.synchronize()
            integ = mk(a.nsteps)
            for h in model.make_neighbor_hooks():
                integ.register_hook(h, stage=DynamicsStage.BEFORE_COMPUTE)
            with integ:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                integ.run(fresh_batch())
                torch.cuda.synchronize()
                md_s = time.perf_counter() - t1
        rec["nvt_ms_per_step"] = md_s * 1e3 / max(1, a.nsteps)
        rec["nvt_total_s"] = md_s
        rec["md_warmup_steps"] = nwarm
        rec["nvt_status"] = "OK"
    except Exception as exc:  # noqa: BLE001
        rec["nvt_status"] = f"FAIL: {type(exc).__name__}: {exc}"[:400]

    rec["vram_peak_GiB"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    if rank == 0:
        d = EX / "results" / f"alchemi_ngpu{a.ngpu}_{job}"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "forces.npz", forces=f, energy_eV=np.array(e))
        (d / "timing.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps(rec, indent=2))
        print(f"ALCHEMI_RECORD {d / 'timing.json'}")

    if distributed:
        DistributedManager.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
