#!/usr/bin/env python3
"""ALCHEMI Toolkit path: UMA FP64 E+F + NVT 300 K on 1/2/4 GPUs.

Fourth comparison path alongside ASE FC FP64, FC LAMMPS, and LibTorch UMA
LAMMPS. Uses the SAME frozen boxes (NaCl 6x6x6 = 1728 atoms, water888 = 2592
atoms) and the SAME merge oracle, so E/F numbers are directly comparable.

Engine   : nvalchemi UMAWrapper (fairchem MLIPPredictUnit under the hood)
Dynamics : nvalchemi NVTNoseHoover (matches LAMMPS `fix nvt`, not Langevin)
Multi-GPU: DomainParallel spatial domain decomposition (torchrun ranks)

Single GPU:  python run_nvalchemi.py --sys nacl6 --ngpu 1
Multi  GPU:  torchrun --nproc_per_node=4 run_nvalchemi.py --sys nacl6 --ngpu 4
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
ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")
CKPT = Path(os.environ.get(
    "UMA_CHECKPOINT", "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"))

SYS = {
    "nacl6": {
        "xyz": ROOT / "src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz",
        "natoms": 1728, "nsteps": 10, "task": "omat",
    },
    "water888": {
        "xyz": ROOT / "src/ML-UMA/examples/water888/water_nvt_300K.extxyz",
        "natoms": 2592, "nsteps": 100, "task": "omol",
    },
}
# Campaign merge-oracle energies (ASE FP64 umas_fast_pytorch + merge_mole).
ORACLE_E = {"nacl6": -5830.9237413382, "water888": -3143.3893774722696}
CAMP = ROOT / "src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign"
ORACLE_F = {
    "nacl6": CAMP / "oracle_ase_umas_fast_merge.npz",
    "water888": CAMP / "oracle_ase_water_merge.npz",
}
DE_TOL, DF_TOL = 1e-6, 1e-5
# Number of individual atoms to report explicitly. Full per-atom arrays are
# saved to forces.npz; this is the human-readable sample for 1728+ atom cells.
N_REPORT_ATOMS = 12


def per_atom_force_report(f_uma: np.ndarray, oracle_npz: Path,
                          n_report: int = N_REPORT_ATOMS) -> dict:
    """Per-atom force parity vs the ASE merge oracle.

    A net-force (sum) check only tests translational invariance and passes
    even when every individual force is wrong, so the real gate is the
    per-atom error distribution plus the worst offenders.
    """
    if not oracle_npz.is_file():
        return {"force_parity": "NO_ORACLE", "oracle_f": str(oracle_npz)}
    o = np.load(oracle_npz)
    if "forces" not in o:
        return {"force_parity": "ORACLE_NO_FORCES"}
    f_ref = np.asarray(o["forces"], dtype=np.float64)
    if f_ref.shape != f_uma.shape:
        return {"force_parity": "SHAPE_MISMATCH",
                "uma_shape": list(f_uma.shape), "oracle_shape": list(f_ref.shape)}

    d = f_uma - f_ref
    comp = np.abs(d)
    mag = np.linalg.norm(d, axis=1)
    ref_mag = np.linalg.norm(f_ref, axis=1)
    nz = ref_mag > 1e-8
    rel = np.zeros_like(mag)
    rel[nz] = mag[nz] / ref_mag[nz]
    order = np.argsort(mag)[::-1][:n_report]

    rec = {
        "force_max_abs_component": float(comp.max()),
        "force_mae_component": float(comp.mean()),
        "force_max_per_atom": float(mag.max()),
        "force_mean_per_atom": float(mag.mean()),
        "force_rms_per_atom": float(np.sqrt((mag**2).mean())),
        "force_max_relative": float(rel.max()) if nz.any() else None,
        "force_ref_mag_min": float(ref_mag.min()),
        "force_ref_mag_max": float(ref_mag.max()),
        "force_ref_mag_mean": float(ref_mag.mean()),
        "n_atoms_compared": int(f_ref.shape[0]),
        "n_atoms_over_tol": int((mag > DF_TOL).sum()),
        "worst_atoms": [
            {
                "index": int(i),
                "abs_err": float(mag[i]),
                "rel_err": float(rel[i]),
                "F_uma": [float(x) for x in f_uma[i]],
                "F_oracle": [float(x) for x in f_ref[i]],
            }
            for i in order
        ],
    }
    rec["force_parity"] = "PASS" if rec["force_max_per_atom"] <= DF_TOL else "FAIL"
    return rec


def build_model(task: str, device: str, dtype: str):
    """UMA wrapper at the campaign's FP64 + merge_mole settings."""
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from nvalchemi.models.uma import UMAWrapper

    settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=True,
        compile=False,
        external_graph_gen=False,
    )
    # Precision override. base_precision_dtype is a first-class
    # InferenceSettings field defaulting to float32; fairchem applies it to
    # BOTH the inputs and the model weights (predict.py `self.model.to(...)`),
    # so this is the same knob the campaign's inference_settings_with_dtype
    # helper uses -- not a shallow input cast.
    settings.base_precision_dtype = getattr(torch, dtype)
    wrapper = UMAWrapper.from_checkpoint(
        str(CKPT), task_name=task, device=device, inference_settings=settings,
    )
    eff = wrapper.predict_unit.inference_settings.base_precision_dtype
    if eff is not getattr(torch, dtype):
        raise RuntimeError(
            f"precision override did not stick: asked {dtype}, got {eff}")
    return wrapper


def observed_param_dtype(wrapper) -> str:
    """Dtype actually carried by model weights, for the result record.

    fairchem casts weights lazily on the first predict call, so this is only
    meaningful after a forward pass. Recording it keeps an FP32 fallback from
    being reported as an FP64 result.
    """
    try:
        for p in wrapper.predict_unit.model.parameters():
            if p.is_floating_point():
                return str(p.dtype)
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", dest="sysname", required=True, choices=list(SYS))
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--dtype", default="float64",
                    choices=("float64", "float32"),
                    help="compute precision (default float64; FP32 is for "
                         "A/B speed probes only, never for parity gates)")
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--timestep-fs", type=float, default=1.0)
    ap.add_argument("--thermostat-time-fs", type=float, default=100.0,
                    help="Nose-Hoover tau_T in fs; 100 fs == LAMMPS "
                         "`fix nvt temp 300 300 0.1` (Tdamp 0.1 ps, metal units)")
    a = ap.parse_args()

    cfg = SYS[a.sysname]
    nsteps = a.nsteps if a.nsteps is not None else cfg["nsteps"]
    job = os.environ.get("SLURM_JOB_ID", "manual")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    from ase.io import read
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import NVTNoseHoover

    distributed = a.ngpu > 1
    if distributed:
        from nvalchemi.distributed import (
            DistributedManager, DomainConfig, DomainParallel,
        )
        DistributedManager.initialize()
        dm = DistributedManager()
        device = torch.device(dm.device)
        world, rank = dm.world_size, dm.rank
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    atoms = read(str(cfg["xyz"]))
    assert len(atoms) == cfg["natoms"], f"{len(atoms)} != {cfg['natoms']}"

    model = build_model(cfg["task"], str(device), a.dtype)
    model.model_config.active_outputs = {"energy", "forces"}

    rec: dict = {
        "path": "nvalchemi", "engine": "nvalchemi-toolkit",
        "sys": a.sysname, "natoms": len(atoms), "ngpu": a.ngpu,
        "world_size": world, "dtype": a.dtype, "task": cfg["task"],
        "nsteps": nsteps, "temperature_K": a.temperature,
        "timestep_fs": a.timestep_fs, "thermostat_time_fs": a.thermostat_time_fs,
        "job": job,
        "merge_mole": True, "checkpoint": str(CKPT),
        "integrator": "NVTNoseHoover",
    }

    # ---- single-point E + F (parity frame) --------------------------------
    dt_t = getattr(torch, a.dtype)
    # from_atoms defaults to float32; pass dtype explicitly or FP64
    # parity is silently lost before the model is even called.
    data = AtomicData.from_atoms(atoms, dtype=dt_t)
    batch = Batch.from_data_list([data]).to(device)
    # Hooks take (HookContext, stage), not a bare Batch. For the standalone
    # single-point we drive them through a HookContext at BEFORE_COMPUTE,
    # the same stage dynamics registers them at.
    from nvalchemi.dynamics import DynamicsStage
    from nvalchemi.hooks import HookContext

    nl_hooks = list(model.make_neighbor_hooks())
    ctx = HookContext(batch=batch, model=model, global_rank=0, workflow=None)
    for hook in nl_hooks:
        hook(ctx, DynamicsStage.BEFORE_COMPUTE)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    out = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    rec["sp_ms"] = (time.perf_counter() - t0) * 1e3
    e = float(out["energy"].reshape(-1)[0].item())
    # Record the dtype actually used, so a silent FP32 fallback cannot be
    # mistaken for an FP64 result when comparing against the other paths.
    rec["energy_dtype"] = str(out["energy"].dtype)
    rec["forces_dtype"] = str(out["forces"].dtype)
    rec["param_dtype"] = observed_param_dtype(model)
    rec["precision_ok"] = (
        rec["param_dtype"] == f"torch.{a.dtype}"
        and rec["forces_dtype"] == f"torch.{a.dtype}"
    )
    f = out["forces"].detach().to(torch.float64).cpu().numpy()
    rec["energy_eV"] = e
    rec["force_absmax"] = float(np.abs(f).max())
    # Net force ~0 is only a translational-invariance sanity check: it passes
    # even if every per-atom force is wrong. Keep it, but gate on per-atom.
    rec["force_sum_abs"] = float(np.abs(f.sum(axis=0)).max())
    rec.update(per_atom_force_report(f, ORACLE_F.get(a.sysname, Path("/nonexistent"))))

    oracle = ORACLE_E.get(a.sysname)
    if oracle is not None:
        rec["oracle_E_eV"] = oracle
        rec["dE_vs_merge_ase"] = abs(e - oracle)
        rec["ef_energy_pass"] = rec["dE_vs_merge_ase"] <= DE_TOL
    # E/F parity requires BOTH energy and per-atom forces.
    rec["ef_pass"] = bool(rec.get("ef_energy_pass")
                          and rec.get("force_parity") == "PASS")

    if rank == 0:
        out_dir = EX / "results" / f"{a.sysname}_ngpu{a.ngpu}_{job}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "forces.npz", forces=f, energy_eV=np.array(e))

    # ---- NVT 300 K --------------------------------------------------------
    md_data = AtomicData.from_atoms(atoms, dtype=dt_t)
    md_batch = Batch.from_data_list([md_data]).to(device)
    md_batch.forces = torch.zeros_like(md_batch.positions)
    md_batch.energy = torch.zeros(1, 1, device=device, dtype=md_batch.positions.dtype)
    md_batch.velocities = torch.zeros_like(md_batch.positions)

    # thermostat_time = tau_T in fs. The LAMMPS paths use
    # `fix nvt temp 300 300 0.1` in metal units, i.e. Tdamp = 0.1 ps = 100 fs,
    # so match that exactly rather than taking a default -- thermostat coupling
    # changes the trajectory and would make the NVT comparison unfair.
    integrator = NVTNoseHoover(
        model=model, dt=a.timestep_fs, temperature=a.temperature,
        thermostat_time=a.thermostat_time_fs, n_steps=nsteps,
    )
    try:
        if distributed:
            mesh = DistributedManager().initialize_mesh(
                mesh_shape=(world,), mesh_dim_names=("domain",))
            cut = float(getattr(model, "cutoff", 6.0) or 6.0)
            dcfg = DomainConfig(cutoff=cut, skin=0.5, mesh=mesh)
            with DomainParallel(dynamics=integrator, config=dcfg,
                                n_steps=nsteps) as dyn:
                owned = dyn.partition(md_batch if rank == 0 else None)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                dyn.run(owned)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                md_s = time.perf_counter() - t1
        else:
            for hook in model.make_neighbor_hooks():
                integrator.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)
            with integrator:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                integrator.run(md_batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                md_s = time.perf_counter() - t1
        rec["nvt_ms_per_step"] = md_s * 1e3 / max(1, nsteps)
        rec["nvt_total_s"] = md_s
        rec["nvt_status"] = "OK"
    except Exception as exc:  # noqa: BLE001
        rec["nvt_status"] = f"FAIL: {type(exc).__name__}: {exc}"[:400]

    if device.type == "cuda":
        rec["vram_peak_GiB"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    if rank == 0:
        out_dir = EX / "results" / f"{a.sysname}_ngpu{a.ngpu}_{job}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "timing.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps(rec, indent=2))
        print(f"NVALCHEMI_RECORD {out_dir / 'timing.json'}")

    if distributed:
        from nvalchemi.distributed import DistributedManager as DM
        DM.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
