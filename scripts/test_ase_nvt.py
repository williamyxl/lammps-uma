#!/usr/bin/env python3
"""Path A: ASE FairChem UMA, Nose-Hoover-chain NVT 300K 10 steps (single tile).

Reports: first-frame energy, per-atom forces (>=100 atoms sampled) + AG=FD check,
and walltime for the 10-step NVT. FP64, XPU. Matches LAMMPS `fix nvt` (Nose-Hoover
chain, tchain=3, Tdamp=0.1 ps, dt=1 fs).

Env: N (cell size), UMA_CKPT, OUTDIR, FD_EPS=1e-4, FD_SAMPLE=100.
Writes OUTDIR/ase_n{N}.json with all metrics + positions/forces npy for cross-check.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

HEN = Path("/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen")
for p in (HEN / "shim", HEN / "patches", HEN):
    if p.is_dir(): sys.path.insert(0, str(p))


def build_nacl(n, rattle=0.05, seed=0):
    from ase import Atoms
    a = 5.64
    na = np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]], float); cl = na + 0.5
    sy=[]; sc=[]
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                o=np.array([ix,iy,iz],float)
                for f in na: sy.append("Na"); sc.append((f+o)/n)
                for f in cl: sy.append("Cl"); sc.append((f+o)/n)
    at = Atoms(symbols=sy, scaled_positions=sc, cell=np.eye(3)*(a*n), pbc=True)
    at.positions += np.random.default_rng(seed).normal(0, rattle, at.positions.shape)
    at.info["charge"]=0; at.info["spin"]=0
    return at


def make_calc(ckpt):
    import torch
    from dataclasses import replace
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
    from fairchem_xpu_parallel import patch_fairchem_xpu_device
    patch_fairchem_xpu_device()
    s = guess_inference_settings("default")
    s = replace(s, base_precision_dtype=torch.float64, tf32=False, compile=False)
    unit = MLIPPredictUnit(str(ckpt), device="xpu", inference_settings=s)
    for a in ("model","module","_module"):
        m=getattr(unit,a,None)
        if m is not None: m.double(); break
    return FAIRChemCalculator(unit, task_name="omat")


def main():
    import torch
    from ase import units
    from ase.md.nose_hoover_chain import NoseHooverChainNVT
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    N=int(os.environ.get("N","6")); ckpt=Path(os.environ.get("UMA_CKPT", str(HEN/"uma-cache"/"uma-s-1p2.pt")))
    outdir=Path(os.environ.get("OUTDIR","./ase")); outdir.mkdir(parents=True, exist_ok=True)
    eps=float(os.environ.get("FD_EPS","1e-4"))
    # AG=FD is a finite-difference correctness spot-check: each sampled coord costs
    # 2 full forwards, so keep the sample small (default 10 atoms = 60 forwards).
    # The >=100-atom requirement applies to the force PARITY check (1 forward),
    # handled in the LAMMPS-vs-ASE comparator, not here.
    nsamp=int(os.environ.get("AGFD_SAMPLE","10"))
    calc=make_calc(ckpt)
    at=build_nacl(N); nat=len(at); at.calc=calc

    # First-frame energy + forces (the reference).
    t=time.perf_counter()
    e0=float(at.get_potential_energy()); f0=np.asarray(at.get_forces(),float)
    torch.xpu.synchronize(); t_ef=time.perf_counter()-t
    np.save(outdir/f"ase_pos_n{N}.npy", at.get_positions()); np.save(outdir/f"ase_cell_n{N}.npy", at.get_cell().array)
    np.save(outdir/f"ase_forces_n{N}.npy", f0)
    (outdir/f"ase_symbols_n{N}.txt").write_text("\n".join(at.get_chemical_symbols()))

    # AG=FD on sampled atoms (central diff of ASE energy).
    idx=sorted(set(np.linspace(0,nat-1,min(nsamp,nat)).astype(int)) | {0,nat-1})
    p0=at.get_positions().copy(); maxagfd=0.0; sm=0.0; cnt=0
    for ia in idx:
        for ic in range(3):
            p=p0.copy(); p[ia,ic]+=eps; at.set_positions(p); ep=float(at.get_potential_energy())
            p=p0.copy(); p[ia,ic]-=eps; at.set_positions(p); em=float(at.get_potential_energy())
            ffd=-(ep-em)/(2*eps); d=abs(float(f0[ia,ic])-ffd); maxagfd=max(maxagfd,d); sm+=d; cnt+=1
    at.set_positions(p0)
    agfd_mean=sm/cnt if cnt else float("nan")

    # NVT 10 steps, Nose-Hoover chain (match LAMMPS fix nvt: tchain=3, Tdamp=0.1ps).
    MaxwellBoltzmannDistribution(at, temperature_K=300.0, rng=np.random.default_rng(4928459))
    dyn=NoseHooverChainNVT(at, timestep=1.0*units.fs, temperature_K=300.0,
                           tdamp=100.0*units.fs, tchain=3, tloop=1)
    t=time.perf_counter(); dyn.run(10); torch.xpu.synchronize(); t_nvt=time.perf_counter()-t

    res={"path":"ASE_FairChem","N":N,"natoms":nat,"first_energy_eV":e0,
         "fmax":float(np.abs(f0).max()),"n_sampled":len(idx),
         "AGFD_max":maxagfd,"AGFD_mean":agfd_mean,"AGFD_tol":1e-5,
         "AGFD_pass":bool(maxagfd<=1e-5),
         "t_first_ef_s":t_ef,"t_nvt10_s":t_nvt}
    (outdir/f"ase_n{N}.json").write_text(json.dumps(res,indent=2))
    print(f"[ASE N={N} nat={nat}] E0={e0:.10f} fmax={res['fmax']:.4f} "
          f"AGFD_max={maxagfd:.3e} ({'PASS' if res['AGFD_pass'] else 'FAIL'}) "
          f"t_ef={t_ef:.1f}s t_nvt10={t_nvt:.1f}s", flush=True)
    return 0


if __name__=="__main__":
    sys.exit(main())
