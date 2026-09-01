#!/usr/bin/env python3
"""Phase-5 parity: LAMMPS pair_style uma (XPU eager) first-frame energy + per-atom
forces vs ASE UMA oracle on the IDENTICAL perturbed NaCl coordinates.

Reads:
  <dir>/positions.npy, cell.npy, symbols.txt   (exact coords used by LAMMPS)
  <dir>/forces_step0.dump                        (LAMMPS step-0 per-atom forces)
  <dir>/log.lammps                               (LAMMPS PE at step 0)
Computes ASE oracle (FAIRChemCalculator + MLIPPredictUnit, xpu, fp64) on the same
coords. Gates |dE|<=1e-6 eV total, per-atom max|dF|<=1e-5 (>=100 atoms sampled).

Env: DIR (the nacl_n<N> dir), UMA_CKPT_FILE, MIN_SAMPLE=100, E_TOL=1e-6, F_TOL=1e-5
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

HEN = Path("/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen")
for p in (HEN / "shim", HEN / "patches", HEN):
    if p.is_dir():
        sys.path.insert(0, str(p))


def read_lammps_dump_forces(path, nat):
    lines = Path(path).read_text().splitlines()
    i = 0
    fx = np.zeros((nat, 3))
    while i < len(lines):
        if lines[i].startswith("ITEM: ATOMS"):
            cols = lines[i].split()[2:]
            ix = {c: k for k, c in enumerate(cols)}
            for j in range(nat):
                v = lines[i + 1 + j].split()
                aid = int(v[ix["id"]]) - 1
                fx[aid, 0] = float(v[ix["fx"]])
                fx[aid, 1] = float(v[ix["fy"]])
                fx[aid, 2] = float(v[ix["fz"]])
            break
        i += 1
    return fx


def lammps_pe_step0(log_path):
    txt = Path(log_path).read_text()
    # find thermo header then the first data row (step 0)
    m = re.search(r"^\s*Step\s+Temp\s+PotEng.*$", txt, re.M)
    if not m:
        # header uses custom names: Step Temp PotEng TotEng Press Fmax
        m = re.search(r"^\s*Step\b.*PotEng.*$", txt, re.M)
    if not m:
        return None
    hdr = m.group(0).split()
    peidx = hdr.index("PotEng")
    after = txt[m.end():].splitlines()
    for ln in after:
        s = ln.split()
        if len(s) >= len(hdr) and s[0] == "0":
            return float(s[peidx])
    return None


def main():
    d = Path(os.environ["DIR"])
    ckpt = Path(os.environ.get("UMA_CKPT_FILE", str(HEN / "uma-cache" / "uma-s-1p2.pt")))
    e_tol = float(os.environ.get("E_TOL", "1e-6"))
    f_tol = float(os.environ.get("F_TOL", "1e-5"))
    min_sample = int(os.environ.get("MIN_SAMPLE", "100"))

    pos = np.load(d / "positions.npy")
    cell = np.load(d / "cell.npy")
    syms = (d / "symbols.txt").read_text().split()
    nat = len(syms)

    f_lmp = read_lammps_dump_forces(d / "forces_step0.dump", nat)
    e_lmp = lammps_pe_step0(d / "log.lammps")

    # ASE oracle on identical coords
    import torch
    from dataclasses import replace
    from ase import Atoms
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
    from fairchem_xpu_parallel import patch_fairchem_xpu_device

    patch_fairchem_xpu_device()
    settings = guess_inference_settings("default")
    settings = replace(settings, base_precision_dtype=torch.float64, tf32=False, compile=False)
    unit = MLIPPredictUnit(str(ckpt), device="xpu", inference_settings=settings)
    for attr in ("model", "module", "_module"):
        mod = getattr(unit, attr, None)
        if mod is not None:
            mod.double(); break
    calc = FAIRChemCalculator(unit, task_name="omat")

    atoms = Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0
    atoms.calc = calc
    e_ase = float(atoms.get_potential_energy())
    f_ase = np.asarray(atoms.get_forces(), dtype=np.float64)

    # sample >=100 atoms
    want = min(max(min_sample, 1), nat)
    idx = np.array(sorted(set(np.linspace(0, nat - 1, want).astype(int)) | {0, nat - 1}))

    dE = abs(e_lmp - e_ase) if e_lmp is not None else float("nan")
    dF = np.abs(f_lmp[idx] - f_ase[idx])
    max_dF = float(dF.max())
    rms_dF = float(np.sqrt(np.mean(dF**2)))
    a, b = f_lmp[idx].ravel(), f_ase[idx].ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    ok = (e_lmp is not None) and (dE <= e_tol) and (max_dF <= f_tol)

    print(f"natoms={nat} sampled={len(idx)}")
    print(f"E_LAMMPS={e_lmp} E_ASE={e_ase:.10f} dE={dE:.3e} eV ({1e3*dE/nat:.3e} meV/atom)")
    print(f"max|dF|={max_dF:.3e} rms|dF|={rms_dF:.3e} cos={cos:.10f}")
    print(f"gates: |dE|<={e_tol} max|dF|<={f_tol} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
