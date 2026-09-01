#!/usr/bin/env python3
"""Phase-6 Gate 1 compare: 2-tile GP vs 1-tile (and vs ASE oracle) for NaCl 4x4x4.

Reads step-0 energy (log.lammps) + per-atom forces (forces_step0.dump) from the
W=1 and W=2 runs on identical coords. Gates:
  |E_w2 - E_w1| <= 1e-6 eV, per-atom max|F_w2 - F_w1| <= 1e-5, cos ~ 1  (>=100 atoms)
Also compares W=1 vs ASE (should already be bit-exact from single-tile campaign).

Env: DIR1 (w1 rundir), DIR2 (w2 rundir), UMA_CKPT_FILE, E_TOL, F_TOL, MIN_SAMPLE.
"""
from __future__ import annotations
import os, re, sys
from pathlib import Path
import numpy as np

HEN = Path("/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen")
for p in (HEN / "shim", HEN / "patches", HEN):
    if p.is_dir(): sys.path.insert(0, str(p))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uma_gates  # P1.5: single source of truth for tolerances


def read_forces(path, nat):
    lines = Path(path).read_text().splitlines()
    f = np.zeros((nat, 3))
    for i, ln in enumerate(lines):
        if ln.startswith("ITEM: ATOMS"):
            cols = ln.split()[2:]; ix = {c: k for k, c in enumerate(cols)}
            for j in range(nat):
                v = lines[i + 1 + j].split()
                a = int(v[ix["id"]]) - 1
                f[a] = [float(v[ix["fx"]]), float(v[ix["fy"]]), float(v[ix["fz"]])]
            break
    return f


def pe0(log):
    txt = Path(log).read_text()
    m = re.search(r"^\s*Step\b.*PotEng.*$", txt, re.M)
    if not m: return None
    hdr = m.group(0).split(); pe = hdr.index("PotEng")
    for ln in txt[m.end():].splitlines():
        s = ln.split()
        if len(s) >= len(hdr) and s[0] == "0":
            return float(s[pe])
    return None


def stats(fa, fb, idx):
    d = np.abs(fa[idx] - fb[idx])
    a, b = fa[idx].ravel(), fb[idx].ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    return float(d.max()), float(np.sqrt(np.mean(d**2))), cos


def main():
    d1 = Path(os.environ["DIR1"]); d2 = Path(os.environ["DIR2"])
    # P1.5: tolerances from the shared uma_gates table. This gate compares two
    # LAMMPS runs on the SAME small system (W=1 vs W=2), so it uses an absolute
    # energy tolerance derived from the per-atom gate (nat is small here).
    f_tol = uma_gates.f_tol()
    nsamp = uma_gates.min_sample()
    e_tol = float(os.environ.get("E_TOL", "1e-6"))
    syms = (d1 / "symbols.txt").read_text().split(); nat = len(syms)
    idx = np.array(sorted(set(np.linspace(0, nat - 1, min(nsamp, nat)).astype(int)) | {0, nat - 1}))

    e1, e2 = pe0(d1 / "log.lammps"), pe0(d2 / "log.lammps")
    # P1.2: a missing step-0 energy (crashed/failed run) must FAIL the gate, not
    # crash ambiguously or slip through.
    if e1 is None or e2 is None:
        print(f"GATE1 FAIL: missing step-0 energy (e_w1={e1}, e_w2={e2})")
        return 2
    f1 = read_forces(d1 / "forces_step0.dump", nat)
    f2 = read_forces(d2 / "forces_step0.dump", nat)

    dE = abs(e2 - e1)
    mx, rms, cos = stats(f2, f1, idx)
    ok_gp = (dE <= e_tol) and (mx <= f_tol)
    print(f"[GP 2-tile vs 1-tile] natoms={nat} sampled={len(idx)}")
    print(f"  E_w1={e1:.10f} E_w2={e2:.10f} dE={dE:.3e} eV")
    print(f"  max|dF|={mx:.3e} rms|dF|={rms:.3e} cos={cos:.10f}  -> {'PASS' if ok_gp else 'FAIL'}")

    # W=1 vs ASE oracle.
    # P1.2: fail closed. Default ok_ase=False so that if the oracle cannot run the
    # gate does NOT silently PASS. Set ALLOW_SKIP=1 to explicitly permit skipping
    # the ASE cross-check (e.g. on a node without fairchem) — then the GP-vs-1tile
    # comparison alone gates.
    allow_skip = os.environ.get("ALLOW_SKIP", "0") == "1"
    ok_ase = False
    try:
        import torch
        from dataclasses import replace
        from ase import Atoms
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings
        from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
        from fairchem_xpu_parallel import patch_fairchem_xpu_device
        patch_fairchem_xpu_device()
        pos = np.load(d1 / "positions.npy"); cell = np.load(d1 / "cell.npy")
        s = guess_inference_settings("default")
        s = replace(s, base_precision_dtype=torch.float64, tf32=False, compile=False)
        unit = MLIPPredictUnit(os.environ["UMA_CKPT_FILE"], device="xpu", inference_settings=s)
        for a in ("model", "module", "_module"):
            m = getattr(unit, a, None)
            if m is not None: m.double(); break
        calc = FAIRChemCalculator(unit, task_name="omat")
        at = Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)
        at.info["charge"] = 0; at.info["spin"] = 0; at.calc = calc
        e_ase = float(at.get_potential_energy()); f_ase = np.asarray(at.get_forces(), float)
        dE1 = abs(e1 - e_ase); mx1, rms1, cos1 = stats(f1, f_ase, idx)
        ok_ase = (dE1 <= e_tol) and (mx1 <= f_tol)
        print(f"[1-tile vs ASE] dE={dE1:.3e} eV max|dF|={mx1:.3e} cos={cos1:.10f} -> {'PASS' if ok_ase else 'FAIL'}")
    except Exception as exc:
        if allow_skip:
            print(f"[1-tile vs ASE] SKIP allowed via ALLOW_SKIP=1 "
                  f"({type(exc).__name__}: {exc})")
            ok_ase = True  # explicitly opted out of the ASE cross-check
        else:
            print(f"[1-tile vs ASE] FAIL: oracle could not run "
                  f"({type(exc).__name__}: {exc}); set ALLOW_SKIP=1 to bypass")
            ok_ase = False

    ok = ok_gp and ok_ase
    print(f"\nGATE1 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
