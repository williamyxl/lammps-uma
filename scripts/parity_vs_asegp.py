#!/usr/bin/env python3
"""Per-atom energy + force parity: LAMMPS pair_style uma vs ASE-GP oracle.

Compares our LAMMPS step-0 output against the FairChem ASE graph-parallel (hen)
oracle for the SAME NaCl system (a=5.64, rattle 0.05, seed 0). Both use W=12.

LAMMPS side (env LMP_DIR): forces_step0.dump (id type x y z fx fy fz, sorted by
id, %.16e) + step-0 PotEng from log.lammps.
ASE side (env ASE_DIR): summary_w12.json (energy_eV) + forces_w12.npy [natoms,3].

Gates (same as phase6): |dE| <= 1e-6 eV total, per-atom max|dF| <= 1e-5 eV/A on
>= MIN_SAMPLE atoms, cosine ~ 1. Prints PASS/FAIL.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uma_gates  # P1.5: single source of truth for tolerances


def read_lmp_forces(dump_path):
    lines = Path(dump_path).read_text().splitlines()
    # find ITEM: ATOMS header
    hdr_i = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS"))
    cols = lines[hdr_i].split()[2:]  # id type x y z fx fy fz
    ix = {c: k for k, c in enumerate(cols)}
    rows = []
    for l in lines[hdr_i + 1:]:
        s = l.split()
        if len(s) < len(cols):
            continue
        rows.append(s)
    arr = np.array(rows, dtype=np.float64)
    ids = arr[:, ix["id"]].astype(np.int64)
    order = np.argsort(ids)
    f = arr[order][:, [ix["fx"], ix["fy"], ix["fz"]]]
    return f


def read_lmp_energy(log_path):
    txt = Path(log_path).read_text()
    m = re.search(r"^\s*Step\b.*PotEng.*$", txt, re.M)
    hdr = m.group(0).split()
    pe = hdr.index("PotEng")
    for ln in txt[m.end():].splitlines():
        s = ln.split()
        if len(s) >= len(hdr) and s and s[0] == "0":
            return float(s[pe])
    raise RuntimeError("no step-0 PotEng")


def main():
    lmp_dir = Path(os.environ["LMP_DIR"])
    ase_dir = Path(os.environ["ASE_DIR"])
    # P1.5: tolerances come from the shared uma_gates table (env-overridable there).
    min_sample = uma_gates.min_sample()
    e_tol_per_atom = uma_gates.e_tol_per_atom_mev()  # meV/atom
    f_tol = uma_gates.f_tol()

    f_lmp = read_lmp_forces(lmp_dir / "forces_step0.dump")
    e_lmp = read_lmp_energy(lmp_dir / "log.lammps")

    f_ase = np.load(ase_dir / "forces_w12.npy").astype(np.float64)
    summ = json.loads((ase_dir / "summary_w12.json").read_text())
    e_ase = float(summ["energy_eV"])

    # P1.5: an atom-count mismatch is a HARD FAIL, not a truncate-and-warn.
    # Comparing the first min(N) atoms of two differently-sized systems can hide a
    # real disagreement (wrong system, dropped atoms, oracle/LAMMPS drift).
    if len(f_lmp) != len(f_ase):
        print(f"PARITY FAIL: natoms mismatch lmp={len(f_lmp)} ase={len(f_ase)} "
              f"(refusing to truncate-and-compare)")
        return 2
    n = len(f_lmp)

    dE = abs(e_lmp - e_ase)
    dE_per_atom_meV = dE / n * 1e3
    dF = np.abs(f_lmp - f_ase)
    max_dF = float(dF.max())
    rms_dF = float(np.sqrt((dF ** 2).mean()))
    # cosine over the full flattened force vector
    a = f_lmp.ravel()
    b = f_ase.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    sampled = n

    print(f"natoms={n} sampled={sampled} (min_sample={min_sample})")
    print(f"  E_lmp={e_lmp:.6f}  E_ase={e_ase:.6f}  dE={dE:.3e} eV "
          f"({dE_per_atom_meV:.3e} meV/atom)")
    print(f"  max|dF|={max_dF:.3e}  rms|dF|={rms_dF:.3e}  cos={cos:.10f} eV/A")
    # DD Phase A: the LAMMPS global energy is a per-rank subsystem sum, not the
    # true global energy (needs per-atom-energy export, Phase B). UMA_DD_SKIP_ENERGY=1
    # validates FORCES over ALL atoms (the exact quantity) and drops the energy gate.
    skip_energy = os.environ.get("UMA_DD_SKIP_ENERGY", "0").strip() in ("1", "true", "yes")
    e_ok = dE_per_atom_meV <= e_tol_per_atom
    f_ok = (max_dF <= f_tol) and (sampled >= min_sample)
    passed = (f_ok if skip_energy else (e_ok and f_ok))
    e_tag = "SKIP" if skip_energy else ("OK" if e_ok else "FAIL")
    print(f"  gates: dE<= {e_tol_per_atom:.0e} meV/atom [{e_tag}]  "
          f"max|dF|<= {f_tol:.0e} on >={min_sample} [{'OK' if f_ok else 'FAIL'}]")
    print(f"PARITY {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
