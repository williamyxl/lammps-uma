#!/usr/bin/env python3
"""P0'.1 step 2: single-tile analytic virial vs finite-difference stress.

Builds a small NaCl cell, runs LAMMPS pair_style uma single-tile, reads the analytic
stress from thermo (pxx..pyz, computed from our strain-autograd virial), then applies
+/- delta strains to the box + fractional coords and finite-differences the energy:
    sigma_ab = -(1/V) dE/deps_ab
PASS if max|sigma_analytic - sigma_FD| <= STRESS_TOL over the 6 Voigt components.

Env: LMP, ART, OUT, N (default 2 -> 64 atoms), DELTA (1e-4), STRESS_TOL (bar).
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

MASS = {"Na": 22.989769, "Cl": 35.453}
BAR_PER_eV_A3 = 1.602176634e6  # 1 eV/A^3 = 1.602e6 bar

# The traced artifacts are rigidly shaped for the N=16 supercell (32768 atoms), so
# the FD system MUST be that exact geometry. Reuse the harness's builder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase6_make_gp_inputs import build_nacl  # noqa: E402


def write_data(path, syms, pos, cell):
    # Orthorhombic only (the single-tile LAMMPS-NL path requires triclinic==0). The
    # FD stress test uses axial strains, which keep the box orthorhombic, so we only
    # ever write diagonal cells here.
    assert abs(cell[1, 0]) < 1e-12 and abs(cell[2, 0]) < 1e-12 and abs(cell[2, 1]) < 1e-12, \
        "virial_fd_check writes orthorhombic cells only"
    lines = ["nacl", "", f"{len(syms)} atoms", "2 atom types", "",
             f"0.0 {cell[0,0]:.16e} xlo xhi", f"0.0 {cell[1,1]:.16e} ylo yhi",
             f"0.0 {cell[2,2]:.16e} zlo zhi", "",
             "Masses", "", f"1 {MASS['Na']}", f"2 {MASS['Cl']}", "", "Atoms # atomic", ""]
    for i, s in enumerate(syms):
        t = 1 if s == "Na" else 2
        lines.append(f"{i+1} {t} {pos[i,0]:.16e} {pos[i,1]:.16e} {pos[i,2]:.16e}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_input(path, art, want_press):
    press = " pxx pyy pzz pxy pxz pyz" if want_press else ""
    txt = f"""units           metal
atom_style      atomic
boundary        p p p
newton          off
read_data       data.nacl
mass            1 22.989769
mass            2 35.453
pair_style      uma precision double
pair_coeff      * * {art} Na Cl
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
thermo          1
thermo_style    custom step pe{press}
thermo_modify   norm no format float %.16e
run             0
"""
    Path(path).write_text(txt)


def run_lmp(d, lmp, want_press):
    write_input(d / "in.sp", os.environ["ART"], want_press)
    ld = os.environ["LD_LIBRARY_PATH"]
    # Only the base (analytic-virial) run needs the strain path; deformed runs are
    # plain single-point energies.
    # UMA_CKPT=0: disable whole-module checkpointing (default ON on XPU) so the
    # strain gradient can propagate for the virial. Also disable for the deformed
    # (energy-only) runs so base and FD use the identical non-checkpointed path.
    vir = "UMA_COMPUTE_VIRIAL=1 UMA_CKPT=0 " if want_press else "UMA_CKPT=0 "
    # UMA_ENGINE_BUILD_GRAPH: forwarded from the caller's env so the engine builds
    # its own (shape-flexible) graph on the plain non-AC module for both base and
    # strained geometries.
    ebg = f'UMA_ENGINE_BUILD_GRAPH={os.environ.get("UMA_ENGINE_BUILD_GRAPH","0")} '
    cmd = (f'mpiexec -n 1 --ppn 1 env UMA_ALLOW_LEGACY_METADATA=1 '
           f'UMA_ALLOW_FAIRCHEM_MISMATCH=1 {vir}{ebg}'
           f'UMA_CHECKPOINT="{os.environ["UMA_CHECKPOINT"]}" LD_LIBRARY_PATH="{ld}" '
           f'gpu_tile_compact.sh {lmp} -in in.sp')
    r = subprocess.run(cmd, shell=True, cwd=str(d),
                       capture_output=True, text=True)
    (d / "lmp.log").write_text(r.stdout + "\n---STDERR---\n" + r.stderr)
    if r.returncode != 0:
        return None
    return parse_thermo(d / "log.lammps", want_press)


def parse_thermo(log, want_press):
    txt = Path(log).read_text()
    m = re.search(r"^\s*Step\s+PotEng.*$", txt, re.M)
    if not m:
        return None
    hdr = m.group(0).split()
    for ln in txt[m.end():].splitlines():
        s = ln.split()
        if len(s) >= len(hdr) and s and s[0] == "0":
            row = {hdr[i]: float(s[i]) for i in range(len(hdr))}
            return row
    return None


def deform(pos0, cell0, eps):
    """Apply strain: cell -> cell(I+eps)^T ; keep fractional coords."""
    F = np.eye(3) + eps
    cell = cell0 @ F.T
    frac = pos0 @ np.linalg.inv(cell0)
    pos = frac @ cell
    return pos, cell


def main():
    lmp = os.environ["LMP"]
    out = Path(os.environ["OUT"])
    n = int(os.environ.get("N", "2"))      # supercell replication -> 8*n^3 atoms
    delta = float(os.environ.get("DELTA", "1e-4"))
    tol_bar = float(os.environ.get("STRESS_TOL", "50.0"))  # bar

    syms, pos0, cell0 = build_nacl(n)
    V = float(abs(np.linalg.det(cell0)))
    base = out / "base"; base.mkdir(parents=True, exist_ok=True)
    write_data(base / "data.nacl", syms, pos0, cell0)
    row = run_lmp(base, lmp, want_press=True)
    if row is None:
        print("FAIL: base LAMMPS run failed", flush=True)
        return 2
    # LAMMPS pressure is -stress; analytic stress tensor (bar):
    sig_an = {
        "xx": -row["Pxx"], "yy": -row["Pyy"], "zz": -row["Pzz"],
        "xy": -row["Pxy"], "xz": -row["Pxz"], "yz": -row["Pyz"],
    }
    print(f"natoms={len(syms)} V={V:.3f} A^3  E0={row['PotEng']:.8f} eV", flush=True)

    # Diagonal (axial) components only: axial strains keep the box orthorhombic,
    # which the single-tile LAMMPS-NL path requires. These validate the virial's
    # normal-stress components (the ones isotropic NPT needs). Shear components need
    # a triclinic path (UMA_ENGINE_BUILD_GRAPH=1) and are left to a follow-on.
    voigt = [("xx", (0, 0)), ("yy", (1, 1)), ("zz", (2, 2))]
    max_err = 0.0
    print(f"{'comp':4} {'sigma_analytic':>16} {'sigma_FD':>16} {'|diff| bar':>14}", flush=True)
    for name, (a, b) in voigt:
        eps_p = np.zeros((3, 3)); eps_m = np.zeros((3, 3))
        # symmetric strain component
        eps_p[a, b] = eps_p[b, a] = delta
        eps_m[a, b] = eps_m[b, a] = -delta
        pp, cp = deform(pos0, cell0, eps_p)
        pm, cm = deform(pos0, cell0, eps_m)
        dp = out / f"p_{name}"; dm = out / f"m_{name}"
        dp.mkdir(exist_ok=True); dm.mkdir(exist_ok=True)
        write_data(dp / "data.nacl", syms, pp, cp)
        write_data(dm / "data.nacl", syms, pm, cm)
        rp = run_lmp(dp, lmp, want_press=False)
        rm = run_lmp(dm, lmp, want_press=False)
        if rp is None or rm is None:
            print(f"FAIL: deformed run failed for {name}", flush=True)
            return 2
        # sigma_ab = (1/V) dE/deps_ab ; off-diagonal symmetric strain adds a factor 2
        fac = 1.0 if a == b else 2.0
        dEde = (rp["PotEng"] - rm["PotEng"]) / (2 * delta) / fac
        sig_fd_bar = (dEde / V) * BAR_PER_eV_A3
        err = abs(sig_an[name] - sig_fd_bar)
        max_err = max(max_err, err)
        print(f"{name:4} {sig_an[name]:16.4f} {sig_fd_bar:16.4f} {err:14.4f}", flush=True)

    ok = max_err <= tol_bar
    print(f"\nmax|sigma_analytic - sigma_FD| = {max_err:.4f} bar (tol {tol_bar}) "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    print("VIRIAL_FD PASS" if ok else "VIRIAL_FD FAIL", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
