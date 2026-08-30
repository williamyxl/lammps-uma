#!/usr/bin/env python3
"""Per-atom force parity for any path's forces.npz vs the ASE merge oracle.

A net-force check (|sum F| ~ 0) only tests translational invariance and will
pass even when every individual atom's force is wrong. This reports the
per-atom error distribution and names the worst atoms explicitly.

  python force_parity.py --forces <path>/forces.npz --sys nacl6
  python force_parity.py --forces A.npz --compare B.npz --sys nacl6 --atoms 0,5,17
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")
CAMP = ROOT / "src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign"
ORACLE = {
    "nacl6": CAMP / "oracle_ase_umas_fast_merge.npz",
    "water888": CAMP / "oracle_ase_water_merge.npz",
}
DF_TOL = 1e-5


def load_forces(p: Path) -> np.ndarray:
    d = np.load(p)
    for k in ("forces", "f", "arr_0"):
        if k in d:
            return np.asarray(d[k], dtype=np.float64)
    raise SystemExit(f"no forces array in {p} (keys: {list(d.keys())})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forces", required=True, help="forces.npz under test")
    ap.add_argument("--sys", dest="sysname", choices=list(ORACLE), default=None)
    ap.add_argument("--compare", default=None,
                    help="reference forces.npz (default: ASE merge oracle for --sys)")
    ap.add_argument("--atoms", default=None,
                    help="comma-separated atom indices to print explicitly")
    ap.add_argument("--top", type=int, default=12, help="worst-N atoms to print")
    ap.add_argument("--tol", type=float, default=DF_TOL)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    f = load_forces(Path(a.forces))
    if a.compare:
        ref_path = Path(a.compare)
    elif a.sysname:
        ref_path = ORACLE[a.sysname]
    else:
        raise SystemExit("need --sys or --compare")
    ref = load_forces(ref_path)
    if f.shape != ref.shape:
        raise SystemExit(f"shape mismatch: {f.shape} vs {ref.shape}")

    d = f - ref
    comp = np.abs(d)
    mag = np.linalg.norm(d, axis=1)
    ref_mag = np.linalg.norm(ref, axis=1)
    nz = ref_mag > 1e-8
    rel = np.zeros_like(mag)
    rel[nz] = mag[nz] / ref_mag[nz]
    over = int((mag > a.tol).sum())

    summary = {
        "under_test": str(a.forces),
        "reference": str(ref_path),
        "n_atoms": int(f.shape[0]),
        "max_abs_component": float(comp.max()),
        "mae_component": float(comp.mean()),
        "max_per_atom": float(mag.max()),
        "mean_per_atom": float(mag.mean()),
        "rms_per_atom": float(np.sqrt((mag**2).mean())),
        "max_relative": float(rel.max()) if nz.any() else None,
        "n_atoms_over_tol": over,
        "tol": a.tol,
        "verdict": "PASS" if mag.max() <= a.tol else "FAIL",
    }

    if a.json:
        print(json.dumps(summary, indent=2))
        return 0 if summary["verdict"] == "PASS" else 1

    print(f"under test : {a.forces}")
    print(f"reference  : {ref_path}")
    print(f"atoms      : {f.shape[0]}")
    print(f"\nforce magnitude (reference): {ref_mag.min():.4f} .. {ref_mag.max():.4f} "
          f"eV/A  (mean {ref_mag.mean():.4f})")
    print("\n=== PER-ATOM FORCE ERROR ===")
    print(f"  max |dF| component : {comp.max():.6e} eV/A")
    print(f"  mean|dF| component : {comp.mean():.6e} eV/A")
    print(f"  max  per-atom |dF| : {mag.max():.6e} eV/A")
    print(f"  mean per-atom |dF| : {mag.mean():.6e} eV/A")
    print(f"  RMS  per-atom |dF| : {np.sqrt((mag**2).mean()):.6e} eV/A")
    if nz.any():
        print(f"  max relative error : {rel.max():.3e}")
    print(f"  atoms over tol {a.tol:g}: {over} / {f.shape[0]}")

    idx = np.argsort(mag)[::-1][:a.top]
    label = f"{a.top} WORST ATOMS"
    if a.atoms:
        idx = np.array([int(x) for x in a.atoms.split(",")])
        label = "SELECTED ATOMS"
    print(f"\n=== {label} ===")
    print(f"{'idx':>6} {'|dF|':>12} {'rel':>10} "
          f"{'F_test (x,y,z)':>34} {'F_ref (x,y,z)':>34}")
    for i in idx:
        print(f"{i:>6} {mag[i]:>12.4e} {rel[i]:>10.2e} "
              f"({f[i, 0]:>9.5f},{f[i, 1]:>9.5f},{f[i, 2]:>9.5f}) "
              f"({ref[i, 0]:>9.5f},{ref[i, 1]:>9.5f},{ref[i, 2]:>9.5f})")

    print(f"\nVERDICT: {summary['verdict']}  (tol {a.tol:g} eV/A on per-atom |dF|)")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
