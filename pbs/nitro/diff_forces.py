#!/usr/bin/env python3
"""Compare two LAMMPS custom force dumps frame by frame.

Usage: diff_forces.py DUMP_A DUMP_B [--tol-max T] [--require-bitwise]
Dump columns must be: id type fx fy fz (sorted by id).
Per common timestep prints max|dF|, rms|dF|, relative error to rms|F|, and the
number of atoms whose force vector is bitwise identical. Exits 1 when a bar is
violated (--tol-max) or when --require-bitwise sees any difference.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def read_dump(path):
    frames = {}
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1])
            n = int(lines[i + 3])
            hdr = lines[i + 8]
            if not hdr.startswith("ITEM: ATOMS"):
                raise SystemExit(f"FAIL: unexpected header in {path}: {hdr}")
            data = np.array(
                [[float(x) for x in ln.split()] for ln in lines[i + 9 : i + 9 + n]]
            )
            order = np.argsort(data[:, 0])
            frames[step] = data[order]
            i += 9 + n
        else:
            i += 1
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol-max", type=float, default=None)
    ap.add_argument("--require-bitwise", action="store_true")
    args = ap.parse_args()
    fa, fb = read_dump(args.a), read_dump(args.b)
    common = sorted(set(fa) & set(fb))
    if not common:
        raise SystemExit("FAIL: no common timesteps")
    bad = False
    for step in common:
        a, b = fa[step], fb[step]
        if a.shape != b.shape or not np.array_equal(a[:, 0], b[:, 0]):
            raise SystemExit(f"FAIL: atom set differs at step {step}")
        fva, fvb = a[:, 2:5], b[:, 2:5]
        d = fva - fvb
        max_abs = float(np.abs(d).max())
        rms_d = float(np.sqrt((d ** 2).mean()))
        rms_f = float(np.sqrt((fva ** 2).mean()))
        same = int(np.all(fva == fvb, axis=1).sum())
        print(
            f"FORCES step={step} natoms={len(a)} max|dF|={max_abs:.3e} "
            f"rms|dF|={rms_d:.3e} rel={rms_d / rms_f if rms_f else float('nan'):.3e} "
            f"bitwise_atoms={same}"
        )
        if args.tol_max is not None and max_abs > args.tol_max:
            bad = True
        if args.require_bitwise and max_abs != 0.0:
            bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
