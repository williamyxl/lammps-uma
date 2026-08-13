#!/usr/bin/env python3
"""Report LAMMPS+checkpointing sweep: energy, per-atom force, NVT timing, and
parity vs the ASE-FC checkpointing reference, for each N."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def parse_lmp(outdir: Path, N: int):
    txt = ""
    for p in (outdir / f"out.{N}", outdir / f"log.{N}"):
        if p.is_file():
            txt += p.read_text(errors="ignore") + "\n"
    e = None
    for m in re.finditer(rf"SP_PE_{N} = (-?\d[\d.eE+-]*)", txt):
        e = float(m.group(1))
    # NVT timing (last Loop time block)
    ms = None
    for m in re.finditer(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+\d+\s+procs for\s+(\d+)\s+steps", txt):
        loop_s, n = float(m.group(1)), int(m.group(2))
        if n > 0:
            ms = loop_s / n * 1e3
    # forces
    f = None
    dump = outdir / f"lmp_f_{N}.dump"
    if dump.is_file():
        L = dump.read_text().splitlines()
        s = next(i for i, l in enumerate(L) if l.startswith("ITEM: ATOMS"))
        cols = L[s].split()[2:]; iid, ifx = cols.index("id"), cols.index("fx")
        rows = []
        for l in L[s + 1:]:
            if l.startswith("ITEM:"):
                break
            p = l.split()
            if len(p) >= len(cols):
                rows.append((float(p[iid]), float(p[ifx]), float(p[ifx + 1]), float(p[ifx + 2])))
        arr = np.array(rows); arr = arr[np.argsort(arr[:, 0])]; f = arr[:, 1:4]
    oom = "out of memory" in txt.lower()
    return e, f, ms, oom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    args = ap.parse_args()
    d = Path(args.outdir)

    rows = []
    for N in args.sizes:
        nat = 8 * N**3
        e_l, f_l, ms, oom = parse_lmp(d, N)
        ase = d / f"ase_{N}.npz"
        e_a = f_a = None
        if ase.is_file():
            z = np.load(ase); e_a = float(z["energy_eV"]); f_a = z["forces"].astype(np.float64)
        dE = (abs(e_l - e_a) if (e_l is not None and e_a is not None) else None)
        dF = (float(np.linalg.norm(f_l - f_a, axis=1).max())
              if (f_l is not None and f_a is not None and f_l.shape == f_a.shape) else None)
        fmax = (float(np.abs(f_l).max()) if f_l is not None else None)
        rows.append((N, nat, e_l, e_a, dE, dF, fmax, ms, oom))

    lines = [
        "# LAMMPS + activation checkpointing (4 GPU eager GP), NaCl NxNxN, FP64",
        "",
        "| N | atoms | E_lammps (eV) | E_ase (eV) | |dE| | max|dF| | |F|max | NVT ms/step | status |",
        "|--:|------:|--------------:|-----------:|-----:|--------:|-------:|------------:|:------:|",
    ]
    for (N, nat, el, ea, dE, dF, fmax, ms, oom) in rows:
        def g(x, f="{:.3e}"):
            return "-" if x is None else f.format(x)
        status = "OOM" if oom else ("OK" if el is not None else "FAIL")
        lines.append(
            f"| {N} | {nat} | {g(el,'{:.6f}')} | {g(ea,'{:.6f}')} | {g(dE)} | "
            f"{g(dF)} | {g(fmax,'{:.4f}')} | {g(ms,'{:.1f}')} | {status} |"
        )
    report = "\n".join(lines) + "\n"
    (d / "LMP_CKPT_REPORT.md").write_text(report)
    print(report)
    print(f"wrote {d/'LMP_CKPT_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
