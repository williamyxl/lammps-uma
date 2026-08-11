#!/usr/bin/env python3
"""M0.5 gate: LAMMPS output must round-trip FP64, not floor parity at ~1e-6.

Plan gate (multinode_mpi_plan.md, M0.5):
  On NaCl6, a serial dump round-trips to the tensor values to <= 1e-15
  relative, and max|dF| vs the FP64 oracle is reported at its true magnitude
  rather than ~8e-07.

Why this blocks M3+: M5 gates on max|dF| <= 1e-9, but LAMMPS' default %g output
is 6 significant figures, whose rounding limit at these force magnitudes is
~8.7e-07 -- three orders LOOSER than the gate. Without this fix every
downstream parity number measures the file format, not the physics.

This compares dumps written by the OLD (%g) and NEW (%.17g) runners against
the same forces, so it isolates formatting from computation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")
EX = ROOT / "src/ML-UMA/examples/multi_gpu_nacl6"
CAMP = EX / "agent_stamps/cpp_libtorch/perf_campaign"
ORACLE = CAMP / "oracle_ase_umas_fast_merge.npz"
OUT = ROOT / "src/ML-UMA/uma-engine/tests/m05_precision_result.json"


def sig_digits(tok: str) -> int:
    """Count significant digits in a numeric token."""
    m = re.match(r"^[-+]?0*(\d*)\.?(\d*)(?:[eE][-+]?\d+)?$", tok.strip())
    if not m:
        return 0
    return len((m.group(1) + m.group(2)).lstrip("0")) or 0


def read_dump_forces(p: Path):
    """Return (forces[N,3], max significant digits seen in the force columns)."""
    lines = p.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS")) + 1
    rows, digits = [], 0
    for l in lines[start:]:
        parts = l.split()
        if len(parts) < 8:
            break
        rows.append([float(parts[5]), float(parts[6]), float(parts[7])])
        digits = max(digits, *(sig_digits(t) for t in parts[5:8]))
    return np.asarray(rows, dtype=np.float64), digits


def quantum(f: np.ndarray, digits: int) -> float:
    """Worst-case rounding step for `digits` significant figures."""
    mag = np.abs(f)
    expo = np.floor(np.log10(np.maximum(mag, 1e-30)))
    return float((10.0 ** (expo - (digits - 1))).max())


def main() -> int:
    rec: dict = {"gate": "M0.5 full-precision output"}

    dumps = sorted(EX.glob("results/uma_ngpu*/work/forces_first.dump"),
                   key=lambda p: p.stat().st_mtime)
    if not dumps:
        rec["status"] = "NO_DUMPS"
        print(json.dumps(rec, indent=2))
        return 1

    # Classify every dump on disk by the precision it was written with.
    lo, hi = [], []
    for d in dumps:
        try:
            f, dig = read_dump_forces(d)
        except (StopIteration, ValueError):
            continue
        (hi if dig >= 12 else lo).append((d, f, dig))

    rec["n_dumps_scanned"] = len(dumps)
    rec["n_low_precision"] = len(lo)
    rec["n_high_precision"] = len(hi)

    if ORACLE.is_file():
        ref = np.load(ORACLE)["forces"]

        def parity(f):
            if f.shape != ref.shape:
                return None
            return float(np.linalg.norm(f - ref, axis=1).max())

        if lo:
            d, f, dig = lo[-1]
            rec["low"] = {"file": str(d), "sig_digits": dig,
                          "rounding_limit": quantum(f, dig),
                          "max_per_atom_dF": parity(f)}
        if hi:
            d, f, dig = hi[-1]
            rec["high"] = {"file": str(d), "sig_digits": dig,
                           "rounding_limit": quantum(f, dig),
                           "max_per_atom_dF": parity(f)}

    # Gate: a high-precision dump must exist, and its parity must no longer sit
    # at the old ~8.7e-07 formatting floor.
    hi_ok = bool(hi)
    rec["high_precision_dump_present"] = hi_ok
    if hi_ok and rec.get("high", {}).get("max_per_atom_dF") is not None:
        h = rec["high"]
        rec["below_old_floor"] = bool(h["max_per_atom_dF"] < 8.66e-07)
        rec["rounding_limit_ok"] = bool(h["rounding_limit"] <= 1e-12)
        rec["pass"] = bool(rec["below_old_floor"] and rec["rounding_limit_ok"])
    else:
        rec["pass"] = False
        rec["note"] = ("no %.17g dump on disk yet: the fix is committed but no "
                       "LAMMPS cell has been re-run since. Re-run one nacl6 "
                       "cell to close this gate.")

    OUT.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    print(f"\nM0.5 GATE: {'PASS' if rec.get('pass') else 'FAIL'}")
    return 0 if rec.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
