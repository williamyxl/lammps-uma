#!/usr/bin/env python3
"""Phase-3: compare C++ engine (uma_parity_cli) energy + per-atom forces vs the
ASE UMA oracle on the perturbed NaCl cells.

Reads oracle_manifest.json (energy + oracle_forces.npy per N) and the C++
outputs (energy parsed from the CLI stdout log, forces from --write-forces
binary). Gates:
    |E_cpp - E_oracle| <= E_TOL (default 1e-6 eV)
    max|F_cpp - F_oracle| <= F_TOL (default 1e-5 eV/Ang), >=100 atoms sampled

Env: OUTDIR, E_TOL=1e-6, F_TOL=1e-5, MIN_SAMPLE=100
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# S8/G11 (audit rev18): tolerances come from the single source scripts/uma_gates.py
# (P1.5), not a local copy that can silently diverge when uma_gates.py is edited.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import uma_gates

OUTDIR = Path(os.environ.get("OUTDIR", "./phase3"))
E_TOL = float(os.environ.get("E_TOL", "1e-6"))  # this gate uses an absolute eV E-tol
F_TOL = uma_gates.f_tol()
MIN_SAMPLE = uma_gates.min_sample()


def parse_cpp_energy(log_path: Path):
    if not log_path.is_file():
        return None
    m = re.search(r"energy=([-\d.eE+]+)\s*eV", log_path.read_text())
    return float(m.group(1)) if m else None


def sample_idx(nat, want):
    want = min(max(want, 1), nat)
    if want >= nat:
        return np.arange(nat)
    stride = nat / want
    idx = sorted({int(i * stride) for i in range(want)} | {0, nat - 1})
    return np.array(idx)


def main() -> int:
    manifest = json.loads((OUTDIR / "oracle_manifest.json").read_text())
    rows = []
    all_ok = True
    for m in manifest:
        n = m["n"]
        if not m.get("oracle_ok"):
            rows.append({"n": n, "status": "ORACLE_FAIL", "error": m.get("error")})
            all_ok = False
            continue
        sdir = OUTDIR / f"nacl_n{n}"
        nat = m["natoms"]
        e_oracle = m["energy_eV"]
        f_oracle = np.load(sdir / "oracle_forces.npy")

        cpp_log = sdir / "cpp.log"
        cpp_forces_bin = sdir / "cpp_forces.bin"
        e_cpp = parse_cpp_energy(cpp_log)
        if e_cpp is None or not cpp_forces_bin.is_file():
            rows.append({"n": n, "natoms": nat, "status": "CPP_MISSING"})
            all_ok = False
            continue
        f_cpp = np.fromfile(cpp_forces_bin, dtype=np.float64).reshape(-1, 3)
        if f_cpp.shape[0] != nat:
            rows.append({"n": n, "natoms": nat, "status": "CPP_SHAPE_MISMATCH",
                         "got": int(f_cpp.shape[0])})
            all_ok = False
            continue

        idx = sample_idx(nat, MIN_SAMPLE)
        dE = abs(e_cpp - e_oracle)
        dF = np.abs(f_cpp[idx] - f_oracle[idx])
        max_dF = float(dF.max())
        rms_dF = float(np.sqrt(np.mean(dF**2)))
        # cosine of full flattened force vectors (sampled)
        a = f_cpp[idx].ravel()
        b = f_oracle[idx].ravel()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
        ok = (dE <= E_TOL) and (max_dF <= F_TOL)
        all_ok = all_ok and ok
        rows.append({
            "n": n, "natoms": nat, "n_sampled": int(len(idx)),
            "E_oracle": e_oracle, "E_cpp": e_cpp, "dE_eV": dE,
            "dE_meV_per_atom": 1e3 * dE / nat,
            "max_dF": max_dF, "rms_dF": rms_dF, "cos": cos,
            "E_TOL": E_TOL, "F_TOL": F_TOL,
            "status": "PASS" if ok else "FAIL",
        })
        print(f"N={n} nat={nat} sampled={len(idx)} "
              f"dE={dE:.3e}eV ({1e3*dE/nat:.3e} meV/at) "
              f"max|dF|={max_dF:.3e} rms|dF|={rms_dF:.3e} cos={cos:.10f} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

    summary = {"E_TOL": E_TOL, "F_TOL": F_TOL, "min_sample": MIN_SAMPLE,
               "rows": rows, "all_pass": all_ok}
    (OUTDIR / "parity_compare.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTDIR}/parity_compare.json  all_pass={all_ok}", flush=True)
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
