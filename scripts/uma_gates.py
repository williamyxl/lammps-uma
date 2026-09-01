#!/usr/bin/env python3
"""Single source of truth for UMA parity/AG=FD gate tolerances (P1.5).

Every comparator (parity_vs_asegp.py, phase6_agfd.py, phase6_gate1_compare.py) and
every PBS gate script must take its thresholds from HERE instead of hard-coding its
own copy (there were ~25 divergent copies; some 100x tighter/looser than others).

Env overrides are still honored so a specific experiment can tighten/loosen a gate
explicitly, but the DEFAULT is one shared table. Import and call the helpers; do not
redefine tolerances locally.

Tolerances (FP64 production contract, validated in REPORT_2path_nvt_comparison.md):
  - energy:   <= 1e-3 meV/atom  (per-atom; a fixed TOTAL tol is unfair at 1e5-1e6
              atoms where ~1e-8 meV/atom FP64 accumulation sums to ~1e-6 eV total)
  - force:    per-atom max|dF| <= 1e-5 eV/A over ALL compared atoms
  - AG=FD:    max|F_autograd - F_finite_difference| <= 1e-5 eV/A
  - cosine:   ~ 1.0 (report; not gated numerically because it is 1.0000000000)
  - min_sample: >= 100 atoms/samples before a PASS is trustworthy
"""
from __future__ import annotations
import os

# --- canonical FP64 gate table ------------------------------------------------
E_TOL_PER_ATOM_MEV = 1e-3   # meV/atom
F_TOL = 1e-5                # eV/A, per-atom max|dF|
AGFD_TOL = 1e-5            # eV/A, max|AG - FD|
MIN_SAMPLE = 100           # atoms / samples
FD_EPS = 1e-4              # A, central-difference step for AG=FD


def _envf(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else float(default)


def _envi(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else int(default)


def e_tol_per_atom_mev() -> float:
    """Per-atom energy tolerance (meV/atom). Env: E_TOL_PER_ATOM_MEV."""
    return _envf("E_TOL_PER_ATOM_MEV", E_TOL_PER_ATOM_MEV)


def f_tol() -> float:
    """Per-atom force tolerance (eV/A). Env: F_TOL."""
    return _envf("F_TOL", F_TOL)


def agfd_tol() -> float:
    """AG=FD force tolerance (eV/A). Env: AG_FD_TOL (legacy) or AGFD_TOL."""
    if os.environ.get("AG_FD_TOL"):
        return float(os.environ["AG_FD_TOL"])
    return _envf("AGFD_TOL", AGFD_TOL)


def min_sample() -> int:
    """Minimum sample count before a PASS is trustworthy. Env: MIN_SAMPLE."""
    return _envi("MIN_SAMPLE", MIN_SAMPLE)


def fd_eps() -> float:
    """Central-difference step (A). Env: FD_EPS."""
    return _envf("FD_EPS", FD_EPS)
