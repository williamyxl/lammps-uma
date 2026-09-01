#!/usr/bin/env python3
"""Tier 1 (f): gate arithmetic — the fail-closed logic must actually fail closed.

Pure Python + numpy (no torch/fairchem). Runs on a login node in < 1 s. Covers the
Sprint-3 P1.1/P1.2/P1.5 fixes and the parity comparator's decision logic:
  - uma_gates single source of truth + env overrides
  - zero samples -> FAIL, atom-count mismatch -> FAIL, perturbed -> FAIL,
    negated -> FAIL, identical -> PASS

Invoke: python3 ci/tests/test_gate_arithmetic.py   (or via pytest under fxpu)
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import uma_gates as g  # noqa: E402


def test_uma_gates_defaults():
    assert g.e_tol_per_atom_mev() == 1e-3
    assert g.f_tol() == 1e-5
    assert g.agfd_tol() == 1e-5
    assert g.min_sample() == 100
    assert g.fd_eps() == 1e-4


def test_uma_gates_env_override(monkeypatch=None):
    old = dict(os.environ)
    try:
        os.environ["F_TOL"] = "2e-5"
        os.environ["AG_FD_TOL"] = "3e-5"   # legacy alias
        os.environ["MIN_SAMPLE"] = "42"
        assert g.f_tol() == 2e-5
        assert g.agfd_tol() == 3e-5
        assert g.min_sample() == 42
    finally:
        os.environ.clear()
        os.environ.update(old)


# --- replicate the parity comparator's decision so we can unit-test it ---------
def parity_decision(f_lmp, f_ase, e_lmp, e_ase, *, f_tol, e_tol_mev, min_sample):
    """Mirror of parity_vs_asegp.main()'s gate logic (P1.5 hardened)."""
    if len(f_lmp) != len(f_ase):
        return "FAIL_NATOMS"
    n = len(f_lmp)
    if n < min_sample:
        return "FAIL_SAMPLE"
    dE_per_atom_meV = abs(e_lmp - e_ase) / n * 1e3
    max_dF = float(np.abs(f_lmp - f_ase).max())
    e_ok = dE_per_atom_meV <= e_tol_mev
    f_ok = max_dF <= f_tol
    return "PASS" if (e_ok and f_ok) else "FAIL"


def _rng(n, seed=0):
    r = np.random.default_rng(seed)
    return r.standard_normal((n, 3))


def test_identical_passes():
    f = _rng(200)
    assert parity_decision(f, f.copy(), -100.0, -100.0,
                           f_tol=1e-5, e_tol_mev=1e-3, min_sample=100) == "PASS"


def test_atom_count_mismatch_fails():
    a, b = _rng(200), _rng(199, seed=1)
    assert parity_decision(a, b, -100.0, -100.0,
                           f_tol=1e-5, e_tol_mev=1e-3, min_sample=100) == "FAIL_NATOMS"


def test_zero_samples_fails():
    z = np.zeros((0, 3))
    assert parity_decision(z, z, -100.0, -100.0,
                           f_tol=1e-5, e_tol_mev=1e-3, min_sample=100) == "FAIL_SAMPLE"


def test_too_few_samples_fails():
    f = _rng(10)
    assert parity_decision(f, f.copy(), -100.0, -100.0,
                           f_tol=1e-5, e_tol_mev=1e-3, min_sample=100) == "FAIL_SAMPLE"


def test_perturbed_force_fails():
    f = _rng(200)
    f2 = f.copy(); f2[3, 1] += 1e-3   # 100x the gate
    assert parity_decision(f, f2, -100.0, -100.0,
                           f_tol=1e-5, e_tol_mev=1e-3, min_sample=100) == "FAIL"


def test_negated_force_fails():
    f = _rng(200)
    assert parity_decision(f, -f, -100.0, -100.0,
                           f_tol=1e-5, e_tol_mev=1e-3, min_sample=100) == "FAIL"


def test_energy_drift_fails():
    f = _rng(200)
    # 1 meV/atom drift over 200 atoms >> 1e-3 meV/atom gate
    assert parity_decision(f, f.copy(), -100.0, -100.0 - 0.2,
                           f_tol=1e-5, e_tol_mev=1e-3, min_sample=100) == "FAIL"


def test_floor_noise_passes():
    f = _rng(200)
    f2 = f + 1e-14 * _rng(200, seed=7)   # FP64 floor
    assert parity_decision(f, f2, -100.0, -100.0 - 1e-9,
                           f_tol=1e-5, e_tol_mev=1e-3, min_sample=100) == "PASS"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    n_fail = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            n_fail += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests)-n_fail}/{len(tests)} passed")
    sys.exit(1 if n_fail else 0)
