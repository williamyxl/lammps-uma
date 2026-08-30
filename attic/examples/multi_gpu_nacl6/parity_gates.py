#!/usr/bin/env python3
"""Parity thresholds and gate checks for uma/kk graph-parallel.

Oracles (same precision family):
  - double: uma traced devices=1
  - mixed:  ASE FairChem float32 workers=1 (eager; traced mixed disagrees ~0.058 eV)
"""

from __future__ import annotations

from typing import Any

PARITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "double": {
        "abs_dE_max": 1e-8,
        "force_max_abs_max": 1e-6,
        "cosine_min": 1.0 - 1e-12,
    },
    "mixed": {
        # FairChem float32 w1↔w2 on NaCl6 is ~1.4e-4; uma GP vs ASE f32@1 ~2e-4.
        "abs_dE_max": 5e-4,
        "force_max_abs_max": 1e-5,
        "cosine_min": 1.0 - 1e-10,
    },
}

UMA_PATH_KEYS = ("uma_double", "uma_mixed")
PRECISION_BY_KEY = {"uma_double": "double", "uma_mixed": "mixed"}


def check_gate(
    key: str,
    *,
    abs_dE: float | None,
    force_max_abs: float | None,
    cosine: float | None,
) -> dict[str, Any]:
    """Return gate dict with per-metric pass/fail vs devices=1 thresholds."""
    prec = PRECISION_BY_KEY.get(key)
    if prec is None:
        return {"applicable": False}
    th = PARITY_THRESHOLDS[prec]
    checks: dict[str, Any] = {
        "applicable": True,
        "precision": prec,
        "thresholds": th,
        "metrics": {
            "abs_dE_vs_uma_d1": abs_dE,
            "force_max_abs_vs_uma_d1": force_max_abs,
            "cosine_vs_uma_d1": cosine,
        },
    }
    passed = True
    detail: dict[str, bool | None] = {}
    if abs_dE is None:
        detail["abs_dE"] = None
        passed = False
    else:
        detail["abs_dE"] = abs_dE <= th["abs_dE_max"]
        passed = passed and detail["abs_dE"]
    if force_max_abs is None:
        detail["force_max_abs"] = None
        passed = False
    else:
        detail["force_max_abs"] = force_max_abs <= th["force_max_abs_max"]
        passed = passed and detail["force_max_abs"]
    if cosine is None:
        detail["cosine"] = None
        passed = False
    else:
        detail["cosine"] = cosine >= th["cosine_min"]
        passed = passed and detail["cosine"]
    checks["detail"] = detail
    checks["passed"] = passed
    return checks


def summarize_gates(gates: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [g for g in gates if g.get("applicable")]
    if not applicable:
        return {"all_passed": None, "n_checked": 0, "n_passed": 0}
    n_passed = sum(1 for g in applicable if g.get("passed"))
    return {
        "all_passed": n_passed == len(applicable),
        "n_checked": len(applicable),
        "n_passed": n_passed,
        "gates": gates,
    }
