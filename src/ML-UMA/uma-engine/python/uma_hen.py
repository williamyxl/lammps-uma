"""Resolve the `hen` shim/patches root without a hardcoded machine path (G15 / S7).

The export/worker layer imports monkeypatch shims from a sibling `hen` checkout.
Previously four files hardcoded `/lus/flare/.../xiaoliyan/workdir/hen` — another
user's absolute path — which the campaign's Tier-0 HARD 4 foreign-path guard would
have caught had its scope included `uma-engine/python/` (it now does, S7).

Resolution order (no machine-specific default):
  1. $UMA_HEN_ROOT
  2. a `hen/` sibling of the repo root (…/lammps-uma/../hen)
Fails loudly (FileNotFoundError) if neither exists, instead of silently importing
nothing and later failing deep inside a monkeypatch with an opaque error.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def hen_root() -> Path:
    env = os.environ.get("UMA_HEN_ROOT")
    if env:
        p = Path(env).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(
                f"UMA_HEN_ROOT={env} does not exist. Point it at the `hen` "
                f"shim/patches checkout.")
        return p.resolve()
    # repo root = .../lammps-uma/src/ML-UMA/uma-engine/python/uma_hen.py -> parents[4]
    repo = Path(__file__).resolve().parents[4]
    sib = repo.parent / "hen"
    if sib.is_dir():
        return sib.resolve()
    raise FileNotFoundError(
        "Cannot locate the `hen` shim/patches root. Set UMA_HEN_ROOT to the `hen` "
        f"checkout (looked for a sibling of the repo at {sib}).")


def add_hen_to_syspath() -> Path:
    """Prepend hen + hen/shim + hen/patches to sys.path; return the resolved root."""
    root = hen_root()
    for p in (root / "shim", root / "patches", root):
        if p.is_dir():
            sys.path.insert(0, str(p))
    return root
