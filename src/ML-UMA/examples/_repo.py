"""Locate ML-UMA / uma-engine / LAMMPS roots from files under examples/."""

from __future__ import annotations

from pathlib import Path


def find_ml_uma_root(start: Path | None = None) -> Path:
    """Walk parents until ``pair_uma.cpp`` and ``uma-engine/`` are present."""
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / "pair_uma.cpp").is_file() and (p / "uma-engine").is_dir():
            return p
    raise RuntimeError(
        "cannot find ML-UMA package root (expected pair_uma.cpp + uma-engine/)"
    )


def find_uma_engine_root(start: Path | None = None) -> Path:
    """Path to vendored ``uma-engine`` under ML-UMA."""
    return find_ml_uma_root(start) / "uma-engine"


def find_lammps_root(start: Path | None = None) -> Path:
    """LAMMPS tree root (``.../lammps``)."""
    return find_ml_uma_root(start).parent.parent


def find_uma_lmp_root(start: Path | None = None) -> Path:
    """Workspace root containing ``lammps/`` (parent of the LAMMPS tree)."""
    return find_lammps_root(start).parent


def example_dir() -> Path:
    """Directory containing this helper (``ML-UMA/examples``)."""
    return Path(__file__).resolve().parent
