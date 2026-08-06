"""Locate the uma-lmp workspace root from any file under ML-UMA/examples/."""

from __future__ import annotations

from pathlib import Path


def find_uma_lmp_root(start: Path | None = None) -> Path:
    """Walk parents until both ``uma-engine/`` and ``lammps/`` exist."""
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / "uma-engine").is_dir() and (p / "lammps").is_dir():
            return p
    raise RuntimeError(
        "cannot find uma-lmp root (expected sibling dirs uma-engine/ and lammps/)"
    )


def example_dir() -> Path:
    """Directory containing this case's inputs (parent of the calling script... use Path(__file__).parent)."""
    return Path(__file__).resolve().parent
