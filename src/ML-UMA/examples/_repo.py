"""Locate ML-UMA / uma-engine / LAMMPS roots from files under examples/."""

from __future__ import annotations

import os
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
    """LAMMPS tree root (directory containing ``src/`` and ``cmake/``)."""
    return find_ml_uma_root(start).parent.parent


def find_uma_lmp_root(start: Path | None = None) -> Path:
    """Workspace parent of the LAMMPS tree (e.g. ``.../workdir``).

    Prefer :func:`find_lammps_root` / :func:`find_uma_lmp_binary` for Delta
    layouts where the clone *is* the LAMMPS tree (``lammps-uma/``).
    """
    return find_lammps_root(start).parent


def example_dir() -> Path:
    """Directory containing this helper (``ML-UMA/examples``)."""
    return Path(__file__).resolve().parent


def find_checkpoint() -> Path:
    """UMA checkpoint path (env ``UMA_CHECKPOINT`` or sibling ``uma-cache``)."""
    env = os.environ.get("UMA_CHECKPOINT")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        find_uma_lmp_root() / "uma-cache" / "uma-s-1p2.pt",
        Path("/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"),
        Path("/mnt/d/workdir/uma-cache/uma-s-1p2.pt"),
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "uma-s-1p2.pt not found; set UMA_CHECKPOINT=/path/to/uma-s-1p2.pt"
    )


def find_uma_lmp_binary() -> Path:
    """Local Kokkos+ML-UMA LAMMPS binary (env ``LMP_UMA`` or ``build-uma/lmp``)."""
    env = os.environ.get("LMP_UMA")
    if env:
        return Path(env).expanduser().resolve()
    root = find_lammps_root()
    candidates = [
        root / "build-uma" / "lmp",
        root / "build-uma" / "lmp_kokkos_cuda",
        # Legacy uma-lmp layout: <workspace>/lammps/build-uma/lmp
        find_uma_lmp_root() / "lammps" / "build-uma" / "lmp",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "local uma/kk LAMMPS binary not found; build build-uma/lmp or set LMP_UMA="
    )


def find_fairchem_lmp_binary() -> Path:
    """Conda FairChem LAMMPS binary (env ``LMP_FC`` or ``$CONDA_PREFIX/bin/lmp``)."""
    env = os.environ.get("LMP_FC")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        Path("/u/xyan11/miniforge3-x86_64/envs/uma312/bin/lmp"),
        Path(os.environ["CONDA_PREFIX"]) / "bin" / "lmp"
        if os.environ.get("CONDA_PREFIX")
        else None,
    ]
    for c in candidates:
        if c is not None and c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "FairChem LAMMPS binary not found; set LMP_FC=/path/to/conda/bin/lmp"
    )
