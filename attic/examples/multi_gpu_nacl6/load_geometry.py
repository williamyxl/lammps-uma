#!/usr/bin/env python3
"""Load the frozen NaCl 6x6x6 rattled geometry (never re-perturb)."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read

EXPECTED_NATOMS = 1728
FIXED_EXTXYZ = (
    Path(__file__).resolve().parent
    / "structures"
    / "nacl6_rattle_fixed.extxyz"
)
FIXED_MANIFEST = FIXED_EXTXYZ.with_suffix("").with_name(
    "nacl6_rattle_fixed.manifest.json"
)
FIXED_NPZ = (
    Path(__file__).resolve().parents[1]
    / "delta_parity"
    / "structures"
    / "structure_nacl6_rattle.npz"
)


def load_nacl6_fixed(path: Path | None = None) -> Atoms:
    """Load fixed NaCl 6x6x6 rocksalt rattle; assert 1728 atoms.

    Prefer the frozen ``.extxyz`` (12 significant digits). Equivalent ``.npz``
    is accepted only as a fallback — never regenerate / re-rattle.
    Env ``FIXED_STRUCTURE`` overrides the default extxyz path when ``path`` is None.
    """
    import os

    if path is not None:
        src = Path(path)
    elif os.environ.get("FIXED_STRUCTURE"):
        src = Path(os.environ["FIXED_STRUCTURE"]).expanduser().resolve()
    else:
        src = FIXED_EXTXYZ
    if not src.is_file():
        if FIXED_NPZ.is_file() and path is None:
            import numpy as np

            d = np.load(FIXED_NPZ, allow_pickle=False)
            atoms = Atoms(
                numbers=d["numbers"],
                positions=d["positions"],
                cell=d["cell"],
                pbc=True,
            )
            atoms.info["source"] = str(FIXED_NPZ)
        else:
            raise FileNotFoundError(
                f"frozen geometry missing: {src} (also checked {FIXED_NPZ})"
            )
    elif src.suffix.lower() == ".npz":
        import numpy as np

        d = np.load(src, allow_pickle=False)
        atoms = Atoms(
            numbers=d["numbers"],
            positions=d["positions"],
            cell=d["cell"],
            pbc=True,
        )
        atoms.info["source"] = str(src.resolve())
    else:
        atoms = read(str(src))
        if not isinstance(atoms, Atoms):
            raise TypeError(f"expected a single Atoms frame from {src}")
        atoms.info["source"] = str(src.resolve())

    expected = int(os.environ.get("EXPECTED_NATOMS", str(EXPECTED_NATOMS)))
    if len(atoms) != expected:
        raise AssertionError(
            f"geometry contract violated: natoms={len(atoms)} "
            f"(expected {expected}) from {atoms.info.get('source')}"
        )
    atoms.set_pbc(True)
    return atoms


def geometry_meta(atoms: Atoms) -> dict:
    """Small metadata block for parity.json."""
    cell = atoms.cell.array
    return {
        "source": atoms.info.get("source", str(FIXED_EXTXYZ)),
        "manifest": str(FIXED_MANIFEST) if FIXED_MANIFEST.is_file() else None,
        "natoms": len(atoms),
        "cell": cell.tolist(),
        "perturbation": {
            "mode": "uniform_box",
            "delta_A": 0.1,
            "seed": 0,
            "note": "Frozen; do not regenerate. Coordinates .12g in extxyz.",
        },
    }


if __name__ == "__main__":
    a = load_nacl6_fixed()
    print(f"loaded {len(a)} atoms from {a.info.get('source')}")
    print(f"cell diagonal = {a.cell.lengths()}")
