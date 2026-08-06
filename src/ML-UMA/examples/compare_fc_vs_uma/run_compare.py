#!/usr/bin/env python3
"""Compare FairChem LAMMPS (Python fix external) vs native pair_style uma/kk.

FairChem's shipped LAMMPS path is *not* a Kokkos pair style — it is
fix external + FAIRChem Predictor on CUDA. Our path is pair_style uma/kk
(LibTorch, Kokkos-enabled binary).

Compares energy, per-atom forces, and wall timing on the same NaCl structure.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXAMPLES))
from _repo import find_uma_engine_root, find_uma_lmp_root  # noqa: E402

ROOT = find_uma_lmp_root()
ENGINE = find_uma_engine_root()


def load_nacl() -> Atoms:
    d = np.load(ROOT / "lammps" / "src" / "ML-UMA" / "examples" / "nacl_minim" / "nacl_init.npz")
    return Atoms(
        numbers=d["numbers"],
        positions=d["positions"],
        cell=d["cell"],
        pbc=True,
    )


def write_data(atoms: Atoms, path: Path) -> None:
    """LAMMPS data file; type1=Na, type2=Cl."""
    from ase.data import atomic_masses

    Z = atoms.get_atomic_numbers()
    types = np.where(Z == 11, 1, 2)
    cell = atoms.cell.array
    lx, ly, lz = np.diag(cell)
    pos = atoms.get_positions()
    lines = [
        "NaCl compare",
        "",
        f"{len(atoms)} atoms",
        "2 atom types",
        "",
        f"0.0 {lx:.16f} xlo xhi",
        f"0.0 {ly:.16f} ylo yhi",
        f"0.0 {lz:.16f} zlo zhi",
        "",
        "Masses",
        "",
        f"1 {atomic_masses[11]:.8f}",
        f"2 {atomic_masses[17]:.8f}",
        "",
        "Atoms # atomic",
        "",
    ]
    for i, (t, p) in enumerate(zip(types, pos), 1):
        lines.append(f"{i} {t} {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}")
    path.write_text("\n".join(lines) + "\n")


def run_ase(atoms: Atoms, ckpt: str, device: str, n_timing: int) -> dict:
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    t0 = time.perf_counter()
    predictor = load_predict_unit(ckpt, device=device)
    calc = FAIRChemCalculator(predictor, task_name="omat")
    a = atoms.copy()
    a.calc = calc
    # warmup
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(), dtype=np.float64)
    load_s = time.perf_counter() - t0

    times = []
    for _ in range(n_timing):
        a.calc.results.clear() if hasattr(a.calc, "results") else None
        # force recompute
        a.positions = a.positions.copy()
        t1 = time.perf_counter()
        e = float(a.get_potential_energy())
        f = np.asarray(a.get_forces(), dtype=np.float64)
        times.append(time.perf_counter() - t1)
    return {
        "label": "ASE FAIRChemCalculator (CUDA)",
        "energy": e,
        "forces": f,
        "load_s": load_s,
        "eval_s_mean": float(np.mean(times)),
        "eval_s_std": float(np.std(times)),
        "eval_s_min": float(np.min(times)),
        "n_timing": n_timing,
    }


def run_fairchem_lammps(atoms: Atoms, ckpt: str, device: str, n_timing: int, work: Path) -> dict:
    """FairChem Python fix-external LAMMPS bridge (not Kokkos pair)."""
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.lammps.lammps_fc import run_lammps_with_fairchem

    data = work / "data.nacl"
    write_data(atoms, data)

    # Minimal script: setup only; run_lammps_with_fairchem injects fix external then run cmds
    inp = work / "in.fc"
    inp.write_text(
        f"""units metal
atom_style atomic
boundary p p p
read_data {data.name}
mass 1 22.989769
mass 2 35.453
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo 1
thermo_style custom step pe
run 0
"""
    )

    t0 = time.perf_counter()
    predictor = load_predict_unit(ckpt, device=device)
    load_s = time.perf_counter() - t0

    # First call = setup + one force eval
    os.chdir(work)
    t1 = time.perf_counter()
    lmp = run_lammps_with_fairchem(predictor, str(inp), "omat")
    first_s = time.perf_counter() - t1

    e = float(lmp.get_thermo("pe"))
    nlocal = lmp.extract_global("nlocal")
    f = np.array(lmp.numpy.extract_atom("f")[:nlocal], dtype=np.float64).copy()
    # Sort by id for stable compare
    tags = np.array(lmp.numpy.extract_atom("id")[:nlocal]).copy()
    order = np.argsort(tags)
    f = f[order]

    times = [first_s]
    # Additional force evals via run 0
    for _ in range(n_timing - 1):
        t2 = time.perf_counter()
        lmp.command("run 0")
        times.append(time.perf_counter() - t2)
        e = float(lmp.get_thermo("pe"))
        f = np.array(lmp.numpy.extract_atom("f")[:nlocal], dtype=np.float64).copy()[order]

    del lmp._predictor
    lmp.close()
    return {
        "label": "FairChem LAMMPS fix external (PyTorch CUDA; not Kokkos pair)",
        "energy": e,
        "forces": f,
        "load_s": load_s,
        "eval_s_mean": float(np.mean(times)),
        "eval_s_std": float(np.std(times)),
        "eval_s_min": float(np.min(times)),
        "n_timing": n_timing,
        "note": "conda LAMMPS has no PKG_KOKKOS; inference is FairChem Predictor via fix external",
    }


def setup_ld_path() -> dict:
    import torch

    env = os.environ.copy()
    vesin = ENGINE / "third_party" / "vesin" / "lib"
    torch_lib = Path(torch.__path__[0]) / "lib"
    parts = [
        "/usr/lib/wsl/lib",
        "/usr/local/cuda/lib64",
        str(vesin),
        str(torch_lib),
    ]
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def run_uma_kk(atoms: Atoms, artifact: Path, precision: str, n_timing: int, work: Path) -> dict:
    """Native pair_style uma/kk. Forces via run 0; timing via NVE Pair / steps."""
    data = work / "data.nacl"
    write_data(atoms, data)
    dump = work / "forces.dump"
    log_sp = work / "log.sp"
    log_nve = work / "log.nve"
    env = setup_ld_path()
    lmp = ROOT / "lammps" / "build-uma" / "lmp"

    inp_sp = work / "in.sp"
    inp_sp.write_text(
        f"""units metal
atom_style atomic
boundary p p p
read_data {data.name}
pair_style uma/kk precision {precision}
pair_coeff * * {artifact} Na Cl
newton off
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo 1
thermo_style custom step pe fmax fnorm
dump 1 all custom 1 {dump.name} id type x y z fx fy fz
dump_modify 1 sort id
run 0
print "Final PE = $(pe)"
"""
    )
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(lmp), "-k", "on", "g", "1", "-sf", "kk", "-in", inp_sp.name, "-log", log_sp.name],
        cwd=str(work),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    total_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:] + "\n" + proc.stdout[-2000:])

    e = None
    for line in log_sp.read_text().splitlines():
        if line.startswith("Final PE"):
            e = float(line.split("=")[1].strip())
    text = dump.read_text().split("ITEM: ATOMS")[-1].strip().splitlines()
    rows = []
    for line in text[1:]:
        parts = line.split()
        if len(parts) >= 8:
            rows.append([float(parts[0]), float(parts[5]), float(parts[6]), float(parts[7])])
    rows = np.array(rows)
    order = np.argsort(rows[:, 0])
    f = rows[order, 1:4]

    # run 0 does not accumulate Pair time; use NVE after warmup
    n_steps = max(n_timing, 5)
    inp_nve = work / "in.nve"
    inp_nve.write_text(
        f"""units metal
atom_style atomic
boundary p p p
read_data {data.name}
pair_style uma/kk precision {precision}
pair_coeff * * {artifact} Na Cl
newton off
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
velocity all create 0.0 1
timestep 0.001
fix 1 all nve
thermo 0
run 0
run {n_steps}
"""
    )
    proc2 = subprocess.run(
        [str(lmp), "-k", "on", "g", "1", "-sf", "kk", "-in", inp_nve.name, "-log", log_nve.name],
        cwd=str(work),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc2.returncode != 0:
        raise RuntimeError(proc2.stderr[-2000:] + "\n" + proc2.stdout[-2000:])

    pair_time = None
    for line in log_nve.read_text().splitlines():
        if line.strip().startswith("Pair") and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            try:
                v = float(parts[2])
                if v > 0:
                    pair_time = v
            except ValueError:
                pass
    eval_mean = (pair_time / n_steps) if pair_time is not None else None

    return {
        "label": f"Native uma/kk precision {precision} (Kokkos+LibTorch+vesin CUDA NL)",
        "energy": e,
        "forces": f,
        "load_s": None,
        "eval_s_mean": float(eval_mean) if eval_mean is not None else None,
        "eval_s_std": None,
        "eval_s_min": float(eval_mean) if eval_mean is not None else None,
        "n_timing": n_steps,
        "pair_section_s": pair_time,
        "wall_subprocess_s": total_s,
        "timing_method": f"LAMMPS Pair / {n_steps} NVE steps after run 0",
        "neighbor_list": "vesin-torch 0.5.8 CUDA",
    }


def force_stats(f_ref: np.ndarray, f: np.ndarray) -> dict:
    df = f - f_ref
    return {
        "force_mae": float(np.mean(np.abs(df))),
        "force_rmse": float(np.sqrt(np.mean(df**2))),
        "force_max_abs": float(np.max(np.abs(df))),
        "force_max_norm_per_atom": float(np.max(np.linalg.norm(df, axis=1))),
        "cosine": float(
            np.dot(f_ref.ravel(), f.ravel())
            / (np.linalg.norm(f_ref) * np.linalg.norm(f) + 1e-30)
        ),
    }


def main() -> int:
    import torch

    ckpt = "/mnt/d/workdir/uma-cache/uma-s-1p2.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_timing = 5
    atoms = load_nacl()
    out_dir = ROOT / "lammps" / "src" / "ML-UMA" / "examples" / "compare_fc_vs_uma"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device} natoms={len(atoms)} n_timing={n_timing}")

    ase = run_ase(atoms, ckpt, device, n_timing)
    print(f"ASE E={ase['energy']:.10f}  eval={ase['eval_s_mean']*1e3:.2f} ms")

    with tempfile.TemporaryDirectory(prefix="fc_lmp_", dir=out_dir) as td:
        fc = run_fairchem_lammps(atoms, ckpt, device, n_timing, Path(td))
    print(f"FairChem-LMP E={fc['energy']:.10f}  eval={fc['eval_s_mean']*1e3:.2f} ms")

    art_mixed = ENGINE / "artifacts" / "uma-s-1p2-omat"
    with tempfile.TemporaryDirectory(prefix="uma_kk_", dir=out_dir) as td:
        uma = run_uma_kk(atoms, art_mixed, "mixed", n_timing, Path(td))
    print(f"uma/kk E={uma['energy']:.10f}  eval~={uma['eval_s_mean']*1e3:.2f} ms")

    report = {
        "system": "NaCl 2x2x2 perturbed seed0",
        "natoms": len(atoms),
        "device": device,
        "checkpoint": ckpt,
        "note": (
            "FairChem LAMMPS path uses fix external + Python Predictor; "
            "conda LAMMPS build has no KOKKOS package. "
            "Native path is pair_style uma/kk (Kokkos+LibTorch+vesin). "
            "Post denorm_energy dtype fix (preserves FP32/FP64)."
        ),
        "ase": {k: v for k, v in ase.items() if k != "forces"},
        "fairchem_lammps": {k: v for k, v in fc.items() if k != "forces"},
        "uma_kk_mixed": {k: v for k, v in uma.items() if k not in ("forces", "stdout_tail")},
        "vs_ase": {
            "fairchem_lammps": {
                "abs_energy_error": abs(fc["energy"] - ase["energy"]),
                **force_stats(ase["forces"], fc["forces"]),
            },
            "uma_kk_mixed": {
                "abs_energy_error": abs(uma["energy"] - ase["energy"]),
                **force_stats(ase["forces"], uma["forces"]),
            },
        },
        "vs_fairchem_lammps": {
            "uma_kk_mixed": {
                "abs_energy_error": abs(uma["energy"] - fc["energy"]),
                **force_stats(fc["forces"], uma["forces"]),
            },
        },
        "timing_ms_per_eval": {
            "ase": ase["eval_s_mean"] * 1e3,
            "fairchem_lammps": fc["eval_s_mean"] * 1e3,
            "uma_kk_mixed": (uma["eval_s_mean"] * 1e3) if uma["eval_s_mean"] else None,
        },
    }

    # Save forces for inspection
    np.savez(
        out_dir / "forces_compare.npz",
        ase=ase["forces"],
        fairchem_lammps=fc["forces"],
        uma_kk=uma["forces"],
    )
    out_json = out_dir / "compare_report.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
