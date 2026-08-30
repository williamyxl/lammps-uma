#!/usr/bin/env python3
"""N=3 NaCl FP64 single-point: ASE vs FairChem fix-external vs uma/kk.

Energy + forces only (no minim / MD for physics). Timing uses repeated
force evals (ASE/FC on fixed geometry; uma/kk via NVE Pair timer after run 0).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk
from ase.data import atomic_masses

def _ml_uma_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pair_uma.cpp").is_file() and (p / "uma-engine").is_dir():
            return p
    raise RuntimeError("cannot find ML-UMA (pair_uma.cpp + uma-engine/)")

_ML = _ml_uma_root()
ENGINE = _ML / "uma-engine"
ROOT = _ML.parent.parent.parent  # workdir parent of lammps tree (legacy layout)
sys.path.insert(0, str(ENGINE / "python"))
from common import inference_settings_with_dtype  # noqa: E402

CKPT = "/mnt/d/workdir/uma-cache/uma-s-1p2.pt"
ART_F64 = ENGINE / "artifacts" / "uma-s-1p2-omat-f64"
OUT_DIR = Path(__file__).resolve().parent


def make_nacl(n: int = 3, a: float = 5.64, perturb: float = 0.05, seed: int = 0):
    """Rock-salt supercell with a small displacement so forces are nonzero."""
    atoms = bulk("NaCl", "rocksalt", a=a, cubic=True) * (n, n, n)
    rng = np.random.default_rng(seed)
    atoms.positions = atoms.positions + rng.uniform(-perturb, perturb, size=atoms.positions.shape)
    return atoms



def write_data(atoms, path: Path) -> None:
    Z = atoms.get_atomic_numbers()
    types = np.where(Z == 11, 1, 2)
    cell = atoms.cell.array
    lx, ly, lz = float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])
    pos = atoms.get_positions()
    lines = [
        "NaCl N3 FP64 compare",
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


def setup_ld_path() -> dict:
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


def fp64_settings_for_fairchem(*, external_graph: bool):
    """FP64 InferenceSettings. ASE calculator builds the graph when external=True;
    FairChem fix-external passes empty edges and needs internal graph gen."""
    settings = inference_settings_with_dtype("float64")
    settings.external_graph_gen = external_graph
    return settings


def run_ase(atoms, n_timing: int) -> dict:
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    settings = fp64_settings_for_fairchem(external_graph=True)
    t0 = time.perf_counter()
    predictor = load_predict_unit(CKPT, device="cuda", inference_settings=settings)
    calc = FAIRChemCalculator(predictor, task_name="omat")
    a = atoms.copy()
    a.calc = calc
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(), dtype=np.float64)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    times = []
    for _ in range(n_timing):
        if hasattr(a.calc, "results"):
            a.calc.results.clear()
        a.positions = a.positions.copy()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        e = float(a.get_potential_energy())
        f = np.asarray(a.get_forces(), dtype=np.float64)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t1)

    return {
        "label": "ASE FAIRChemCalculator FP64 (CUDA)",
        "energy": e,
        "forces": f,
        "load_s": load_s,
        "eval_s_mean": float(np.mean(times)),
        "eval_s_std": float(np.std(times)),
        "eval_s_min": float(np.min(times)),
        "n_timing": n_timing,
        "times_s": times,
    }


def run_fairchem_lammps(atoms, n_timing: int, work: Path) -> dict:
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.lammps.lammps_fc import run_lammps_with_fairchem

    # fix external supplies empty edges → must use internal NL
    settings = fp64_settings_for_fairchem(external_graph=False)
    data = work / "data.nacl"
    write_data(atoms, data)
    inp = work / "in.fc"
    inp.write_text(
        f"""units metal
atom_style atomic
boundary p p p
read_data {data.name}
mass 1 22.98976928
mass 2 35.453
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo 1
thermo_style custom step pe
run 0
"""
    )

    t0 = time.perf_counter()
    predictor = load_predict_unit(CKPT, device="cuda", inference_settings=settings)
    load_s = time.perf_counter() - t0

    os.chdir(work)
    t1 = time.perf_counter()
    lmp = run_lammps_with_fairchem(predictor, str(inp), "omat")
    first_s = time.perf_counter() - t1

    e = float(lmp.get_thermo("pe"))
    nlocal = lmp.extract_global("nlocal")
    tags = np.array(lmp.numpy.extract_atom("id")[:nlocal]).copy()
    order = np.argsort(tags)
    f = np.array(lmp.numpy.extract_atom("f")[:nlocal], dtype=np.float64).copy()[order]

    times = [first_s]
    for _ in range(max(0, n_timing - 1)):
        t2 = time.perf_counter()
        lmp.command("run 0")
        times.append(time.perf_counter() - t2)
        e = float(lmp.get_thermo("pe"))
        f = np.array(lmp.numpy.extract_atom("f")[:nlocal], dtype=np.float64).copy()[order]

    steady = times[1:] if len(times) > 1 else times
    del lmp._predictor
    lmp.close()
    return {
        "label": "FairChem LAMMPS fix external (PyTorch CUDA, internal NL; bridge uses FP32 cell)",
        "energy": e,
        "forces": f,
        "load_s": load_s,
        "eval_s_mean": float(np.mean(times)),
        "eval_s_std": float(np.std(times)),
        "eval_s_min": float(np.min(times)),
        "eval_s_steady_mean": float(np.mean(steady)),
        "n_timing": n_timing,
        "times_s": times,
    }


def run_uma_kk(atoms, n_timing_steps: int, work: Path) -> dict:
    """Single-point forces via run 0 dump; timing via NVE Pair / steps."""
    data = work / "data.nacl"
    write_data(atoms, data)
    dump = work / "forces.dump"
    log_sp = work / "log.sp"
    log_nve = work / "log.nve"
    env = setup_ld_path()
    lmp = ROOT / "lammps" / "build-uma" / "lmp"

    # --- energy + forces at fixed geometry ---
    inp_sp = work / "in.sp"
    inp_sp.write_text(
        f"""units metal
atom_style atomic
boundary p p p
read_data {data.name}
pair_style uma/kk precision double
pair_coeff * * {ART_F64} Na Cl
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
    proc = subprocess.run(
        [str(lmp), "-k", "on", "g", "1", "-sf", "kk", "-in", inp_sp.name, "-log", log_sp.name],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "uma/kk SP failed:\n" + proc.stdout[-2000:] + "\n" + proc.stderr[-2000:]
        )

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
    rows = np.asarray(rows, dtype=np.float64)
    f = rows[np.argsort(rows[:, 0]), 1:4]

    # --- timing: run 0 warm + NVE steps; Pair / n_steps ---
    inp_nve = work / "in.nve"
    inp_nve.write_text(
        f"""units metal
atom_style atomic
boundary p p p
read_data {data.name}
pair_style uma/kk precision double
pair_coeff * * {ART_F64} Na Cl
newton off
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
velocity all create 0.0 1
timestep 0.001
fix 1 all nve
thermo 0
run 0
run {n_timing_steps}
"""
    )
    proc2 = subprocess.run(
        [str(lmp), "-k", "on", "g", "1", "-sf", "kk", "-in", inp_nve.name, "-log", log_nve.name],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc2.returncode != 0:
        raise RuntimeError(
            "uma/kk NVE failed:\n" + proc2.stdout[-2000:] + "\n" + proc2.stderr[-2000:]
        )

    pair_s = None
    for line in log_nve.read_text().splitlines():
        if line.strip().startswith("Pair") and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            try:
                v = float(parts[2])
                if v > 0:
                    pair_s = v
            except ValueError:
                pass
    eval_mean = (pair_s / n_timing_steps) if pair_s else None

    return {
        "label": "Native uma/kk precision double (Kokkos+LibTorch+vesin CUDA)",
        "energy": e,
        "forces": f,
        "load_s": None,
        "eval_s_mean": eval_mean,
        "eval_s_std": None,
        "eval_s_min": eval_mean,
        "n_timing": n_timing_steps,
        "pair_section_s": pair_s,
        "timing_method": f"LAMMPS Pair / {n_timing_steps} NVE steps after run 0",
    }


def main() -> int:
    n = 3
    n_timing = 5
    atoms = make_nacl(n)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"NaCl {n}x{n}x{n} FP64  natoms={len(atoms)}  "
        f"GPU={torch.cuda.get_device_name(0)}"
    )

    ase = run_ase(atoms, n_timing)
    print(
        f"ASE  E={ase['energy']:.10f}  "
        f"{ase['eval_s_mean']*1e3:.1f} ms/eval"
    )

    with tempfile.TemporaryDirectory(prefix="fc_", dir=OUT_DIR) as td:
        fc = run_fairchem_lammps(atoms, n_timing, Path(td))
    print(
        f"FC   E={fc['energy']:.10f}  "
        f"steady={fc['eval_s_steady_mean']*1e3:.1f} ms/eval"
    )

    with tempfile.TemporaryDirectory(prefix="uma_", dir=OUT_DIR) as td:
        uma = run_uma_kk(atoms, n_timing, Path(td))
    print(
        f"uma  E={uma['energy']:.10f}  "
        f"{(uma['eval_s_mean'] or 0)*1e3:.1f} ms/eval"
    )

    report = {
        "system": f"NaCl {n}x{n}x{n} rocksalt a=5.64",
        "natoms": len(atoms),
        "precision": "float64 / double",
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": CKPT,
        "artifact_uma": str(ART_F64),
        "note": (
            "Single-point energy+forces only. "
            "FairChem LAMMPS = fix external + Predictor (not Kokkos). "
            "uma/kk = pair_style uma/kk precision double."
        ),
        "ase": {k: v for k, v in ase.items() if k not in ("forces", "times_s")},
        "fairchem_lammps": {
            k: v for k, v in fc.items() if k not in ("forces", "times_s")
        },
        "uma_kk_double": {
            k: v for k, v in uma.items() if k not in ("forces",)
        },
        "vs_ase": {
            "fairchem_lammps": {
                "abs_energy_error": abs(fc["energy"] - ase["energy"]),
                **force_stats(ase["forces"], fc["forces"]),
            },
            "uma_kk_double": {
                "abs_energy_error": abs(uma["energy"] - ase["energy"]),
                **force_stats(ase["forces"], uma["forces"]),
            },
        },
        "vs_fairchem_lammps": {
            "uma_kk_double": {
                "abs_energy_error": abs(uma["energy"] - fc["energy"]),
                **force_stats(fc["forces"], uma["forces"]),
            },
        },
        "timing_ms_per_eval": {
            "ase": ase["eval_s_mean"] * 1e3,
            "fairchem_lammps_mean_incl_first": fc["eval_s_mean"] * 1e3,
            "fairchem_lammps_steady": fc["eval_s_steady_mean"] * 1e3,
            "uma_kk_double": None
            if uma["eval_s_mean"] is None
            else uma["eval_s_mean"] * 1e3,
        },
    }

    np.savez(
        OUT_DIR / "forces_n3_f64.npz",
        ase=ase["forces"],
        fairchem_lammps=fc["forces"],
        uma_kk=uma["forces"],
    )
    out_json = OUT_DIR / "compare_report.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    print("\n==== SUMMARY ====")
    print(json.dumps({
        "energies_eV": {
            "ase": ase["energy"],
            "fairchem_lammps": fc["energy"],
            "uma_kk": uma["energy"],
        },
        "timing_ms": report["timing_ms_per_eval"],
        "vs_ase_energy": {
            "fc": report["vs_ase"]["fairchem_lammps"]["abs_energy_error"],
            "uma": report["vs_ase"]["uma_kk_double"]["abs_energy_error"],
        },
        "vs_ase_force_max_abs": {
            "fc": report["vs_ase"]["fairchem_lammps"]["force_max_abs"],
            "uma": report["vs_ase"]["uma_kk_double"]["force_max_abs"],
        },
        "vs_ase_force_mae": {
            "fc": report["vs_ase"]["fairchem_lammps"]["force_mae"],
            "uma": report["vs_ase"]["uma_kk_double"]["force_mae"],
        },
    }, indent=2))
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
