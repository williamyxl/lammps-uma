#!/usr/bin/env python3
"""End-to-end FP64 parity: ASE (float64) vs exported module vs C++/LAMMPS."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXAMPLES))
from _repo import find_uma_lmp_root  # noqa: E402

ROOT = find_uma_lmp_root()
sys.path.insert(0, str(ROOT / "uma-engine" / "python"))

from common import inference_settings_with_dtype, resolve_device  # noqa: E402


def main() -> int:
    device = resolve_device("cuda")
    artifact = ROOT / "uma-engine" / "artifacts" / "uma-s-1p2-omat-f64"
    npz = ROOT / "lammps" / "src" / "ML-UMA" / "examples" / "nacl_minim" / "nacl_init.npz"
    ckpt = "/mnt/d/workdir/uma-cache/uma-s-1p2.pt"
    settings = inference_settings_with_dtype("float64")

    data = np.load(npz)
    atoms = Atoms(
        numbers=data["numbers"],
        positions=data["positions"],
        cell=data["cell"],
        pbc=True,
    )

    # --- ASE FP64 ---
    predictor = load_predict_unit(ckpt, device=device, inference_settings=settings)
    atoms.calc = FAIRChemCalculator(predictor, task_name="omat")
    e_ase = float(atoms.get_potential_energy())
    f_ase = np.asarray(atoms.get_forces(), dtype=np.float64)

    # --- Python export path (parity_nacl) ---
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'uma-engine' / 'python'}:{env.get('PYTHONPATH', '')}"
    py = subprocess.run(
        [
            sys.executable,
            str(ROOT / "uma-engine" / "python" / "parity_nacl.py"),
            "--dtype",
            "float64",
            "--artifact",
            str(artifact),
            "--npz",
            str(npz),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    # last JSON object in stdout
    py_out = json.loads(py.stdout[py.stdout.find("{") :])
    assert py.returncode == 0, py.stderr

    # --- C++ CLI ---
    torch_lib = __import__("torch").__path__[0] + "/lib"
    env["LD_LIBRARY_PATH"] = torch_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    struct = ROOT / "lammps" / "src" / "ML-UMA" / "examples" / "nacl_minim" / "structure_f64.txt"
    cli = ROOT / "uma-engine" / "build" / "uma_parity_cli"
    cpp = subprocess.run(
        [str(cli), str(artifact), str(struct)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    # parse "energy=... eV"
    e_cpp = None
    for line in cpp.stdout.splitlines():
        if "energy=" in line:
            e_cpp = float(line.split("energy=")[1].split()[0])
        if "compute_dtype=" in line:
            assert "float64" in line, line

    # --- LAMMPS run 0 ---
    # run_sp.sh writes default log.lammps (not run.log)
    lmp_log = ROOT / "lammps" / "src" / "ML-UMA" / "examples" / "nacl_f64" / "log.lammps"
    subprocess.run(
        [str(ROOT / "lammps" / "src" / "ML-UMA" / "examples" / "nacl_f64" / "run_sp.sh")],
        cwd=str(ROOT / "lammps" / "src" / "ML-UMA" / "examples" / "nacl_f64"),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    e_lmp = None
    for line in lmp_log.read_text().splitlines():
        if line.startswith("Final PE"):
            e_lmp = float(line.split("=")[1].strip())
        if "compute_dtype=float64" in line:
            pass  # good

    report = {
        "ase_energy": e_ase,
        "python_export_energy": py_out["exported_energy"],
        "python_abs_energy_error_vs_ase": py_out["abs_energy_error"],
        "python_max_force_error_vs_ase": py_out["max_force_error"],
        "cpp_energy": e_cpp,
        "cpp_abs_energy_error_vs_ase": None if e_cpp is None else abs(e_cpp - e_ase),
        "lammps_energy": e_lmp,
        "lammps_abs_energy_error_vs_ase": None if e_lmp is None else abs(e_lmp - e_ase),
        "python_passed": py_out["passed"],
        # C++ uses engine CPU NL; allow looser gate than FairChem-graph Python path
        "cpp_energy_tol": 1e-3,
        "cpp_passed": e_cpp is not None and abs(e_cpp - e_ase) < 1e-3,
        "lammps_passed": e_lmp is not None and abs(e_lmp - e_ase) < 1e-3,
    }
    report["passed"] = bool(
        report["python_passed"] and report["cpp_passed"] and report["lammps_passed"]
    )
    print(json.dumps(report, indent=2))
    out = ROOT / "lammps" / "src" / "ML-UMA" / "examples" / "nacl_f64" / "parity_f64_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
