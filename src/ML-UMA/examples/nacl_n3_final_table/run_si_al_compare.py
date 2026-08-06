#!/usr/bin/env python3
"""Si diamond + Al FCC single-point compare (same protocol as NaCl final table).

ASE FP64 / FairChem fix-external / uma/kk double / uma/kk mixed.
Lattice scale 1.01, ideal sites. Updates final_table.json systems block.
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

os.environ.setdefault("PYTHONUNBUFFERED", "1")
from ase import Atoms
from ase.build import bulk
from ase.data import atomic_masses, chemical_symbols

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXAMPLES))
from _repo import find_uma_lmp_root  # noqa: E402

ROOT = find_uma_lmp_root()
sys.path.insert(0, str(ROOT / "uma-engine" / "python"))
from common import inference_settings_with_dtype  # noqa: E402

CKPT = "/mnt/d/workdir/uma-cache/uma-s-1p2.pt"
ART_F64 = ROOT / "uma-engine" / "artifacts" / "uma-s-1p2-omat-f64"
ART_MIX = ROOT / "uma-engine" / "artifacts" / "uma-s-1p2-omat"
OUT_DIR = Path(__file__).resolve().parent


def make_si(n: int = 3, a: float = 5.43, scale: float = 1.01) -> Atoms:
    atoms = bulk("Si", "diamond", a=a, cubic=True) * (n, n, n)
    atoms.set_cell(atoms.cell.array * scale, scale_atoms=True)
    return atoms


def make_al(n: int = 3, a: float = 4.05, scale: float = 1.01) -> Atoms:
    atoms = bulk("Al", "fcc", a=a, cubic=True) * (n, n, n)
    atoms.set_cell(atoms.cell.array * scale, scale_atoms=True)
    return atoms


def write_data(atoms: Atoms, path: Path, title: str) -> list[str]:
    """Write LAMMPS data; type i <-> unique Z in sorted order. Returns element symbols."""
    Z = atoms.get_atomic_numbers()
    uniq = sorted(set(int(z) for z in Z))
    z_to_type = {z: i + 1 for i, z in enumerate(uniq)}
    types = np.array([z_to_type[int(z)] for z in Z], dtype=np.int32)
    symbols = [chemical_symbols[z] for z in uniq]
    cell = atoms.cell.array
    lx, ly, lz = float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])
    pos = atoms.get_positions()
    lines = [
        title,
        "",
        f"{len(atoms)} atoms",
        f"{len(uniq)} atom types",
        "",
        f"0.0 {lx:.16f} xlo xhi",
        f"0.0 {ly:.16f} ylo yhi",
        f"0.0 {lz:.16f} zlo zhi",
        "",
        "Masses",
        "",
    ]
    for i, z in enumerate(uniq, 1):
        lines.append(f"{i} {atomic_masses[z]:.8f}")
    lines += ["", "Atoms # atomic", ""]
    for i, (t, p) in enumerate(zip(types, pos), 1):
        lines.append(f"{i} {t} {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}")
    path.write_text("\n".join(lines) + "\n")
    return symbols


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
    vesin = ROOT / "uma-engine" / "third_party" / "vesin" / "lib"
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


def fp64_settings(*, external_graph: bool):
    settings = inference_settings_with_dtype("float64")
    settings.external_graph_gen = external_graph
    return settings


def run_ase(atoms: Atoms, n_timing: int) -> dict:
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    settings = fp64_settings(external_graph=True)
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
        "path": "ASE FP64",
        "energy_eV": e,
        "forces": f,
        "ms_per_eval": float(np.mean(times) * 1e3),
        "load_s": load_s,
    }


def run_fairchem_lammps(atoms: Atoms, n_timing: int, work: Path, title: str) -> dict:
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.lammps.lammps_fc import run_lammps_with_fairchem

    settings = fp64_settings(external_graph=False)
    data = work / "data.lmp"
    write_data(atoms, data, title)
    inp = work / "in.fc"
    inp.write_text(
        f"""units metal
atom_style atomic
boundary p p p
read_data {data.name}
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
        "path": "FairChem fix external",
        "energy_eV": e,
        "forces": f,
        "ms_per_eval": float(np.mean(steady) * 1e3),
        "load_s": load_s,
    }


def run_uma_kk(
    atoms: Atoms,
    *,
    precision: str,
    artifact: Path,
    symbols: list[str],
    n_timing_steps: int,
    work: Path,
    title: str,
) -> dict:
    data = work / "data.lmp"
    write_data(atoms, data, title)
    dump = work / "forces.dump"
    log_sp = work / "log.sp"
    log_nve = work / "log.nve"
    env = setup_ld_path()
    lmp = ROOT / "lammps" / "build-uma" / "lmp"
    el = " ".join(symbols)

    inp_sp = work / "in.sp"
    inp_sp.write_text(
        f"""units metal
atom_style atomic
boundary p p p
read_data {data.name}
pair_style uma/kk precision {precision}
pair_coeff * * {artifact} {el}
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
            f"uma/kk {precision} SP failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
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
    rows = np.array(rows)
    f = rows[np.argsort(rows[:, 0]), 1:4]

    inp_nve = work / "in.nve"
    inp_nve.write_text(
        f"""units metal
atom_style atomic
boundary p p p
read_data {data.name}
pair_style uma/kk precision {precision}
pair_coeff * * {artifact} {el}
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
            f"uma/kk {precision} NVE failed:\n{proc2.stdout[-2000:]}\n{proc2.stderr[-2000:]}"
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
    ms = (pair_s / n_timing_steps) * 1e3 if pair_s else None
    return {
        "path": f"uma/kk precision {precision}",
        "energy_eV": e,
        "forces": f,
        "ms_per_eval": ms,
        "pair_section_s": pair_s,
    }


def row_from(path_result: dict, f_ase: np.ndarray) -> dict:
    e = path_result["energy_eV"]
    e_ase = None  # filled by caller via abs_dE
    stats = force_stats(f_ase, path_result["forces"]) if path_result.get("forces") is not None else {}
    return {
        "path": path_result["path"],
        "energy_eV": e,
        "ms_per_eval": path_result["ms_per_eval"],
        **{k: stats.get(k, 0.0) for k in (
            "force_mae", "force_rmse", "force_max_abs", "force_max_norm_per_atom", "cosine"
        )},
    }


def compare_system(
    *,
    key: str,
    description: str,
    atoms: Atoms,
    npz_name: str,
    n_timing: int = 5,
) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Z = atoms.get_atomic_numbers()
    uniq = sorted(set(int(z) for z in Z))
    symbols = [chemical_symbols[z] for z in uniq]
    title = f"{key} compare"

    np.savez(
        OUT_DIR / npz_name,
        numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        cell=atoms.cell.array,
    )
    print(f"\n=== {description}  natoms={len(atoms)}  symbols={symbols} ===", flush=True)

    ase = run_ase(atoms, n_timing)
    print(f"ASE   E={ase['energy_eV']:.10f}  {ase['ms_per_eval']:.1f} ms", flush=True)
    f_ase = ase["forces"]

    with tempfile.TemporaryDirectory(prefix=f"fc_{key}_", dir=OUT_DIR) as td:
        fc = run_fairchem_lammps(atoms, n_timing, Path(td), title)
    print(f"FC    E={fc['energy_eV']:.10f}  {fc['ms_per_eval']:.1f} ms", flush=True)

    with tempfile.TemporaryDirectory(prefix=f"uma64_{key}_", dir=OUT_DIR) as td:
        uma64 = run_uma_kk(
            atoms,
            precision="double",
            artifact=ART_F64,
            symbols=symbols,
            n_timing_steps=n_timing,
            work=Path(td),
            title=title,
        )
    print(f"uma64 E={uma64['energy_eV']:.10f}  {uma64['ms_per_eval']:.1f} ms", flush=True)

    with tempfile.TemporaryDirectory(prefix=f"umamix_{key}_", dir=OUT_DIR) as td:
        umamix = run_uma_kk(
            atoms,
            precision="mixed",
            artifact=ART_MIX,
            symbols=symbols,
            n_timing_steps=n_timing,
            work=Path(td),
            title=title,
        )
    print(f"mixed E={umamix['energy_eV']:.10f}  {umamix['ms_per_eval']:.1f} ms", flush=True)

    def finish(r: dict, is_ref: bool = False) -> dict:
        out = {
            "path": r["path"],
            "energy_eV": r["energy_eV"],
            "ms_per_eval": r["ms_per_eval"],
        }
        if is_ref:
            out.update(
                abs_dE_vs_ase_f64=0.0,
                force_mae=0.0,
                force_rmse=0.0,
                force_max_abs=0.0,
                force_max_norm_per_atom=0.0,
                cosine=1.0,
            )
        else:
            out["abs_dE_vs_ase_f64"] = abs(r["energy_eV"] - ase["energy_eV"])
            out.update(force_stats(f_ase, r["forces"]))
        return out

    block = {
        "system": description,
        "natoms": len(atoms),
        "symbols": symbols,
        "structure": npz_name,
        "rows": [
            finish(ase, is_ref=True),
            finish(fc),
            finish(uma64),
            finish(umamix),
        ],
    }
    np.savez(
        OUT_DIR / f"forces_{key}.npz",
        ase=f_ase,
        fairchem_lammps=fc["forces"],
        uma_kk_double=uma64["forces"],
        uma_kk_mixed=umamix["forces"],
    )
    (OUT_DIR / f"{key}_block.json").write_text(json.dumps(block, indent=2) + "\n")
    print(f"wrote {OUT_DIR / f'{key}_block.json'}", flush=True)
    return block


def main() -> int:
    n_timing = 5
    gpu = torch.cuda.get_device_name(0)

    # Preserve existing NaCl block from current final_table.json
    table_path = OUT_DIR / "final_table.json"
    old = json.loads(table_path.read_text()) if table_path.exists() else {}
    if "systems" in old:
        nacl = old["systems"].get("nacl_n3_a1p01")
    elif "rows" in old:
        nacl = {
            "system": old.get("system"),
            "natoms": old.get("natoms"),
            "symbols": ["Na", "Cl"],
            "structure": old.get("structure", "structure_a1p01.npz"),
            "rows": old["rows"],
        }
    else:
        nacl = None

    only = os.environ.get("ONLY_SYSTEM", "").strip().lower()  # si|al|both|""

    si = None
    al = None
    if only in ("", "both", "si"):
        si = compare_system(
            key="si",
            description="Si 3x3x3 diamond a=5.484300 Å (=5.43*1.01), ideal sites (ELASTIC lattice)",
            atoms=make_si(3, 5.43, 1.01),
            npz_name="structure_si_a1p01.npz",
            n_timing=n_timing,
        )
    else:
        p = OUT_DIR / "si_block.json"
        if p.exists():
            si = json.loads(p.read_text())
            print(f"reusing {p}", flush=True)

    if only in ("", "both", "al"):
        # 3x3x3 = 108 atoms — 4x4x4 FP64 previously hung/thrashed on Titan V
        al = compare_system(
            key="al",
            description="Al 3x3x3 fcc a=4.090500 Å (=4.05*1.01), ideal sites",
            atoms=make_al(3, 4.05, 1.01),
            npz_name="structure_al_a1p01.npz",
            n_timing=n_timing,
        )
    else:
        p = OUT_DIR / "al_block.json"
        if p.exists():
            al = json.loads(p.read_text())
            print(f"reusing {p}", flush=True)

    systems = {}
    if nacl is not None:
        systems["nacl_n3_a1p01"] = nacl
    if si is not None:
        systems["si_diamond_a1p01"] = si
    if al is not None:
        systems["al_fcc_a1p01"] = al
    if "si_diamond_a1p01" not in systems or "al_fcc_a1p01" not in systems:
        missing = [k for k in ("si_diamond_a1p01", "al_fcc_a1p01") if k not in systems]
        raise SystemExit(f"missing systems {missing}; set ONLY_SYSTEM or provide *_block.json")

    report = {
        "gpu": gpu,
        "reference": "ASE FAIRChemCalculator FP64",
        "note": (
            "Single-point E+F. Lattice scale 1.01, ideal sites. "
            "ASE has no mixed mode; mixed uma/kk vs ASE FP64. "
            "Post-fix: denorm_energy preserves compute dtype. "
            "Si from LAMMPS ELASTIC diamond lattice; Al fcc metal."
        ),
        "denorm_fix": "uma-engine/src/postprocess.cpp preserves FP32/FP64 dtype",
        "systems": systems,
    }
    # Keep flat NaCl top-level fields for backward compat with older readers
    if nacl is not None:
        report["system"] = nacl["system"]
        report["natoms"] = nacl["natoms"]
        report["structure"] = nacl.get("structure")
        report["rows"] = nacl["rows"]

    table_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {table_path}", flush=True)
    print(json.dumps({k: [(r["path"], r["abs_dE_vs_ase_f64"], r["ms_per_eval"]) for r in v["rows"]]
                      for k, v in systems.items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
