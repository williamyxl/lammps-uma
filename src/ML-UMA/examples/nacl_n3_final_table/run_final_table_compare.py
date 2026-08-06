#!/usr/bin/env python3
"""Final-table compare: ASE FP64 / FairChem fix-ext / uma/kk double / mixed.

Geometry protocol (same for every path)
---------------------------------------
1. Build perfect crystal (ASE ``bulk``).
2. Isotropically scale cell and fractional positions by ``LATTICE_SCALE`` (1.01).
3. Atomic perturbation — applied **once**, then frozen to npz:
     mode   = uniform_box
     delta  = 0.05 Å
     rule   = each Cartesian component independently ~ Unif[-delta, +delta]
     rng    = numpy Generator(PCG64(seed)), seed = 0 for all systems
     wrap   = wrap atoms back into the periodic cell after the rattle
4. All four evaluators load that **same** npz (no further geometry changes).

Environment
-----------
ONLY_SYSTEM=nacl|si|si4|al|all   (default all)
SKIP_BUILD=1                 reuse existing rattled npz files
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

# --- perturbation scheme (frozen contract) ---
LATTICE_SCALE = 1.01
PERTURB_DELTA_A = 0.10  # Å, half-width of uniform box (was 0.05; larger |F| for clearer force SNR)
PERTURB_SEED = 0
PERTURB_MODE = "uniform_box"


def perturb_atoms(
    atoms: Atoms,
    *,
    delta_A: float = PERTURB_DELTA_A,
    seed: int = PERTURB_SEED,
) -> dict:
    """Displace every atom once; return metadata describing the draw.

    Each Cartesian component is drawn independently from Unif[-delta, +delta].
    Positions are wrapped into the cell afterward. The input ``atoms`` is
    modified in place; call this exactly once before freezing to npz.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    disp = rng.uniform(-delta_A, delta_A, size=atoms.positions.shape)
    atoms.positions = atoms.positions + disp
    atoms.wrap()
    return {
        "mode": PERTURB_MODE,
        "delta_A": float(delta_A),
        "seed": int(seed),
        "rng": "numpy.random.Generator(PCG64(seed))",
        "distribution": f"Unif[-{delta_A}, +{delta_A}] Å per Cartesian component",
        "wrap": True,
        "disp_max_abs_A": float(np.max(np.abs(disp))),
        "disp_rms_A": float(np.sqrt(np.mean(disp**2))),
    }


def build_ideal(name: str) -> Atoms:
    if name == "nacl":
        atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (3, 3, 3)
    elif name == "si":
        atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (3, 3, 3)
    elif name == "si4":
        atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (4, 4, 4)
    elif name == "al":
        atoms = bulk("Al", "fcc", a=4.05, cubic=True) * (3, 3, 3)
    else:
        raise ValueError(name)
    atoms.set_cell(atoms.cell.array * LATTICE_SCALE, scale_atoms=True)
    return atoms


def freeze_structure(name: str, npz_path: Path) -> tuple[Atoms, dict]:
    """Build → scale → rattle once → save npz. Returns (atoms, perturb_meta)."""
    atoms = build_ideal(name)
    meta = perturb_atoms(atoms)
    meta["lattice_scale"] = LATTICE_SCALE
    np.savez(
        npz_path,
        numbers=atoms.get_atomic_numbers().astype(np.int32),
        positions=atoms.get_positions().astype(np.float64),
        cell=atoms.cell.array.astype(np.float64),
        perturb_mode=np.array(meta["mode"]),
        perturb_delta_A=np.array(meta["delta_A"]),
        perturb_seed=np.array(meta["seed"]),
        lattice_scale=np.array(meta["lattice_scale"]),
    )
    print(
        f"froze {npz_path.name}: natoms={len(atoms)}  "
        f"δ={meta['delta_A']} Å  seed={meta['seed']}  "
        f"disp_rms={meta['disp_rms_A']:.4f} Å",
        flush=True,
    )
    return atoms, meta


def load_structure(npz_path: Path) -> tuple[Atoms, dict]:
    d = np.load(npz_path, allow_pickle=False)
    atoms = Atoms(
        numbers=d["numbers"],
        positions=d["positions"],
        cell=d["cell"],
        pbc=True,
    )
    meta = {
        "mode": str(d["perturb_mode"]) if "perturb_mode" in d.files else PERTURB_MODE,
        "delta_A": float(d["perturb_delta_A"]) if "perturb_delta_A" in d.files else PERTURB_DELTA_A,
        "seed": int(d["perturb_seed"]) if "perturb_seed" in d.files else PERTURB_SEED,
        "lattice_scale": float(d["lattice_scale"]) if "lattice_scale" in d.files else LATTICE_SCALE,
        "source_npz": npz_path.name,
    }
    return atoms, meta


def write_data(atoms: Atoms, path: Path, title: str) -> list[str]:
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
        "f_ref_max_abs": float(np.max(np.abs(f_ref))),
        "f_ref_rms": float(np.sqrt(np.mean(f_ref**2))),
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


SYSTEM_SPECS = {
    "nacl": {
        "key": "nacl_n3_rattle",
        "npz": "structure_nacl_rattle.npz",
        "description": (
            f"NaCl 3x3x3 rocksalt a={5.64 * LATTICE_SCALE:.6f} Å "
            f"(=5.64*{LATTICE_SCALE}), uniform-box rattle δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "si": {
        "key": "si_diamond_rattle",
        "npz": "structure_si_rattle.npz",
        "description": (
            f"Si 3x3x3 diamond a={5.43 * LATTICE_SCALE:.6f} Å "
            f"(=5.43*{LATTICE_SCALE}), uniform-box rattle δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "si4": {
        "key": "si_diamond_n4_rattle",
        "npz": "structure_si4_rattle.npz",
        "description": (
            f"Si 4x4x4 diamond a={5.43 * LATTICE_SCALE:.6f} Å "
            f"(=5.43*{LATTICE_SCALE}), uniform-box rattle δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "al": {
        "key": "al_fcc_rattle",
        "npz": "structure_al_rattle.npz",
        "description": (
            f"Al 3x3x3 fcc a={4.05 * LATTICE_SCALE:.6f} Å "
            f"(=4.05*{LATTICE_SCALE}), uniform-box rattle δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
}


def compare_system(name: str, n_timing: int = 5) -> dict:
    spec = SYSTEM_SPECS[name]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / spec["npz"]

    if os.environ.get("SKIP_BUILD") == "1" and npz_path.exists():
        atoms, perturb = load_structure(npz_path)
        print(f"reusing frozen geometry {npz_path.name}", flush=True)
    else:
        atoms, perturb = freeze_structure(name, npz_path)

    # Reload from disk so every path sees exactly what was written
    atoms, perturb = load_structure(npz_path)

    Z = atoms.get_atomic_numbers()
    uniq = sorted(set(int(z) for z in Z))
    symbols = [chemical_symbols[z] for z in uniq]
    title = f"{spec['key']} compare"
    key = name

    print(
        f"\n=== {spec['description']}  natoms={len(atoms)}  symbols={symbols} ===",
        flush=True,
    )
    print(f"geometry source: {npz_path}  (shared by all 4 paths)", flush=True)

    ase = run_ase(atoms, n_timing)
    print(
        f"ASE   E={ase['energy_eV']:.10f}  {ase['ms_per_eval']:.1f} ms  "
        f"|F|_max={np.max(np.abs(ase['forces'])):.3e}",
        flush=True,
    )
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
                f_ref_max_abs=float(np.max(np.abs(f_ase))),
                f_ref_rms=float(np.sqrt(np.mean(f_ase**2))),
            )
        else:
            out["abs_dE_vs_ase_f64"] = abs(r["energy_eV"] - ase["energy_eV"])
            out.update(force_stats(f_ase, r["forces"]))
        return out

    block = {
        "system": spec["description"],
        "natoms": len(atoms),
        "symbols": symbols,
        "structure": spec["npz"],
        "perturbation": perturb,
        "rows": [
            finish(ase, is_ref=True),
            finish(fc),
            finish(uma64),
            finish(umamix),
        ],
    }
    np.savez(
        OUT_DIR / f"forces_{key}_rattle.npz",
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
    only = os.environ.get("ONLY_SYSTEM", "all").strip().lower()
    names = ["nacl", "si", "si4", "al"] if only in ("", "all", "both") else [only]
    for n in names:
        if n not in SYSTEM_SPECS:
            raise SystemExit(f"unknown ONLY_SYSTEM={n!r}; use nacl|si|si4|al|all")

    systems = {}
    for name in names:
        block = compare_system(name, n_timing=n_timing)
        systems[SYSTEM_SPECS[name]["key"]] = block

    table_path = OUT_DIR / "final_table.json"
    # Always keep other systems already in the table when running a subset
    if table_path.exists():
        old = json.loads(table_path.read_text())
        for k, v in (old.get("systems") or {}).items():
            systems.setdefault(k, v)

    report = {
        "gpu": gpu,
        "reference": "ASE FAIRChemCalculator FP64",
        "perturbation": {
            "mode": PERTURB_MODE,
            "delta_A": PERTURB_DELTA_A,
            "seed": PERTURB_SEED,
            "lattice_scale": LATTICE_SCALE,
            "rule": (
                "Build ideal crystal → scale lattice×positions by lattice_scale → "
                "add Unif[-delta,delta] Å per Cartesian component (PCG64 seed) once → "
                "wrap into cell → freeze npz → all 4 paths load that npz."
            ),
        },
        "note": (
            "Single-point E+F. Same frozen rattled geometry for ASE / FairChem / "
            "uma double / uma mixed. ASE has no mixed mode; mixed vs ASE FP64. "
            "Post-fix: denorm_energy preserves compute dtype."
        ),
        "denorm_fix": "uma-engine/src/postprocess.cpp preserves FP32/FP64 dtype",
        "systems": systems,
    }
    # Flat NaCl mirror for older readers
    nacl_key = "nacl_n3_rattle"
    if nacl_key in systems:
        report["system"] = systems[nacl_key]["system"]
        report["natoms"] = systems[nacl_key]["natoms"]
        report["structure"] = systems[nacl_key]["structure"]
        report["rows"] = systems[nacl_key]["rows"]

    table_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {table_path}", flush=True)
    summary = {
        k: [(r["path"], r["abs_dE_vs_ase_f64"], r.get("cosine"), r["ms_per_eval"]) for r in v["rows"]]
        for k, v in systems.items()
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
