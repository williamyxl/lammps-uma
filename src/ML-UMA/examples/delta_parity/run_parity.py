#!/usr/bin/env python3
"""Delta A100 4-path parity: total energy + per-atom forces.

Paths
-----
1. ASE FairChem API FP64
2. FairChem LAMMPS fix external (conda ``lmp``)
3. Local ``pair_style uma/kk precision double``
4. Local ``pair_style uma/kk precision mixed``

Systems (frozen under ``structures/``)
--------------------------------------
nacl/al/si 3x3x3, nacl4/si4 (4x4x4), al5 (5x5x5), plus larger n7/n8 sizes
(Unif[-0.10,0.10] Å atomic rattle only, seed=0; no lattice scale)

Environment
-----------
ONLY_SYSTEM=nacl|al|si|si4|nacl4|nacl5|nacl6|al5|nacl7|si7|nacl8|al8|si8|nsafe|n7|n8|all
             (nsafe = nacl4,al5,si4 — ASE FP64-safe ~500 atoms on A100 40GB)
SKIP_BUILD=1                 reuse frozen npz when present
UMA_CHECKPOINT=...           default: workdir/uma-cache/uma-s-1p2.pt
LMP_UMA=...                  local Kokkos+ML-UMA binary
LMP_FC=...                   conda FairChem lmp (used via python lammps API / PATH)
N_TIMING=5
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def release_cuda(tag: str = "") -> None:
    """Drop Python-held CUDA tensors so the next path (incl. uma/kk subprocess) can use VRAM."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    free, total = torch.cuda.mem_get_info()
    used = total - free
    label = f" ({tag})" if tag else ""
    print(
        f"GPU VRAM after release{label}: "
        f"used={used / (1024**3):.2f} GiB  free={free / (1024**3):.2f} GiB / "
        f"{total / (1024**3):.2f} GiB",
        flush=True,
    )

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from ase import Atoms
from ase.data import atomic_masses, chemical_symbols

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXAMPLES))
from _repo import (  # noqa: E402
    find_checkpoint,
    find_fairchem_lmp_binary,
    find_lammps_root,
    find_uma_engine_root,
    find_uma_lmp_binary,
)

ENGINE = find_uma_engine_root()
LAMMPS_ROOT = find_lammps_root()
sys.path.insert(0, str(ENGINE / "python"))
from common import inference_settings_with_dtype  # noqa: E402

OUT = Path(__file__).resolve().parent
STRUCT = OUT / "structures"
RESULTS = OUT / "results"

ART_F64 = ENGINE / "artifacts" / "uma-s-1p2-omat-f64"
ART_MIX = ENGINE / "artifacts" / "uma-s-1p2-omat"

PERTURB_DELTA_A = 0.10
PERTURB_SEED = 0
PERTURB_MODE = "uniform_box"

SYSTEM_SPECS = {
    "nacl": {
        "key": "nacl_n3_rattle",
        "npz": "structure_nacl_rattle.npz",
        "description": (
            f"NaCl 3x3x3 rocksalt a=5.64 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "al": {
        "key": "al_fcc_rattle",
        "npz": "structure_al_rattle.npz",
        "description": (
            f"Al 3x3x3 fcc a=4.05 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "si": {
        "key": "si_diamond_rattle",
        "npz": "structure_si_rattle.npz",
        "description": (
            f"Si 3x3x3 diamond a=5.43 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "si4": {
        "key": "si_diamond_n4_rattle",
        "npz": "structure_si4_rattle.npz",
        "description": (
            f"Si 4x4x4 diamond a=5.43 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "nacl4": {
        "key": "nacl_n4_rattle",
        "npz": "structure_nacl4_rattle.npz",
        "description": (
            f"NaCl 4x4x4 rocksalt a=5.64 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "nacl5": {
        "key": "nacl_n5_rattle",
        "npz": "structure_nacl5_rattle.npz",
        "description": (
            f"NaCl 5x5x5 rocksalt a=5.64 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "nacl6": {
        "key": "nacl_n6_rattle",
        "npz": "structure_nacl6_rattle.npz",
        "description": (
            f"NaCl 6x6x6 rocksalt a=5.64 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "al5": {
        "key": "al_fcc_n5_rattle",
        "npz": "structure_al5_rattle.npz",
        "description": (
            f"Al 5x5x5 fcc a=4.05 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "nacl7": {
        "key": "nacl_n7_rattle",
        "npz": "structure_nacl7_rattle.npz",
        "description": (
            f"NaCl 7x7x7 rocksalt a=5.64 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "si7": {
        "key": "si_diamond_n7_rattle",
        "npz": "structure_si7_rattle.npz",
        "description": (
            f"Si 7x7x7 diamond a=5.43 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "nacl8": {
        "key": "nacl_n8_rattle",
        "npz": "structure_nacl8_rattle.npz",
        "description": (
            f"NaCl 8x8x8 rocksalt a=5.64 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "al8": {
        "key": "al_fcc_n8_rattle",
        "npz": "structure_al8_rattle.npz",
        "description": (
            f"Al 8x8x8 fcc a=4.05 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
    "si8": {
        "key": "si_diamond_n8_rattle",
        "npz": "structure_si8_rattle.npz",
        "description": (
            f"Si 8x8x8 diamond a=5.43 Å, uniform-box rattle "
            f"δ={PERTURB_DELTA_A} Å seed={PERTURB_SEED}"
        ),
    },
}

DEFAULT_SYSTEMS = ("nacl", "al", "si", "si4")
# Proven ASE FP64-safe on A100 40GB (~500 atoms; 7x7x7 NaCl/Si OOMed)
NSAFE_SYSTEMS = ("nacl4", "al5", "si4")
N7_SYSTEMS = ("nacl7", "al8", "si7")
N8_SYSTEMS = ("nacl8", "al8", "si8")


def resolve_systems(only: str) -> list[str]:
    only = only.strip().lower()
    if only in ("", "all", "both"):
        return list(DEFAULT_SYSTEMS)
    if only in ("nsafe", "nacl4,al5,si4"):
        return list(NSAFE_SYSTEMS)
    if only in ("n7", "nacl7,al8,si7"):
        return list(N7_SYSTEMS)
    if only in ("n8", "nacl8,al8,si8"):
        return list(N8_SYSTEMS)
    names = [p.strip() for p in only.split(",") if p.strip()]
    for n in names:
        if n not in SYSTEM_SPECS:
            raise SystemExit(
                f"unknown system {n!r}; use "
                f"{'|'.join(SYSTEM_SPECS)}|nsafe|n7|n8|all (comma-separated ok)"
            )
    return names


def ensure_structures(names: list[str]) -> None:
    need = [n for n in names if not (STRUCT / SYSTEM_SPECS[n]["npz"]).is_file()]
    if not need and os.environ.get("SKIP_BUILD") == "1":
        return
    freeze = OUT / "freeze_structures.py"
    targets = need if need else names
    if os.environ.get("SKIP_BUILD") == "1" and not need:
        return
    subprocess.run([sys.executable, str(freeze), *targets], check=True)


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
    vesin = ENGINE / "third_party" / "vesin" / "lib"
    torch_lib = Path(torch.__path__[0]) / "lib"
    parts = [
        str(vesin),
        str(torch_lib),
        "/usr/local/cuda/lib64",
        "/usr/lib/wsl/lib",
    ]
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    # Prefer conda FairChem lmp on PATH for python `import lammps`
    try:
        fc = find_fairchem_lmp_binary()
        env["PATH"] = f"{fc.parent}:{env.get('PATH', '')}"
    except FileNotFoundError:
        pass
    return env


def fp64_settings(*, external_graph: bool):
    settings = inference_settings_with_dtype("float64")
    settings.external_graph_gen = external_graph
    return settings


def _pack(path: str, e: float, f: np.ndarray, ms: float | None, **extra) -> dict:
    return {
        "path": path,
        "energy_eV": float(e),
        "natoms": len(f),
        "forces": np.asarray(f, dtype=np.float64),
        "ms_per_eval": None if ms is None else float(ms),
        **extra,
    }


def run_ase(atoms: Atoms, ckpt: Path, n_timing: int) -> dict:
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    settings = fp64_settings(external_graph=True)
    t0 = time.perf_counter()
    predictor = load_predict_unit(str(ckpt), device="cuda", inference_settings=settings)
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

    out = _pack("ASE FairChem FP64", e, f, float(np.mean(times) * 1e3), load_s=load_s)
    a.calc = None
    del calc, predictor, a
    release_cuda("ASE")
    return out


def run_fairchem_lammps(atoms: Atoms, ckpt: Path, n_timing: int, work: Path, title: str) -> dict:
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

    # Ensure python lammps picks up conda binary/lib
    fc_lmp = find_fairchem_lmp_binary()
    os.environ["PATH"] = f"{fc_lmp.parent}:{os.environ.get('PATH', '')}"

    t0 = time.perf_counter()
    predictor = load_predict_unit(str(ckpt), device="cuda", inference_settings=settings)
    load_s = time.perf_counter() - t0

    cwd = Path.cwd()
    os.chdir(work)
    try:
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
        if hasattr(lmp, "_predictor"):
            del lmp._predictor
        lmp.close()
        del lmp
    finally:
        os.chdir(cwd)

    del predictor
    out = _pack(
        "FairChem LAMMPS fix external",
        e,
        f,
        float(np.mean(steady) * 1e3),
        load_s=load_s,
        fairchem_lmp=str(fc_lmp),
        note="lammps_fc builds cell in FP32; not a pure FP64 path",
    )
    release_cuda("FC")
    return out


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
    if not (artifact / "model_traced.pt").is_file():
        raise FileNotFoundError(
            f"missing {artifact / 'model_traced.pt'} — export with export_omat.py first"
        )
    data = work / "data.lmp"
    write_data(atoms, data, title)
    dump = work / "forces.dump"
    log_sp = work / "log.sp"
    log_nve = work / "log.nve"
    env = setup_ld_path()
    lmp = find_uma_lmp_binary()
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
    return _pack(
        f"uma/kk precision {precision}",
        e,
        f,
        ms,
        pair_section_s=pair_s,
        lmp_uma=str(lmp),
        artifact=str(artifact),
    )


def compare_system(name: str, ckpt: Path, n_timing: int) -> dict:
    spec = SYSTEM_SPECS[name]
    npz_path = STRUCT / spec["npz"]
    atoms, perturb = load_structure(npz_path)

    Z = atoms.get_atomic_numbers()
    uniq = sorted(set(int(z) for z in Z))
    symbols = [chemical_symbols[z] for z in uniq]
    title = f"{spec['key']} compare"
    key = name
    work_root = RESULTS / key
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== {spec['description']}  natoms={len(atoms)}  symbols={symbols} ===",
        flush=True,
    )
    print(f"geometry: {npz_path}", flush=True)

    ase = run_ase(atoms, ckpt, n_timing)
    print(
        f"ASE   E={ase['energy_eV']:.10f}  "
        f"{ase['ms_per_eval']:.1f} ms  |F|_max={np.max(np.abs(ase['forces'])):.3e}",
        flush=True,
    )
    f_ase = ase["forces"]
    release_cuda("pre-FC")

    for sub in ("fc", "uma_double", "uma_mixed"):
        (work_root / sub).mkdir(parents=True, exist_ok=True)

    fc = run_fairchem_lammps(atoms, ckpt, n_timing, work_root / "fc", title)
    print(
        f"FC    E={fc['energy_eV']:.10f}  {fc['ms_per_eval']:.1f} ms",
        flush=True,
    )
    release_cuda("pre-uma64")

    uma64 = run_uma_kk(
        atoms,
        precision="double",
        artifact=ART_F64,
        symbols=symbols,
        n_timing_steps=n_timing,
        work=work_root / "uma_double",
        title=title,
    )
    print(
        f"uma64 E={uma64['energy_eV']:.10f}  {uma64['ms_per_eval']:.1f} ms",
        flush=True,
    )
    release_cuda("pre-mixed")

    umamix = run_uma_kk(
        atoms,
        precision="mixed",
        artifact=ART_MIX,
        symbols=symbols,
        n_timing_steps=n_timing,
        work=work_root / "uma_mixed",
        title=title,
    )
    print(
        f"mixed E={umamix['energy_eV']:.10f}  {umamix['ms_per_eval']:.1f} ms",
        flush=True,
    )
    release_cuda("post-mixed")

    def finish(r: dict, is_ref: bool = False) -> dict:
        out = {
            "path": r["path"],
            "energy_eV": r["energy_eV"],
            "ms_per_eval": r["ms_per_eval"],
        }
        for k in ("fairchem_lmp", "lmp_uma", "artifact", "note", "load_s", "pair_section_s"):
            if k in r:
                out[k] = r[k]
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

    # Persist total energy + per-atom forces for all paths
    np.savez(
        work_root / "per_atom_forces.npz",
        numbers=atoms.get_atomic_numbers().astype(np.int32),
        positions=atoms.get_positions().astype(np.float64),
        cell=atoms.cell.array.astype(np.float64),
        energy_ase_eV=np.array(ase["energy_eV"]),
        energy_fc_eV=np.array(fc["energy_eV"]),
        energy_uma_double_eV=np.array(uma64["energy_eV"]),
        energy_uma_mixed_eV=np.array(umamix["energy_eV"]),
        forces_ase=f_ase,
        forces_fairchem_lammps=fc["forces"],
        forces_uma_kk_double=uma64["forces"],
        forces_uma_kk_mixed=umamix["forces"],
    )

    block = {
        "system": spec["description"],
        "natoms": len(atoms),
        "symbols": symbols,
        "structure": str(npz_path.relative_to(OUT)),
        "perturbation": perturb,
        "note": "Total energy (scalar) + per-atom forces. No per-atom energy.",
        "rows": [
            finish(ase, is_ref=True),
            finish(fc),
            finish(uma64),
            finish(umamix),
        ],
    }
    (work_root / f"{key}_block.json").write_text(json.dumps(block, indent=2) + "\n")
    print(f"wrote {work_root / 'per_atom_forces.npz'}", flush=True)
    return block


def main() -> int:
    n_timing = int(os.environ.get("N_TIMING", "5"))
    ckpt = find_checkpoint()
    gpu = torch.cuda.get_device_name(0)
    only = os.environ.get("ONLY_SYSTEM", "all")
    names = resolve_systems(only)

    print(f"gpu={gpu}", flush=True)
    print(f"checkpoint={ckpt}", flush=True)
    print(f"engine={ENGINE}", flush=True)
    print(f"lammps_root={LAMMPS_ROOT}", flush=True)
    print(f"LMP_UMA={find_uma_lmp_binary()}", flush=True)
    print(f"LMP_FC={find_fairchem_lmp_binary()}", flush=True)
    print(f"systems={names}", flush=True)
    print(f"ART_F64={ART_F64} exists_model={(ART_F64 / 'model_traced.pt').is_file()}", flush=True)
    print(f"ART_MIX={ART_MIX} exists_model={(ART_MIX / 'model_traced.pt').is_file()}", flush=True)

    ensure_structures(names)
    RESULTS.mkdir(parents=True, exist_ok=True)

    systems = {}
    for name in names:
        block = compare_system(name, ckpt, n_timing=n_timing)
        systems[SYSTEM_SPECS[name]["key"]] = block
        release_cuda(f"post-system:{name}")

    table_path = RESULTS / "parity_table.json"
    if table_path.exists():
        old = json.loads(table_path.read_text())
        for k, v in (old.get("systems") or {}).items():
            systems.setdefault(k, v)

    report = {
        "gpu": gpu,
        "checkpoint": str(ckpt),
        "reference": "ASE FAIRChemCalculator FP64",
        "lmp_uma": str(find_uma_lmp_binary()),
        "lmp_fc": str(find_fairchem_lmp_binary()),
        "artifacts": {"double": str(ART_F64), "mixed": str(ART_MIX)},
        "perturbation": {
            "mode": PERTURB_MODE,
            "delta_A": PERTURB_DELTA_A,
            "seed": PERTURB_SEED,
        },
        "note": (
            "Same frozen rattled geometry for all 4 paths. Outputs: total energy "
            "(scalar) and per-atom force arrays in results/<sys>/per_atom_forces.npz."
        ),
        "systems": systems,
    }
    table_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {table_path}", flush=True)
    summary = {
        k: [
            (r["path"], r["energy_eV"], r["abs_dE_vs_ase_f64"], r.get("cosine"))
            for r in v["rows"]
        ]
        for k, v in systems.items()
    }
    print(json.dumps(summary, indent=2), flush=True)

    # Markdown + Cursor canvas reports
    try:
        from write_reports import main as write_reports_main

        write_reports_main([str(table_path)])
    except Exception as exc:
        print(f"warning: write_reports failed: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
