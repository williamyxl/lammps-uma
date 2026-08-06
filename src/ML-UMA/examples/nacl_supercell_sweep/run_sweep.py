#!/usr/bin/env python3
"""Sweep NxNxN NaCl supercells with uma/kk; find max N that fits on Titan V.

For each N in [lo, hi]:
  - build rocksalt supercell (8*N^3 atoms)
  - run LAMMPS pair_style uma/kk precision mixed, run 0 (+ optional NVE steps)
  - record success/OOM and peak nvidia-smi GPU memory
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.data import atomic_masses

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXAMPLES))
from _repo import find_uma_engine_root, find_uma_lmp_root  # noqa: E402

ROOT = find_uma_lmp_root()
ENGINE = find_uma_engine_root()


def write_data(atoms, path: Path) -> None:
    Z = atoms.get_atomic_numbers()
    types = np.where(Z == 11, 1, 2)
    cell = atoms.cell.array
    lx, ly, lz = float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])
    # Orthogonal rocksalt only
    pos = atoms.get_positions()
    lines = [
        f"NaCl {path.stem}",
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


def make_nacl(n: int, a: float = 5.64):
    return bulk("NaCl", "rocksalt", a=a, cubic=True) * (n, n, n)


def nvidia_mem_mib() -> tuple[float, float, float]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    total, used, free = [float(x) for x in out.split(",")]
    return total, used, free


def run_one(
    n: int,
    work: Path,
    lmp: Path,
    artifact: Path,
    nve_steps: int,
    poll_s: float,
) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    atoms = make_nacl(n)
    natoms = len(atoms)
    data = work / f"data.nacl_{n}"
    write_data(atoms, data)
    inp = work / f"in.n{n}"
    log = work / f"log.n{n}"
    out = work / f"out.n{n}"
    nve = ""
    if nve_steps > 0:
        nve = f"""
velocity all create 0.0 1
timestep 0.001
fix 1 all nve
thermo 0
run 0
run {nve_steps}
"""
    else:
        nve = """
thermo 1
thermo_style custom step pe
run 0
"""
    inp.write_text(
        f"""units metal
atom_style atomic
boundary p p p
read_data {data.name}
pair_style uma/kk precision mixed
pair_coeff * * {artifact} Na Cl
newton off
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
{nve}
"""
    )

    total0, used0, free0 = nvidia_mem_mib()
    peak_used = used0
    stop = threading.Event()

    def poller():
        nonlocal peak_used
        while not stop.wait(poll_s):
            try:
                _, used, _ = nvidia_mem_mib()
                peak_used = max(peak_used, used)
            except Exception:
                pass

    thr = threading.Thread(target=poller, daemon=True)
    thr.start()
    t0 = time.perf_counter()
    env = os.environ.copy()
    # Prefer fresh CUDA allocator bookkeeping
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.run(
        [str(lmp), "-k", "on", "g", "1", "-sf", "kk", "-in", inp.name, "-log", log.name],
        cwd=work,
        env=env,
        stdout=open(out, "w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    wall = time.perf_counter() - t0
    stop.set()
    thr.join(timeout=2.0)
    # final sample
    try:
        _, used1, free1 = nvidia_mem_mib()
        peak_used = max(peak_used, used1)
    except Exception:
        used1, free1 = peak_used, float("nan")

    text = out.read_text(errors="replace") + "\n" + log.read_text(errors="replace")
    oom = bool(
        re.search(
            r"out of memory|CUDA error|cudaMalloc|c10::Error|bad_alloc|Killed",
            text,
            re.I,
        )
    )
    pe = None
    m = re.search(r"^\s*0\s+([-+0-9.eE]+)\s*$", text, re.M)
    if m:
        try:
            pe = float(m.group(1))
        except ValueError:
            pass
    pair_s = None
    for line in text.splitlines():
        if line.strip().startswith("Pair") and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            try:
                v = float(parts[2])
                if v > 0:
                    pair_s = v
            except Exception:
                pass

    ok = proc.returncode == 0 and not oom
    return {
        "N": n,
        "natoms": natoms,
        "ok": ok,
        "oom": oom or (proc.returncode != 0 and "memory" in text.lower()),
        "returncode": proc.returncode,
        "wall_s": wall,
        "energy": pe,
        "pair_s": pair_s,
        "gpu_total_MiB": total0,
        "gpu_used_before_MiB": used0,
        "gpu_free_before_MiB": free0,
        "gpu_peak_used_MiB": peak_used,
        "gpu_used_after_MiB": used1,
        "gpu_delta_peak_MiB": peak_used - used0,
        "log": str(log),
        "out": str(out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=6)
    ap.add_argument("--n-max", type=int, default=12)
    ap.add_argument("--nve-steps", type=int, default=2, help="extra MD steps after run 0")
    ap.add_argument("--poll-s", type=float, default=0.05)
    ap.add_argument(
        "--lmp",
        type=Path,
        default=ROOT / "lammps" / "build-uma" / "lmp",
    )
    ap.add_argument(
        "--artifact",
        type=Path,
        default=ENGINE / "artifacts" / "uma-s-1p2-omat",
    )
    ap.add_argument(
        "--work",
        type=Path,
        default=ROOT / "lammps" / "src" / "ML-UMA" / "examples" / "nacl_supercell_sweep" / "work",
    )
    args = ap.parse_args()

    vesin = ENGINE / "third_party" / "vesin" / "lib"
    torch_lib = subprocess.check_output(
        ["python", "-c", "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))"],
        text=True,
    ).strip()
    os.environ["LD_LIBRARY_PATH"] = (
        f"{vesin}:{torch_lib}"
        + (f":{os.environ['LD_LIBRARY_PATH']}" if os.environ.get("LD_LIBRARY_PATH") else "")
    )

    results = []
    print(
        f"GPU sweep NaCl NxNxN N={args.n_min}..{args.n_max} "
        f"artifact={args.artifact} lmp={args.lmp}"
    )
    total, used, free = nvidia_mem_mib()
    print(f"GPU before sweep: total={total:.0f} used={used:.0f} free={free:.0f} MiB")

    max_fit = None
    for n in range(args.n_min, args.n_max + 1):
        print(f"\n=== N={n}  natoms={8 * n**3} ===", flush=True)
        r = run_one(n, args.work, args.lmp, args.artifact, args.nve_steps, args.poll_s)
        results.append(r)
        status = "OK" if r["ok"] else ("OOM" if r["oom"] else f"FAIL rc={r['returncode']}")
        print(
            f"  {status}  wall={r['wall_s']:.1f}s  "
            f"peak_used={r['gpu_peak_used_MiB']:.0f} MiB  "
            f"delta={r['gpu_delta_peak_MiB']:.0f} MiB  "
            f"E={r['energy']}",
            flush=True,
        )
        if r["ok"]:
            max_fit = n
        else:
            # still continue to map the wall; larger N may also fail
            pass

    report = {
        "gpu": "NVIDIA TITAN V",
        "gpu_total_MiB": total,
        "gpu_used_before_sweep_MiB": used,
        "precision": "mixed",
        "neighbor_list": "vesin CUDA",
        "n_range": [args.n_min, args.n_max],
        "nve_steps": args.nve_steps,
        "max_N_fit": max_fit,
        "max_natoms_fit": None if max_fit is None else 8 * max_fit**3,
        "results": results,
    }
    out_json = args.work.parent / "sweep_report.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print("\n==== SUMMARY ====")
    print(f"max N that completed: {max_fit}  (natoms={report['max_natoms_fit']})")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
