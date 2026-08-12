#!/usr/bin/env python3
"""Phase-P0 LAMMPS + LibTorch-UMA run (single point E+F, NVT-300K timing).

pair_style uma precision double (single rank), full-precision output so parity
is not floored by %g. Writes <sys>_uma.json (energy, timing) and <sys>_uma.npz
(per-atom forces sorted by id).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from p0_common import (
    ART_F64,
    NVT_STEPS,
    NVT_TEMP_K,
    ROOT,
    SYSTEMS,
    results_dir,
)


def find_lmp() -> Path:
    env = os.environ.get("LMP_UMA")
    if env:
        return Path(env).expanduser().resolve()
    cand = ROOT / "build-uma" / "lmp"
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError("set LMP_UMA or build build-uma/lmp")


def parse_loop_blocks(text: str) -> list[dict]:
    loop_re = re.compile(
        r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps"
    )
    pair_re = re.compile(
        r"^Pair\s+\|\s+([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)", re.M
    )
    out = []
    for m in loop_re.finditer(text):
        nxt = loop_re.search(text, m.end())
        chunk = text[m.end(): nxt.start() if nxt else len(text)]
        pm = pair_re.search(chunk)
        n = int(m.group(3))
        loop_s = float(m.group(1))
        pair_s = float(pm.group(1)) if pm else None
        out.append({
            "nsteps": n,
            "loop_s": loop_s,
            "pair_s": pair_s,
            "loop_ms_per_step": loop_s / n * 1e3 if n else None,
            "pair_ms_per_step": pair_s / n * 1e3 if (pair_s and n) else None,
        })
    return out


def parse_dump_forces(dump: Path) -> np.ndarray:
    lines = dump.read_text().splitlines()
    s = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS"))
    cols = lines[s].split()[2:]
    i_id, i_fx = cols.index("id"), cols.index("fx")
    rows = []
    for l in lines[s + 1:]:
        if l.startswith("ITEM:"):
            break
        p = l.split()
        if len(p) < len(cols):
            continue
        rows.append((float(p[i_id]), float(p[i_fx]), float(p[i_fx + 1]), float(p[i_fx + 2])))
    arr = np.array(rows)
    arr = arr[np.argsort(arr[:, 0])]
    return arr[:, 1:4]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("system", choices=sorted(SYSTEMS))
    ap.add_argument("--nsteps", type=int, default=NVT_STEPS)
    ap.add_argument("--devices", type=int, default=int(os.environ.get("UMA_DEVICES", "1")))
    args = ap.parse_args()

    sysinfo = SYSTEMS[args.system]
    natoms = int(sysinfo["natoms"])
    art = ART_F64
    if not (art / "model_traced.pt").is_file():
        raise FileNotFoundError(art / "model_traced.pt")

    # Multi-GPU (devices>1) needs the MP worker + per-(world,natoms) shards.
    if args.devices > 1:
        worker = os.environ.get("UMA_LIBTORCH_MP_WORKER")
        if not worker:
            cand = ROOT / "src" / "ML-UMA" / "uma-engine" / "build-cpp-mp" / "uma_libtorch_mp_worker"
            if cand.is_file():
                worker = str(cand)
                os.environ["UMA_LIBTORCH_MP_WORKER"] = worker
        if not worker or not Path(worker).is_file():
            raise FileNotFoundError("UMA_LIBTORCH_MP_WORKER missing for devices>1")
        os.environ["UMA_MP_NATOMS"] = str(natoms)
        os.environ["UMA_STRUCTURE_NATOMS"] = str(natoms)
        for r in range(args.devices):
            shard = art / f"model_mp_w{args.devices}_n{natoms}_r{r}.pt"
            if not shard.is_file():
                raise FileNotFoundError(f"missing MP shard {shard} (run p0_export_mp)")

    out = results_dir()
    tag = f"{args.system}_d{args.devices}"
    work = out / f"{tag}_uma_work"
    work.mkdir(parents=True, exist_ok=True)
    data = work / Path(sysinfo["data"]).name
    shutil.copy2(sysinfo["data"], data)
    dump = work / "sp_forces.dump"
    log = work / "log.uma"
    elems = " ".join(sysinfo["elements"])

    inp = work / "in.p0"
    inp.write_text(f"""units metal
atom_style atomic
boundary p p p
newton off
read_data {data.name}
pair_style uma precision double devices {args.devices}
pair_coeff * * {art} {elems}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {0.001}
thermo 1
thermo_style custom step temp pe ke etotal
thermo_modify norm no format float %.17g

# (1) single point E+F, full precision dump
dump 1 all custom 1 {dump.name} id type x y z fx fy fz
dump_modify 1 sort id format float %.17g
run 0
undump 1
print "P0_SP_PE = $(pe:%.17g)"

# (2) timed NVT 300 K
velocity all create {NVT_TEMP_K} 12345 mom yes rot yes dist gaussian
fix 1 all nvt temp {NVT_TEMP_K} {NVT_TEMP_K} $(100.0*dt)
run {args.nsteps}
print "P0_NVT_PE = $(pe:%.17g)  T = $(temp:%.17g)"
""")

    lmp = find_lmp()
    cmd = [str(lmp), "-in", inp.name, "-log", log.name]
    (work / "cmd.txt").write_text(" ".join(cmd) + "\n")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    (work / "stdout.txt").write_text(r.stdout)
    (work / "stderr.txt").write_text(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"lmp failed rc={r.returncode}\n{r.stderr[-3000:]}")

    text = log.read_text() + "\n" + r.stdout
    blocks = parse_loop_blocks(text)
    nvt_block = blocks[-1] if blocks else {}
    f = parse_dump_forces(dump)
    e = next((float(l.split("=")[1]) for l in text.splitlines() if l.startswith("P0_SP_PE")), None)
    if e is None:
        raise RuntimeError("P0_SP_PE not found in log")

    nvt_ms = nvt_block.get("pair_ms_per_step") or nvt_block.get("loop_ms_per_step")
    rec = {
        "path": "lammps_uma_precision_double",
        "system": args.system,
        "natoms": int(f.shape[0]),
        "dtype": "float64",
        "devices": args.devices,
        "lmp": str(lmp),
        "energy_eV": e,
        "nvt_steps": args.nsteps,
        "nvt_ms_per_step": nvt_ms,
        "nvt_pair_ms_per_step": nvt_block.get("pair_ms_per_step"),
        "nvt_loop_ms_per_step": nvt_block.get("loop_ms_per_step"),
        "nvt_block": nvt_block,
        "lmp_wall_s": wall,
        "force_abs_max": float(np.abs(f).max()),
        "force_net": [float(x) for x in f.sum(axis=0)],
    }
    (out / f"{tag}_uma.json").write_text(json.dumps(rec, indent=2) + "\n")
    np.savez(out / f"{tag}_uma.npz", forces=f, energy_eV=np.array(e, dtype=np.float64))
    print(json.dumps(rec, indent=2))
    print(f"wrote {out / (tag + '_uma.npz')} forces{f.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
