#!/usr/bin/env python3
"""M3 multi-node LAMMPS-UMA: single point (E + per-atom F) + NVT-300K timing.

Scheme A replicated: run under mpiexec with N MPI ranks, each `pair_style uma
precision double devices 1` (one GPU/rank). Each rank assembles the tag-ordered
global system, runs the full traced model on its GPU, keeps its owned forces;
rank 0 contributes energy.

Usage:
  --build-input : emit the LAMMPS deck (SP dump + timed NVT) into --work
  --parse       : parse E (full precision), per-atom F, NVT ms/step, rank count
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "polaris_p0"))
from p0_common import ART_F64, SYSTEMS  # noqa: E402

NVT_TEMP_K = 300.0


def build_input(system: str, out: Path, nsteps: int, art: Path) -> Path:
    info = SYSTEMS[system]
    elems = " ".join(info["elements"])
    data = Path(info["data"])
    dump = out / "m3_forces.dump"
    deck = out / "in.m3"
    deck.write_text(f"""units metal
atom_style atomic
boundary p p p
newton off
read_data {data}
# One rank per GPU (devices 1); multi-node Scheme A triggers on nprocs>1.
pair_style uma precision double devices 1
pair_coeff * * {art} {elems}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep 0.001
thermo 1
thermo_style custom step temp pe
thermo_modify norm no format float %.17g

# (1) single point E + per-atom F (full precision)
dump 1 all custom 1 {dump} id type x y z fx fy fz
dump_modify 1 sort id format float %.17g
run 0
undump 1
print "M3_SP_PE = $(pe:%.17g)"

# (2) timed NVT 300 K (no force dumps)
velocity all create {NVT_TEMP_K} 12345 mom yes rot yes dist gaussian
fix 1 all nvt temp {NVT_TEMP_K} {NVT_TEMP_K} $(100.0*dt)
run {nsteps}
print "M3_NVT_PE = $(pe:%.17g)  T = $(temp:%.17g)"
""")
    return deck


def _loop_blocks(text: str):
    loop_re = re.compile(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps")
    pair_re = re.compile(r"^Pair\s+\|\s+([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)", re.M)
    out = []
    for m in loop_re.finditer(text):
        nxt = loop_re.search(text, m.end())
        chunk = text[m.end(): nxt.start() if nxt else len(text)]
        pm = pair_re.search(chunk)
        n = int(m.group(3)); loop_s = float(m.group(1))
        pair_s = float(pm.group(1)) if pm else None
        out.append({"nsteps": n, "loop_s": loop_s, "pair_s": pair_s,
                    "loop_ms_per_step": loop_s / n * 1e3 if n else None,
                    "pair_ms_per_step": pair_s / n * 1e3 if (pair_s and n) else None})
    return out


def parse_results(work: Path, out_json: Path, system: str, nsteps: int) -> dict:
    info = SYSTEMS[system]
    txt = ""
    for p in (work / "out.m3", work / "log.m3"):
        if p.is_file():
            txt += p.read_text(errors="ignore") + "\n"
    e_sp = None
    for m in re.finditer(r"M3_SP_PE = (\S+)", txt):
        e_sp = float(m.group(1))
    grids = [int(a) * int(b) * int(c)
             for a, b, c in re.findall(r"(\d+) by (\d+) by (\d+) MPI processor grid", txt)]
    blocks = _loop_blocks(txt)
    nvt_block = blocks[-1] if blocks else {}
    nvt_ms = nvt_block.get("pair_ms_per_step") or nvt_block.get("loop_ms_per_step")

    dump = work / "m3_forces.dump"
    f = None
    if dump.is_file():
        lines = dump.read_text().splitlines()
        s = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS"))
        cols = lines[s].split()[2:]
        i_id, i_fx = cols.index("id"), cols.index("fx")
        rows = []
        for l in lines[s + 1:]:
            if l.startswith("ITEM:"):
                break
            p = l.split()
            if len(p) >= len(cols):
                rows.append((float(p[i_id]), float(p[i_fx]), float(p[i_fx + 1]), float(p[i_fx + 2])))
        arr = np.array(rows)
        arr = arr[np.argsort(arr[:, 0])]
        f = arr[:, 1:4]

    rec = {
        "system": system,
        "natoms": info["natoms"],
        "energy_eV": e_sp,
        "ranks": max(grids) if grids else 0,
        "nvt_steps": nsteps,
        "nvt_ms_per_step": nvt_ms,
        "nvt_pair_ms_per_step": nvt_block.get("pair_ms_per_step"),
        "nvt_loop_ms_per_step": nvt_block.get("loop_ms_per_step"),
        "force_abs_max": (float(np.abs(f).max()) if f is not None else None),
        "force_net": ([float(x) for x in f.sum(0)] if f is not None else None),
    }
    out_json.write_text(json.dumps(rec, indent=2) + "\n")
    if f is not None:
        np.savez(out_json.with_suffix(".npz"), forces=f,
                 energy_eV=np.array(e_sp if e_sp is not None else np.nan))
    print(json.dumps(rec, indent=2))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("system", choices=sorted(SYSTEMS))
    ap.add_argument("--work", required=True)
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--build-input", action="store_true")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--tag", default="mn")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if args.build_input:
        print(str(build_input(args.system, work, args.nsteps, ART_F64)))
    if args.parse:
        parse_results(work, work / f"m3_{args.tag}.json", args.system, args.nsteps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
