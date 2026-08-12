#!/usr/bin/env python3
"""M3 same-node graph-parallel LAMMPS-UMA: single point (E + per-atom F) + NVT timing.

Uses `pair_style uma precision double devices N` (1 MPI rank, N GPUs). The engine
GraphParallelRuntime forks N workers, edge-shards the graph so each GPU holds
~1/N of it (memory-sharded -> NaCl 8x8x8 = 4096 fits on 4 GPUs). This is the
4-GPU/1-node ground truth the multi-node run must reproduce.

Requires model_mp_w{N}_n{natoms}_r*.pt shards + uma_libtorch_mp_worker.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "polaris_p0"))
from p0_common import ART_F64, ENGINE, ROOT, SYSTEMS  # noqa: E402

NVT_TEMP_K = 300.0


def build_input(system: str, work: Path, nsteps: int, art: Path, devices: int) -> Path:
    info = SYSTEMS[system]
    elems = " ".join(info["elements"])
    data = Path(info["data"])
    dump = work / "gp_forces.dump"
    deck = work / "in.gp"
    deck.write_text(f"""units metal
atom_style atomic
boundary p p p
newton off
read_data {data}
pair_style uma precision double devices {devices}
pair_coeff * * {art} {elems}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep 0.001
thermo 1
thermo_style custom step temp pe
thermo_modify norm no format float %.17g

dump 1 all custom 1 {dump} id type x y z fx fy fz
dump_modify 1 sort id format float %.17g
run 0
undump 1
print "GP_SP_PE = $(pe:%.17g)"

velocity all create {NVT_TEMP_K} 12345 mom yes rot yes dist gaussian
fix 1 all nvt temp {NVT_TEMP_K} {NVT_TEMP_K} $(100.0*dt)
run {nsteps}
print "GP_NVT_PE = $(pe:%.17g)  T = $(temp:%.17g)"
""")
    return deck


def loop_blocks(text: str):
    loop_re = re.compile(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps")
    pair_re = re.compile(r"^Pair\s+\|\s+([0-9.eE+-]+)\s+\|", re.M)
    out = []
    for m in loop_re.finditer(text):
        nxt = loop_re.search(text, m.end())
        chunk = text[m.end(): nxt.start() if nxt else len(text)]
        pm = pair_re.search(chunk)
        n = int(m.group(3)); loop_s = float(m.group(1))
        pair_s = float(pm.group(1)) if pm else None
        out.append({"nsteps": n, "loop_ms_per_step": loop_s / n * 1e3 if n else None,
                    "pair_ms_per_step": pair_s / n * 1e3 if (pair_s and n) else None})
    return out


def parse_forces(dump: Path):
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
    return arr[np.argsort(arr[:, 0])][:, 1:4]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("system", choices=sorted(SYSTEMS))
    ap.add_argument("--work", required=True)
    ap.add_argument("--devices", type=int, required=True)
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--tag", default="gp")
    args = ap.parse_args()

    info = SYSTEMS[args.system]
    natoms = info["natoms"]
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    art = ART_F64

    # shard preflight
    worker = os.environ.get("UMA_LIBTORCH_MP_WORKER", "")
    if not worker or not Path(worker).is_file():
        raise FileNotFoundError(f"UMA_LIBTORCH_MP_WORKER missing: {worker!r}")
    for r in range(args.devices):
        shard = art / f"model_mp_w{args.devices}_n{natoms}_r{r}.pt"
        if not shard.is_file():
            raise FileNotFoundError(f"missing shard {shard}")

    deck = build_input(args.system, work, args.nsteps, art, args.devices)
    lmp = Path(os.environ.get("LMP_UMA", ROOT / "build-uma" / "lmp"))
    env = os.environ.copy()
    env["UMA_MP_NATOMS"] = str(natoms)
    env["UMA_STRUCTURE_NATOMS"] = str(natoms)
    env["UMA_FORBID_RAY_GP"] = "1"

    t0 = time.perf_counter()
    r = subprocess.run([str(lmp), "-in", deck.name, "-log", "log.gp"],
                       cwd=work, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    (work / "stdout.txt").write_text(r.stdout)
    (work / "stderr.txt").write_text(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"lmp failed rc={r.returncode}\n{r.stderr[-3000:]}")

    text = (work / "log.gp").read_text() + "\n" + r.stdout
    # Require a real number: the deck echo line prints the literal "$(pe:%.17g)".
    e = None
    for m in re.finditer(r"GP_SP_PE = (-?\d[\d.eE+-]*)", text):
        e = float(m.group(1))
    blocks = loop_blocks(text)
    nvt = blocks[-1] if blocks else {}
    nvt_ms = nvt.get("pair_ms_per_step") or nvt.get("loop_ms_per_step")
    f = parse_forces(work / "gp_forces.dump")

    rec = {"system": args.system, "natoms": natoms, "devices": args.devices,
           "energy_eV": e, "nvt_steps": args.nsteps, "nvt_ms_per_step": nvt_ms,
           "force_abs_max": float(np.abs(f).max()),
           "force_net": [float(x) for x in f.sum(0)], "wall_s": wall}
    out_json = work.parent / f"m3_{args.tag}.json"
    out_json.write_text(json.dumps(rec, indent=2) + "\n")
    np.savez(out_json.with_suffix(".npz"), forces=f,
             energy_eV=np.array(e if e is not None else np.nan))
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
