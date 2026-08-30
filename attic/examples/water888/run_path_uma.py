#!/usr/bin/env python3
"""uma/kk — first-frame E+F parity + post-warmup NVT Pair timing (default 100 steps)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time

from pathlib import Path

import numpy as np

from path_common import (
    ART_F64,
    DEFAULT_DATA,
    find_uma_lmp,
    out_dir,
    parse_lammps_dump_frames,
    parse_lammps_run_blocks,
    setup_ld_path,
    write_timing,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--nsteps", type=int, default=int(os.environ.get("NSTEPS", "100")))
    ap.add_argument("--ngpus", type=int, default=int(os.environ.get("NGPUS", "1")))
    ap.add_argument(
        "--uma-devices",
        type=int,
        default=int(os.environ.get("UMA_DEVICES", os.environ.get("NGPUS", "1"))),
    )
    args = ap.parse_args()
    nsteps = max(1, args.nsteps)
    ngpus = max(1, args.ngpus)
    uma_devices = max(1, args.uma_devices)
    out = out_dir("uma")
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    art = ART_F64
    if not (art / "model_traced.pt").is_file():
        raise FileNotFoundError(art / "model_traced.pt")

    # W8nk product default: plain pair_style uma (no Kokkos). UMA_USE_KOKKOS=1 → uma/kk A/B.
    use_kk = os.environ.get("UMA_USE_KOKKOS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    pair_style = (
        f"pair_style uma/kk precision double devices {uma_devices}"
        if use_kk
        else f"pair_style uma precision double devices {uma_devices}"
    )

    data = work / "water_nvt_300K_atomic_metal.data"
    shutil.copy2(args.data, data)
    dump_sp = work / "forces_first.dump"
    log = work / "log.uma"

    # First-frame dump once (parity); NVT has no force dumps.
    inp = work / "in.uma_fair"
    inp.write_text(
        f"""units metal
atom_style atomic
boundary p p p
newton off
read_data {data.name}
{pair_style}
pair_coeff * * {art} O H
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep 0.001
thermo 10
thermo_style custom step temp pe ke etotal
thermo_modify norm no

# (1) warmup + first-frame E+F dump
dump 1 all custom 1 {dump_sp.name} id type x y z fx fy fz
dump_modify 1 sort id format float %.17g
run 0
undump 1
print "FIRST_PE = $(pe:%.17g)"

# (2) timed SP proxy: 1 NVE step → Pair ms (advances; NVT timing still valid)
fix _sp all nve
run 1
unfix _sp

# (3) timed NVT (no dumps)
fix 1 all nvt temp 300.0 300.0 0.1
run {nsteps}
print "Done. Final T = $(temp) K, PE = $(pe) eV"
"""
    )

    lmp = find_uma_lmp()
    env = setup_ld_path(natoms=648, uma_devices=uma_devices)
    env["UMA_DEVICES"] = str(uma_devices)
    if uma_devices > 1:
        worker = env.get("UMA_LIBTORCH_MP_WORKER")
        if not worker or not Path(worker).is_file():
            raise FileNotFoundError(
                "UMA_LIBTORCH_MP_WORKER missing for multi-GPU; build build-cpp-mp"
            )
        shard = (
            ART_F64
            / f"model_mp_w{uma_devices}_n648_r0.pt"
        )
        if not shard.is_file():
            raise FileNotFoundError(
                f"missing {shard} — run export_mp_water888.slurm first"
            )
    cmd = [str(lmp)]
    if use_kk:
        cmd.extend(["-k", "on", "g", str(ngpus), "-sf", "kk"])
    cmd.extend(
        [
            "-var",
            "UMA_DEVICES",
            str(uma_devices),
            "-in",
            inp.name,
            "-log",
            log.name,
        ]
    )
    (work / "cmd.txt").write_text(" ".join(cmd) + "\n")

    t_wall0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True)
    lmp_wall_s = time.perf_counter() - t_wall0
    (work / "stdout.txt").write_text(r.stdout)
    (work / "stderr.txt").write_text(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"uma lmp failed rc={r.returncode}\n{r.stderr[-3000:]}")

    log_text = log.read_text()
    blocks = parse_lammps_run_blocks(log_text)
    if len(blocks) < 3:
        raise RuntimeError(f"expected ≥3 Loop blocks, got {len(blocks)}: {blocks}")
    sp_block = blocks[1]
    nvt_block = blocks[2]

    f_frames, _ = parse_lammps_dump_frames(dump_sp)
    f = f_frames[0]
    e = None
    for line in log_text.splitlines():
        if line.startswith("FIRST_PE"):
            e = float(line.split("=")[1].strip())
    if e is None:
        raise RuntimeError("FIRST_PE not found")

    sp_ms = sp_block.get("pair_ms_per_step") or sp_block.get("loop_ms_per_step")
    nvt_pair_ms = nvt_block.get("pair_ms_per_step")
    nvt_loop_ms = nvt_block.get("loop_ms_per_step")
    nvt_ms = nvt_pair_ms if nvt_pair_ms is not None else nvt_loop_ms

    write_timing(
        out,
        {
            "path": "uma_kk_precision_double",
            "key": "uma",
            "jobid": os.environ.get("SLURM_JOB_ID"),
            "natoms": int(f.shape[0]),
            "nsteps": nsteps,
            "temperature_K": 300.0,
            "dtype": "float64",
            "ngpus": ngpus,
            "uma_devices": uma_devices,
            "lmp": str(lmp),
            "energy_eV": e,
            "sp_ms": sp_ms,
            "nvt_ms_per_step": nvt_ms,
            "nvt_ms_total": (nvt_ms * nsteps) if nvt_ms is not None else None,
            "nvt_pair_ms_per_step": nvt_pair_ms,
            "nvt_loop_ms_per_step": nvt_loop_ms,
            "sp_block": sp_block,
            "nvt_block": nvt_block,
            "all_loop_blocks": blocks,
            "lmp_wall_s": lmp_wall_s,
            "timing_source_sp": "lammps_post_warmup_nve1_pair",
            "timing_source_nvt": "lammps_post_warmup_nvt_pair",
            "cold_start_excluded": True,
            "warmup": True,
            "parity_frame": "first",
            "nvt_frame_dumps": False,
            "note": (
                "First-frame E+F from run 0 dump; NVT Pair ms/step for speed. "
                "lmp_wall_s includes cold start (not used for compare)."
            ),
        },
        forces=f,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
