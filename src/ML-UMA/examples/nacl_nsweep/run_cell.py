#!/usr/bin/env python3
"""Run one N-sweep cell: LAMMPS NVT @300 K on N GPUs, classify OOM, record JSON.

Functional gate  : LAMMPS completes the NVT block and reports finite T/PE.
Capacity gate    : no CUDA OOM anywhere (LAMMPS stdout or any worker log).
Parity gate      : handled separately by parity_cell.py (single-GPU oracle).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EX = Path(__file__).resolve().parent
ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")
sys.path.insert(0, str(ROOT / "src/ML-UMA/examples/multi_gpu_nacl6"))
from path_common import (  # noqa: E402
    parse_lammps_dump_frames,
    parse_lammps_run_blocks,
)

OOM_PAT = re.compile(
    r"CUDA out of memory|out of memory|OutOfMemoryError|cudaErrorMemoryAllocation"
    r"|CUBLAS_STATUS_ALLOC_FAILED|NCCL.*unhandled cuda error|c10::OutOfMemoryError",
    re.I,
)


class VramSampler:
    """Poll nvidia-smi during the run to capture peak VRAM per GPU.

    The ceiling this sweep measures is VRAM, so peak used/total must be
    recorded directly — inferring it from whether the job crashed only tells
    us pass/fail, not headroom, and gives nothing to extrapolate from.
    """

    def __init__(self, period_s: float = 0.5):
        self.period = period_s
        self._proc = None
        self.path = None

    def start(self, path: Path):
        self.path = path
        script = (
            "while true; do nvidia-smi --query-gpu=index,memory.used,memory.total "
            "--format=csv,noheader,nounits; sleep %s; done" % self.period
        )
        self._proc = subprocess.Popen(
            ["bash", "-c", script],
            stdout=open(path, "w"), stderr=subprocess.DEVNULL, text=True,
        )

    def stop(self) -> dict:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        peak: dict[int, float] = {}
        total: dict[int, float] = {}
        if self.path and Path(self.path).is_file():
            for line in Path(self.path).read_text(errors="ignore").splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 3:
                    continue
                try:
                    idx, used, tot = int(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                peak[idx] = max(peak.get(idx, 0.0), used)
                total[idx] = tot
        if not peak:
            return {}
        return {
            "vram_peak_MiB_per_gpu": {str(k): peak[k] for k in sorted(peak)},
            "vram_total_MiB_per_gpu": {str(k): total[k] for k in sorted(total)},
            "vram_peak_MiB_max": max(peak.values()),
            "vram_total_MiB": max(total.values()) if total else None,
            "vram_headroom_MiB": (max(total.values()) - max(peak.values()))
            if total else None,
            "vram_util_frac": round(max(peak.values()) / max(total.values()), 4)
            if total else None,
        }


def classify(stdout: str, stderr: str, mp_log_dir: Path | None) -> tuple[bool, str]:
    blob = stdout + "\n" + stderr
    if mp_log_dir and mp_log_dir.is_dir():
        for f in sorted(mp_log_dir.glob("*.log")):
            try:
                blob += "\n" + f.read_text(errors="ignore")
            except OSError:
                pass
    m = OOM_PAT.search(blob)
    if m:
        line = next((ln.strip() for ln in blob.splitlines()
                     if OOM_PAT.search(ln)), m.group(0))
        return True, line[:300]
    return False, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--uma-devices", type=int, default=4)
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--artifact", required=True)
    a = ap.parse_args()

    n, dev = a.n, a.uma_devices
    natoms = 8 * n**3
    job = os.environ.get("SLURM_JOB_ID", "manual")
    out = EX / "results" / f"n{n}_{job}"
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    art = Path(a.artifact)
    mp_log_dir = Path(os.environ.get("UMA_MP_LOG_DIR", out / "mp_logs"))

    data = work / Path(a.data).name
    shutil.copy2(a.data, data)
    dump = work / "forces_first.dump"
    log = work / "log.uma"
    inp = work / "in.nsweep"
    inp.write_text(f"""units metal
atom_style atomic
boundary p p p
newton off
read_data {data.name}
pair_style uma precision double devices {dev}
pair_coeff * * {art} Na Cl
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep 0.001
thermo 1
thermo_style custom step temp pe ke etotal
thermo_modify norm no

dump 1 all custom 1 {dump.name} id type x y z fx fy fz
dump_modify 1 sort id
run 0
undump 1
print "FIRST_PE = $(pe)"

fix _sp all nve
run 1
unfix _sp

fix 1 all nvt temp 300.0 300.0 0.1
run {a.nsteps}
print "FINAL_T = $(temp)"
print "FINAL_PE = $(pe)"
""")

    lmp = os.environ.get("LMP_UMA", str(ROOT / "build-uma/lmp"))
    cmd = [lmp, "-in", inp.name, "-log", log.name]
    sampler = VramSampler()
    sampler.start(work / "vram_samples.csv")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=work, env=os.environ.copy(),
                       capture_output=True, text=True)
    wall = time.perf_counter() - t0
    vram = sampler.stop()
    (work / "stdout.txt").write_text(r.stdout)
    (work / "stderr.txt").write_text(r.stderr)

    oom, oom_line = classify(r.stdout, r.stderr, mp_log_dir)
    rec: dict = {
        "n": n, "natoms": natoms, "cell_A": round(n * 5.64, 3),
        "uma_devices": dev, "nsteps": a.nsteps, "job": job,
        "returncode": r.returncode, "wall_s": round(wall, 1),
        "oom": oom, "oom_line": oom_line,
        "artifact": str(art), "dtype": "float64",
        "pair_style": f"pair_style uma precision double devices {dev}",
    }
    rec.update(vram)

    if r.returncode == 0 and log.is_file():
        text = log.read_text()
        blocks = parse_lammps_run_blocks(text)
        rec["n_loop_blocks"] = len(blocks)
        if len(blocks) >= 3:
            rec["sp_ms"] = blocks[1].get("pair_ms_per_step")
            rec["nvt_pair_ms_per_step"] = blocks[2].get("pair_ms_per_step")
            rec["nvt_loop_ms_per_step"] = blocks[2].get("loop_ms_per_step")
        # Require a numeric value: the log also contains the echoed input line
        # (`print "FIRST_PE = $(pe)"`), and \S+ matched that literal, raising
        # ValueError on '$(pe)"' and masking the real failure.
        num = r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
        for key, pat in (("energy_eV", rf"FIRST_PE = {num}"),
                         ("final_T_K", rf"FINAL_T = {num}"),
                         ("final_PE_eV", rf"FINAL_PE = {num}")):
            hits = re.findall(pat, text)
            if hits:
                rec[key] = float(hits[-1])
        if dump.is_file():
            frames, _ = parse_lammps_dump_frames(dump)
            f = frames[0]
            np.savez(out / "forces.npz", forces=np.asarray(f, np.float64),
                     energy_eV=np.array(rec.get("energy_eV", np.nan)))
            rec["force_absmax"] = float(np.abs(f).max())
            # Net force must vanish for a periodic cell; a large residual means
            # the multi-GPU force reduction dropped or double-counted a shard.
            rec["force_sum_abs"] = float(np.abs(f.sum(axis=0)).max())
        T = rec.get("final_T_K")
        rec["functional"] = bool(
            rec.get("nvt_pair_ms_per_step")
            and T is not None and np.isfinite(T) and 100.0 < T < 900.0
            and rec.get("force_sum_abs", 1.0) < 1e-6
        )
        rec["status"] = "OK" if rec["functional"] else "RAN_BUT_SUSPECT"
    else:
        rec["functional"] = False
        rec["status"] = "OOM" if oom else "FAIL"
        rec["stderr_tail"] = r.stderr[-1500:]

    (out / "cell.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in (
        "n", "natoms", "status", "oom", "functional",
        "nvt_pair_ms_per_step", "final_T_K", "force_sum_abs",
        "vram_peak_MiB_max", "vram_total_MiB", "vram_util_frac") if k in rec},
        indent=2))
    print(f"NSWEEP_RECORD {out / 'cell.json'}")
    # Exit 0 even on OOM: OOM is a measurement, not a job failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
