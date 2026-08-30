#!/usr/bin/env python3
"""Gate one V7 cell: per-atom E/F parity + SP and NVT timing vs W8nk.

Per-atom forces are the gate. |sum F| ~ 0 only tests translational invariance
and is bit-identical under sign inversion, so it is recorded but never decides.

  python v7_gate.py --wave w18 --sys nacl6 --ngpu 4 --job 12345 --write
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")
EXAMPLES = ROOT / "src/ML-UMA/examples"
CAMP = EXAMPLES / "multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign"
EX = {"nacl6": EXAMPLES / "multi_gpu_nacl6", "water888": EXAMPLES / "water888"}
ORACLE_F = {
    "nacl6": CAMP / "oracle_ase_umas_fast_merge.npz",
    "water888": CAMP / "oracle_ase_water_merge.npz",
}
ORACLE_E = {"nacl6": -5830.9237413382, "water888": -3143.3893774722696}
# W8nk product bars (nvt_pair_ms_per_step) — the thing V7 must beat.
W8NK_NVT = {("nacl6", 2): 161.94, ("nacl6", 4): 92.097,
            ("water888", 2): 164.82, ("water888", 4): 95.744}
W8NK_NVT_1 = {"nacl6": 296.48}
ASE_UFAST = {("nacl6", 2): 191.6, ("nacl6", 4): 164.5,
             ("water888", 2): 165.5, ("water888", 4): 94.5}
DE_TOL, DF_TOL = 1e-6, 1e-5


def find_result(sysname: str, job: str) -> Path | None:
    hits = sorted((EX[sysname] / "results").glob(f"uma_ngpu*_{job}"))
    return hits[0] if hits else None


def tick_stats(mp_log_dir: Path) -> dict:
    """Median of the W18 per-rank counters, warm ticks only."""
    keys = ("ms_fwd", "ms_bwd", "ms_force_ar", "ms_h2d", "ms_wshard",
            "ms_pad", "ms_prep", "ms_bar_pre_bwd", "ms_post", "ms_accounted")
    per: dict[str, list[float]] = {k: [] for k in keys}
    if not mp_log_dir.is_dir():
        return {}
    for f in sorted(mp_log_dir.glob("worker_r*.log")):
        txt = f.read_text(errors="ignore")
        lines = [l for l in txt.splitlines() if l.startswith("PERF_TICK")]
        for l in lines[1:]:               # drop first (cold) tick
            for k in keys:
                m = re.search(rf"\b{k}=([0-9.eE+-]+)", l)
                if m:
                    try:
                        per[k].append(float(m.group(1)))
                    except ValueError:
                        pass
    return {f"tick_{k}": round(float(np.median(v)), 4)
            for k, v in per.items() if v}


def parent_stats(mp_log_dir: Path) -> dict:
    p = mp_log_dir / "parent.log"
    if not p.is_file():
        return {}
    keys = ("ms_vesin", "ms_wait_workers", "ms_total", "ms_shard", "ms_pub")
    per: dict[str, list[float]] = {k: [] for k in keys}
    lines = [l for l in p.read_text(errors="ignore").splitlines()
             if l.startswith("PERF_PARENT")]
    for l in lines[1:]:
        for k in keys:
            m = re.search(rf"\b{k}=([0-9.eE+-]+)", l)
            if m:
                per[k].append(float(m.group(1)))
    return {f"parent_{k}": round(float(np.median(v)), 4)
            for k, v in per.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True)
    ap.add_argument("--sys", dest="sysname", required=True, choices=list(EX))
    ap.add_argument("--ngpu", type=int, required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rec: dict = {"ver": a.wave, "sys": a.sysname, "ngpu": a.ngpu, "job": a.job,
                 "build": "v7"}
    res = find_result(a.sysname, a.job)
    if res is None or not (res / "timing.json").is_file():
        rec["status"] = "NO_RESULTS"
        print(json.dumps(rec, indent=2))
        return 3

    t = json.loads((res / "timing.json").read_text())
    rec["natoms"] = t.get("natoms")
    rec["sp_ms"] = t.get("sp_ms")
    rec["nvt_pair_ms_per_step"] = t.get("nvt_pair_ms_per_step")
    rec["nsteps"] = t.get("nsteps")
    rec["energy_eV"] = t.get("energy_eV")
    rec["timing_source_internal"] = "lammps Pair/Loop timers (cross-check)"

    # Authoritative timing: measured in the SLURM script around the bash
    # command, outside LAMMPS and outside any in-code counter.
    slurm_log = CAMP / "logs" / f"v7-{a.wave}-{a.sysname}-n{a.ngpu}-{a.job}.out"
    if slurm_log.is_file():
        txt = slurm_log.read_text(errors="ignore")
        m = re.search(r"SHELL_TIMING .*", txt)
        if m:
            for k, cast in (("pass_a_s", float), ("pass_b_s", float),
                            ("nsteps_a", int), ("nsteps_b", int), ("dn", int),
                            ("nvt_ms_per_step_shell", float)):
                mm = re.search(rf"\b{k}=([0-9.eE+-]+)", m.group(0))
                if mm:
                    rec[f"shell_{k}"] = cast(mm.group(1))
            rec["timing_source"] = "shell (SLURM script, external to LAMMPS)"

    # ---- per-atom E/F parity ----
    oe = ORACLE_E.get(a.sysname)
    if oe is not None and rec["energy_eV"] is not None:
        rec["dE_vs_merge_ase"] = abs(rec["energy_eV"] - oe)
    fz, onpz = res / "forces.npz", ORACLE_F.get(a.sysname)
    if fz.is_file() and onpz and onpz.is_file():
        f = np.load(fz)["forces"]
        fr = np.load(onpz)["forces"]
        if f.shape == fr.shape:
            mag = np.linalg.norm(f - fr, axis=1)
            rec.update({
                "force_max_per_atom": float(mag.max()),
                "force_mean_per_atom": float(mag.mean()),
                "force_rms_per_atom": float(np.sqrt((mag**2).mean())),
                "n_atoms_over_tol": int((mag > DF_TOL).sum()),
                "n_atoms": int(f.shape[0]),
                "worst_atoms": [
                    {"index": int(i), "abs_err": float(mag[i]),
                     "F": [float(x) for x in f[i]],
                     "F_oracle": [float(x) for x in fr[i]]}
                    for i in np.argsort(mag)[::-1][:5]],
            })
            # |sum F| recorded for completeness; NOT a gate.
            rec["force_sum_abs"] = float(np.abs(f.sum(axis=0)).max())
        else:
            rec["force_shape_mismatch"] = [list(f.shape), list(fr.shape)]

    ef = bool(rec.get("dE_vs_merge_ase", 1) <= DE_TOL
              and rec.get("force_max_per_atom", 1) <= DF_TOL)
    rec["ef_pass"] = ef

    # ---- speed vs W8nk ----
    key = (a.sysname, a.ngpu)
    base = W8NK_NVT.get(key) or (W8NK_NVT_1.get(a.sysname)
                                 if a.ngpu == 1 else None)
    ms = rec["nvt_pair_ms_per_step"]
    if base and ms:
        rec["baseline_w8nk_ms"] = base
        rec["delta_vs_w8nk_ms"] = round(ms - base, 3)
        rec["speed_win_ge_1ms"] = bool(ms - base <= -1.0)
    ase = ASE_UFAST.get(key)
    if ase and ms:
        rec["ase_ufast"] = ase
        rec["le_ase_nvt"] = ms <= ase

    # ---- W18 counters ----
    mp = CAMP / f"matrix/v7_{a.wave}_{a.sysname}_ngpu{a.ngpu}/mp_logs"
    rec.update(tick_stats(mp))
    rec.update(parent_stats(mp))
    acc, wait = rec.get("tick_ms_accounted"), rec.get("parent_ms_wait_workers")
    if acc and wait:
        rec["accounting_residual_ms"] = round(wait - acc, 3)
        rec["accounting_closed"] = bool(abs(wait - acc) <= 2.0)

    if not ef:
        rec["status"] = "EF_FAIL"
        rc = 2
    elif rec.get("speed_win_ge_1ms"):
        rec["status"] = "PASS_PROMOTE_CANDIDATE"
        rc = 0
    else:
        rec["status"] = "PASS_NO_PROMOTE"
        rc = 0
    rec["promote"] = bool(ef and rec.get("speed_win_ge_1ms"))

    if a.write:
        out = CAMP / f"gate_v7_{a.wave}_{a.sysname}_ngpu{a.ngpu}.json"
        out.write_text(json.dumps(rec, indent=2) + "\n")
        rec["_written"] = str(out)
    print(json.dumps(rec, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
