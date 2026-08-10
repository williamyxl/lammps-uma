#!/usr/bin/env python3
"""Gate one W17 CUDA-graph cell: graph path tag + E/F vs ASE merge oracle + speed.

Usage:
  python w17_gate.py --job 21015029 --sys nacl6 --ngpu 2 --ver w17c [--write]

Emits gate_v6_<ver>_<sys>_ngpu<N>.json in the campaign dir when --write is given.
Exit code 0 = PASS (graph replay + E/F), 1 = graph fail, 2 = E/F fail, 3 = missing data.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")
CAMP = ROOT / "src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign"
EX = {
    "nacl6": ROOT / "src/ML-UMA/examples/multi_gpu_nacl6",
    "water888": ROOT / "src/ML-UMA/examples/water888",
}
ORACLE_NPZ = {
    "nacl6": CAMP / "oracle_ase_umas_fast_merge.npz",
    "water888": CAMP / "oracle_ase_water_merge.npz",
}
ORACLE_JSON = {
    "nacl6": CAMP / "oracle_ase_merge_mole.json",
    "water888": CAMP / "oracle_ase_water_merge_mole.json",
}
# W8nk product bars (nvt_pair_ms_per_step) and ASE ufast floors.
W8NK = {("nacl6", 2): 161.94, ("nacl6", 4): 92.097,
        ("water888", 2): 164.82, ("water888", 4): 95.744}
ASE_UFAST = {("nacl6", 2): 191.6, ("nacl6", 4): 164.5,
             ("water888", 2): 165.5, ("water888", 4): 94.5}
DE_TOL = 1e-6
DF_TOL = 1e-5

PATHS = ("graph_replay", "graph_capture", "graph_warmup", "graph_fail_eager", "eager")


def find_log(sysname: str, job: str) -> Path | None:
    p = EX[sysname] / "logs" / f"path_uma-{job}.out"
    return p if p.is_file() else None


def find_results(sysname: str, job: str) -> Path | None:
    base = EX[sysname] / "results"
    hits = sorted(base.glob(f"uma_ngpu*_{job}"))
    return hits[0] if hits else None


def job_mp_log_dir(job: str) -> Path | None:
    """Read UMA_MP_LOG_DIR out of the job's recorded submit environment."""
    import subprocess

    for cmd in (["scontrol", "show", "job", job],
                ["sacct", "-j", job, "-n", "-X", "-o", "Comment%512"]):
        try:
            t = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        except Exception:
            continue
        m = re.search(r"UMA_MP_LOG_DIR=([^\s,]+)", t)
        if m:
            return Path(m.group(1))
    return None


def scan_worker_paths(job: str, sysname: str, ngpu: int, ver: str,
                      mp_log_dir: Path | None = None) -> dict:
    """Collect path= tags from THIS job's worker logs plus the job stdout.

    Scoped to one mp_logs dir: a bare matrix/* glob picks up stale worker logs
    from earlier waves and silently mis-gates the graph path.
    """
    tags: dict[str, int] = {}
    errors: list[str] = []
    d = mp_log_dir or job_mp_log_dir(job)
    if d is None:
        d = CAMP / f"matrix/{sysname}_uma_ufast_{ver.rstrip('abc')}_ngpu{ngpu}/mp_logs"
    cand = sorted(d.glob("worker_r*.log")) if d.is_dir() else []
    log = find_log(sysname, job)
    texts = []
    for f in cand:
        try:
            texts.append((str(f), f.read_text(errors="ignore")))
        except OSError:
            pass
    if log:
        texts.append((str(log), log.read_text(errors="ignore")))
    for _, t in texts:
        for m in re.finditer(r"path=([a-z_]+)", t):
            tags[m.group(1)] = tags.get(m.group(1), 0) + 1
        # Keep the TorchScript frame after the header line: the header alone is
        # generic ("operation failed in the TorchScript interpreter") and hides
        # which op is capture-illegal.
        # Traceback lines start at column 0, so stop at the next worker/PERF
        # record rather than at the first non-indented line.
        stop = r"(?=\numa_libtorch_mp_worker |\nPERF_|\nJOBID=|\Z)"
        for m in re.finditer(r"CAPTURE FAILED: (.+?)" + stop, t, re.S):
            block = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
            head = block[0] if block else ""
            frame = next((ln for ln in block
                          if "<--- HERE" in ln or "RuntimeError" in ln), "")
            culprit = next((block[i - 1] for i, ln in enumerate(block)
                            if "<--- HERE" in ln and i), "")
            e = " | ".join(x for x in (head, culprit, frame) if x)[:400]
            if e and e not in errors:
                errors.append(e)
    return {"path_counts": tags, "capture_errors": errors[:5]}


def gate_ef(sysname: str, res: Path) -> dict:
    out: dict = {}
    fz = res / "forces.npz"
    tj = json.loads((res / "timing.json").read_text())
    out["energy_eV"] = tj.get("energy_eV")
    oj = ORACLE_JSON.get(sysname)
    if oj and oj.is_file():
        od = json.loads(oj.read_text())
        oe = None
        for k in ("umas_fast_merge_E", "ase_merge_E", "uma_merge_E", "energy_eV"):
            if isinstance(od.get(k), (int, float)):
                oe = od[k]
                break
        if oe is None:
            for row in od.get("rows", []):
                if row.get("merge_mole") and row.get("execution_mode") == "umas_fast_pytorch":
                    oe = row.get("energy_eV")
        if oe is not None and out["energy_eV"] is not None:
            out["oracle_E"] = oe
            out["dE_vs_merge_ase"] = abs(out["energy_eV"] - oe)
    onpz = ORACLE_NPZ.get(sysname)
    if fz.is_file() and onpz and onpz.is_file():
        f = np.load(fz)
        o = np.load(onpz)
        fk = next((k for k in ("forces", "f", "arr_0") if k in f), None)
        ok = next((k for k in ("forces", "f", "arr_0") if k in o), None)
        if fk and ok and f[fk].shape == o[ok].shape:
            d = np.abs(f[fk] - o[ok])
            out["force_max_abs"] = float(d.max())
            out["force_mae"] = float(d.mean())
    de = out.get("dE_vs_merge_ase")
    fm = out.get("force_max_abs")
    out["ef_pass"] = bool(
        (de is not None and de <= DE_TOL) and (fm is None or fm <= DF_TOL)
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--sys", dest="sysname", default="nacl6", choices=list(EX))
    ap.add_argument("--ngpu", type=int, default=2)
    ap.add_argument("--ver", default="w17c")
    ap.add_argument("--mp-log-dir", default=None,
                    help="worker mp_logs dir (default: from job env)")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    res = find_results(a.sysname, a.job)
    stamp: dict = {
        "ver": a.ver,
        "sys": a.sysname,
        "ngpu": a.ngpu,
        "job": a.job,
        "art": "uma-s-1p2-omat-f64-fast-cgraph",
        "UMA_CUDA_GRAPH": 1,
        "UMA_EDGE_PAD": 1,
    }
    stamp.update(scan_worker_paths(
        a.job, a.sysname, a.ngpu, a.ver,
        Path(a.mp_log_dir) if a.mp_log_dir else None))
    counts = stamp["path_counts"]
    stamp["graph_path"] = next((p for p in PATHS if counts.get(p)), "unknown")
    stamp["graph_replay_n"] = counts.get("graph_replay", 0)
    stamp["graph_captured"] = counts.get("graph_capture", 0) > 0

    if res is None or not (res / "timing.json").is_file():
        stamp["status"] = "NO_RESULTS"
        print(json.dumps(stamp, indent=2))
        return 3

    tj = json.loads((res / "timing.json").read_text())
    ms = tj.get("nvt_pair_ms_per_step")
    stamp["nvt_pair_ms_per_step"] = ms
    stamp["sp_ms"] = tj.get("sp_ms")
    stamp.update(gate_ef(a.sysname, res))

    key = (a.sysname, a.ngpu)
    base = W8NK.get(key)
    ase = ASE_UFAST.get(key)
    if ms is not None and base is not None:
        stamp["baseline_w8nk_ms"] = base
        stamp["delta_vs_w8nk_ms"] = round(ms - base, 3)
    if ms is not None and ase is not None:
        stamp["ase_ufast"] = ase
        stamp["le_ase_nvt"] = ms <= ase
        stamp["delta_vs_ase_ufast_ms"] = round(ms - ase, 3)

    ef = bool(stamp.get("ef_pass"))
    graph_ok = stamp["graph_captured"] and stamp["graph_path"] != "graph_fail_eager"
    win = stamp.get("delta_vs_w8nk_ms")
    stamp["speed_win_ge_1ms"] = bool(win is not None and win <= -1.0)
    stamp["promote"] = bool(
        graph_ok and ef and (stamp["speed_win_ge_1ms"] or stamp.get("le_ase_nvt"))
    )
    if not graph_ok:
        stamp["status"] = "CAPTURE_FAIL"
        rc = 1
    elif not ef:
        stamp["status"] = "EF_FAIL"
        rc = 2
    else:
        stamp["status"] = "PASS_PROMOTE" if stamp["promote"] else "PASS_NO_PROMOTE"
        rc = 0

    if a.write:
        out = CAMP / f"gate_v6_{a.ver}_{a.sysname}_ngpu{a.ngpu}.json"
        out.write_text(json.dumps(stamp, indent=2) + "\n")
        stamp["_written"] = str(out)
    print(json.dumps(stamp, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
