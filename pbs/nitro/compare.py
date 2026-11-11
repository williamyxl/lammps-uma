#!/usr/bin/env python3
"""Parse LAMMPS thermo logs and MP_PERF lines.

Subcommands
  pe   LOG STEP                 print the PE at STEP with %.16e
  bar  LOG STEP REF TOL         exit 1 unless |PE(STEP) - REF| < TOL
  perf LOG [SKIP]               JSON with median / MAD / min / max of the MP_PERF
                                fields over the lines after the first SKIP (default 5)
  bit  LOG_A LOG_B STEP         print whether PE(STEP) is bitwise equal and the diff
  abba JOBDIR [JOBDIR ...]     >=3 jobs. Each jobdir has slots/*/perf.json and
                               slots/*/label. Verdict over per-job medians.
  abba-test                    synthetic cases: sign clash and wide spread do not win
"""
import json
import re
import statistics
import sys
from pathlib import Path

THERMO = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
FIELDS = (
    "n_edges_shard", "ms_graph", "ms_pre", "ms_vesin", "ms_shardpad",
    "ms_fwd", "ms_bwd", "ms_force_ar", "ms_virial", "ms_total",
    "ms_allgather", "ms_allreduce",
)


def pe_at(log, step):
    for line in Path(log).read_text(errors="replace").splitlines():
        m = THERMO.match(line)
        if m and int(m.group(1)) == step:
            return float(m.group(3))
    raise SystemExit(f"FAIL: no thermo row for step {step} in {log}")


def perf(log, skip):
    rows = []
    for line in Path(log).read_text(errors="replace").splitlines():
        if "MP_PERF rank=0" not in line:
            continue
        row = {}
        for f in FIELDS:
            m = re.search(rf"\b{f}=([0-9.eE+-]+)", line)
            if m:
                row[f] = float(m.group(1))
        rows.append(row)
    rows = rows[skip:]
    if not rows:
        raise SystemExit(f"FAIL: no MP_PERF rows after skipping {skip} in {log}")
    out = {"n_rows": len(rows)}
    for f in FIELDS:
        vals = [r[f] for r in rows if f in r]
        if not vals:
            continue
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals])
        out[f] = {"median": med, "mad": mad, "min": min(vals), "max": max(vals)}
    return out


def _job_arm_stats(jobdir):
    """Per-arm median of slot medians, and the median of those slots' MADs."""
    root = Path(jobdir) / "slots"
    groups = {}
    if not root.is_dir():
        raise SystemExit(f"FAIL: {jobdir} has no slots/")
    for slot in sorted(p for p in root.iterdir() if p.is_dir()):
        label_path = slot / "label"
        perf_path = slot / "perf.json"
        if not label_path.is_file() or not perf_path.is_file():
            continue
        label = label_path.read_text().strip()
        perf_row = json.loads(perf_path.read_text())
        cell = perf_row.get("ms_total")
        if not cell:
            continue
        groups.setdefault(label, {"medians": [], "mads": []})
        groups[label]["medians"].append(cell["median"])
        groups[label]["mads"].append(cell["mad"])
    out = {}
    for label, g in groups.items():
        out[label] = {
            "median": statistics.median(g["medians"]),
            "mad": statistics.median(g["mads"]),
        }
    if "BASE" not in out:
        raise SystemExit(f"FAIL: {jobdir} has no BASE slots with ms_total")
    return out


def abba_verdict(jobs):
    """jobs: list of {label: {median, mad}} including BASE.

    An arm wins only when every job's gain has the same (positive) sign and the
    median gain exceeds max(3 * pooled within-job MAD, between-job range of gains).
    Gain is BASE median minus arm median, in ms.
    """
    if len(jobs) < 3:
        raise SystemExit("FAIL: abba needs at least 3 jobs")
    arms = []
    for job in jobs:
        for label in job:
            if label != "BASE" and label not in arms:
                arms.append(label)
    report = {"n_jobs": len(jobs), "arms": {}}
    for arm in arms:
        gains = []
        mads = []
        per_job = []
        for job in jobs:
            if arm not in job:
                raise SystemExit(f"FAIL: a job is missing arm {arm}")
            gain = job["BASE"]["median"] - job[arm]["median"]
            mad = statistics.median([job["BASE"]["mad"], job[arm]["mad"]])
            gains.append(gain)
            mads.append(mad)
            per_job.append({"gain_ms": gain, "mad_ms": mad,
                            "base_ms": job["BASE"]["median"], "arm_ms": job[arm]["median"]})
        med_gain = statistics.median(gains)
        pooled = statistics.median(mads)
        spread = max(gains) - min(gains)
        same = all(g > 0 for g in gains) or all(g < 0 for g in gains)
        bar = max(3 * pooled, spread)
        win = same and med_gain > 0 and med_gain > bar
        report["arms"][arm] = {
            "per_job": per_job,
            "median_gain_ms": med_gain,
            "pooled_mad_ms": pooled,
            "between_job_range_ms": spread,
            "same_sign": same,
            "bar_ms": bar,
            "win": win,
        }
    return report


def _write_slot(job, label, median, mad):
    slot = Path(job) / "slots" / label
    slot.mkdir(parents=True)
    (slot / "label").write_text(label + "\n")
    (slot / "perf.json").write_text(json.dumps({
        "n_rows": 25,
        "ms_total": {"median": median, "mad": mad, "min": median, "max": median},
    }))


def abba_self_test():
    import tempfile
    cases = {
        "win": (
            [(100, 80), (100, 80), (101, 81)],
            True,
        ),
        "sign": (
            [(100, 90), (100, 90), (100, 101)],
            False,
        ),
        "spread": (
            [(100, 95), (100, 95), (100, 80)],
            False,
        ),
    }
    for name, (pairs, expect) in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            jobs = []
            for i, (base, arm) in enumerate(pairs):
                job = Path(tmp) / f"job{i}"
                _write_slot(job, "BASE", base, 0.01)
                _write_slot(job, "ARM", arm, 0.01)
                jobs.append(_job_arm_stats(job))
            got = abba_verdict(jobs)["arms"]["ARM"]["win"]
            if got != expect:
                raise SystemExit(f"FAIL: abba-test {name}: win={got} expected={expect}")
    print("ABBA_TEST_OK")


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cmd = argv[1]
    if cmd == "pe":
        print(f"{pe_at(argv[2], int(argv[3])):.16e}")
    elif cmd == "bar":
        pe = pe_at(argv[2], int(argv[3]))
        ref, tol = float(argv[4]), float(argv[5])
        d = abs(pe - ref)
        print(f"BAR step={argv[3]} pe={pe:.16e} ref={ref:.16e} dE={d:.3e} tol={tol:.1e}")
        if d >= tol:
            raise SystemExit(1)
    elif cmd == "perf":
        skip = int(argv[3]) if len(argv) > 3 else 5
        print(json.dumps(perf(argv[2], skip), indent=2, sort_keys=True))
    elif cmd == "bit":
        a, b = pe_at(argv[2], int(argv[4])), pe_at(argv[3], int(argv[4]))
        print(f"BIT step={argv[4]} equal={a == b} diff={abs(a - b):.3e}")
    elif cmd == "abba-test":
        abba_self_test()
    elif cmd == "abba":
        if len(argv) < 5:
            raise SystemExit("FAIL: abba needs at least 3 job directories")
        jobs = [_job_arm_stats(p) for p in argv[2:]]
        print(json.dumps(abba_verdict(jobs), indent=2, sort_keys=True))
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
