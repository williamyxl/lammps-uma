#!/usr/bin/env python3
"""Parse LAMMPS thermo logs and MP_PERF lines.

Subcommands
  pe   LOG STEP                 print the PE at STEP with %.16e
  bar  LOG STEP REF TOL         exit 1 unless |PE(STEP) - REF| < TOL
  perf LOG [SKIP]               JSON with median / MAD / min / max of the MP_PERF
                                fields over the lines after the first SKIP (default 5)
  bit  LOG_A LOG_B STEP         print whether PE(STEP) is bitwise equal and the diff
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
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
