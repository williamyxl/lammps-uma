#!/usr/bin/env python3
"""Record a stage failure (e.g. export OOM) as an N-sweep data point."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EX = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--job", default=os.environ.get("SLURM_JOB_ID", "manual"))
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    out = EX / "results" / f"n{a.n}_{a.job}"
    out.mkdir(parents=True, exist_ok=True)
    rec = {
        "n": a.n, "natoms": 8 * a.n**3, "job": a.job,
        "stage": a.stage, "status": a.status, "note": a.note,
        "functional": False,
        "oom": "oom" in a.note.lower() or a.status.upper() == "OOM",
    }
    (out / "cell.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"NSWEEP_RECORD {out / 'cell.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
