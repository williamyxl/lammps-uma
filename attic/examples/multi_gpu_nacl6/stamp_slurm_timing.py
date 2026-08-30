#!/usr/bin/env python3
"""Overwrite path ms_per_eval from SLURM-measured wall time (sole timing source).

Usage (from _run_common.sh after timed run_multigpu.py):
  python stamp_slurm_timing.py --results-dir DIR --wall-s SEC --n-timing N --paths ase
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--wall-s", type=float, required=True)
    ap.add_argument("--n-timing", type=int, required=True)
    ap.add_argument(
        "--paths",
        default=os.environ.get("ONLY_PATHS", ""),
        help="comma-separated path keys to stamp (default ONLY_PATHS)",
    )
    ap.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    args = ap.parse_args()

    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    if not paths:
        raise SystemExit("stamp_slurm_timing: no paths to stamp")
    if args.n_timing < 1:
        raise SystemExit("n_timing must be >= 1")
    if args.wall_s <= 0:
        raise SystemExit(f"wall_s must be > 0, got {args.wall_s}")

    # Sole reported metric: amortize full path-job wall over N_TIMING evals.
    ms = (args.wall_s / args.n_timing) * 1e3

    parity_path = args.results_dir / "parity.json"
    if not parity_path.is_file():
        raise SystemExit(f"missing {parity_path}")

    report = json.loads(parity_path.read_text())
    stamped = []
    for row in report.get("rows") or []:
        key = row.get("key")
        if key not in paths:
            continue
        # Keep python/pair measurement only as debug breadcrumb.
        if row.get("ms_per_eval") is not None and "ms_per_eval_python" not in row:
            row["ms_per_eval_python"] = row["ms_per_eval"]
        row["ms_per_eval"] = ms
        row["slurm_wall_s"] = args.wall_s
        row["timing_source"] = "slurm_wall"
        row["slurm_job_id"] = args.job_id or None
        stamped.append(key)

    if not stamped:
        raise SystemExit(
            f"no matching rows for paths={paths} in {parity_path} "
            f"(keys={[r.get('key') for r in report.get('rows') or []]})"
        )

    report["timing_policy"] = {
        "source": "slurm_wall",
        "note": (
            "ms_per_eval = 1000 * slurm_wall_s / N_TIMING; "
            "wall covers run_multigpu.py only (load + evals + teardown), "
            "not module/rebuild/collect_results"
        ),
        "slurm_wall_s": args.wall_s,
        "n_timing": args.n_timing,
        "ms_per_eval": ms,
        "paths_stamped": stamped,
        "slurm_job_id": args.job_id or None,
    }
    parity_path.write_text(json.dumps(report, indent=2) + "\n")

    stamp = {
        "slurm_wall_s": args.wall_s,
        "n_timing": args.n_timing,
        "ms_per_eval": ms,
        "paths": stamped,
        "slurm_job_id": args.job_id or None,
        "results_dir": str(args.results_dir),
    }
    (args.results_dir / "timing_slurm.json").write_text(
        json.dumps(stamp, indent=2) + "\n"
    )
    print(
        f"SLURM timing: wall={args.wall_s:.4f}s  N_TIMING={args.n_timing}  "
        f"ms_per_eval={ms:.4f}  paths={stamped}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
