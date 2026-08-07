#!/usr/bin/env python3
"""Merge geom_sweep/*/probe_status.json into SWEEP.md + SWEEP.json after jobs finish."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results" / "geom_sweep"


def main() -> int:
    rows = []
    for st in sorted(ROOT.glob("N*/**/probe_status.json")):
        rows.append(json.loads(st.read_text()))
    if not rows:
        print(f"no probe_status.json under {ROOT}")
        return 1

    by_n: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_n[int(r["n"])][r["path"]] = r

    # Active paths only (uma_mixed disabled — do not require for N*).
    paths = ["ase", "fc", "uma_double"]
    lines = [
        "# Phase G1 OOM sweep",
        "",
        f"_Stamp:_ auto-merged from `probe_status.json` under `{ROOT}`.",
        "",
        "**Precision:** ase / fc / `uma_double` (FP64). `uma_mixed` disabled.",
        "",
        "| N | natoms | ase | fc | uma_double | all_pass |",
        "|--:|------:|-----|----|------------|----------|",
    ]
    safe = []
    for n in sorted(by_n):
        m = by_n[n]
        natoms = next(
            (int(v["natoms"]) for v in m.values() if "natoms" in v),
            8 * n**3,
        )
        cells = []
        ok = True
        for p in paths:
            s = (m.get(p) or {}).get("status", "MISSING")
            cells.append(s)
            if s != "PASS":
                ok = False
        if ok:
            safe.append(n)
        lines.append(
            f"| {n} | {natoms} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{'YES' if ok else 'no'} |"
        )

    n_star = max(safe) if safe else None
    lines += [
        "",
        f"**N\\*** (largest all-pass over ase/fc/uma_double) = **{n_star}**",
        "",
        "Notes:",
        "",
        "- N=12 ASE = OOM on A100 40GB (13824 atoms).",
        "- N=12 fc / uma_double may still be PENDING in the queue.",
        "",
    ]
    (ROOT / "SWEEP.md").write_text("\n".join(lines) + "\n")
    summary = {"rows": rows, "n_star": n_star, "safe_N": safe}
    (ROOT / "SWEEP.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
