#!/usr/bin/env python3
"""Summarize a V7 wave: per-atom E/F, SP + NVT timing, and the W18 breakdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CAMP = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    cells = []
    for f in sorted(CAMP.glob(f"gate_v7_{a.wave}_*.json")):
        try:
            cells.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    if not cells:
        print(f"no gate_v7_{a.wave}_* stamps yet")
        return 1
    if a.json:
        print(json.dumps(cells, indent=2))
        return 0

    def fmt(v, spec: str, width: int) -> str:
        """Format or dash, right-aligned. Avoids nested quotes in f-strings."""
        if v is None:
            return "-".rjust(width)
        return format(v, spec).rjust(width)

    print(f"\n=== V7 {a.wave} ===")
    print(f"{'sys':<10}{'GPUs':>5}{'SP ms':>9}{'NVT ms':>9}{'vs W8nk':>9}"
          f"{'|dE|':>10}{'maxPA|dF|':>11}{'>tol':>6}{'status':>22}")
    for c in sorted(cells, key=lambda x: (x.get("sys", ""), x.get("ngpu", 0))):
        print(f"{c.get('sys',''):<10}{c.get('ngpu',0):>5}"
              + fmt(c.get("sp_ms"), ".1f", 9)
              + fmt(c.get("nvt_pair_ms_per_step"), ".2f", 9)
              + fmt(c.get("delta_vs_w8nk_ms"), "+.2f", 9)
              + fmt(c.get("dE_vs_merge_ase"), ".1e", 10)
              + fmt(c.get("force_max_per_atom"), ".1e", 11)
              + str(c.get("n_atoms_over_tol", "-")).rjust(6)
              + str(c.get("status", "")).rjust(22))

    # W18 accounting: did we actually close the 33.7 ms gap?
    have = [c for c in cells if c.get("tick_ms_accounted")]
    if have:
        print("\n--- worker step breakdown (median ms/rank) ---")
        ks = ("tick_ms_h2d", "tick_ms_wshard", "tick_ms_pad", "tick_ms_prep",
              "tick_ms_fwd", "tick_ms_bar_pre_bwd", "tick_ms_bwd",
              "tick_ms_post", "tick_ms_accounted")
        hdr = "".join(f"{k.replace('tick_ms_',''):>9}" for k in ks)
        print(f"{'sys':<10}{'GPUs':>5}{hdr}{'wait':>9}{'resid':>8}")
        for c in sorted(have, key=lambda x: (x.get("sys",""), x.get("ngpu",0))):
            row = "".join(f"{c.get(k,0):>9.2f}" for k in ks)
            print(f"{c.get('sys',''):<10}{c.get('ngpu',0):>5}{row}"
                  f"{c.get('parent_ms_wait_workers',0):>9.2f}"
                  f"{c.get('accounting_residual_ms',0):>8.2f}")
        closed = [c for c in have if c.get("accounting_closed")]
        print(f"\naccounting closed (residual <= 2 ms): "
              f"{len(closed)}/{len(have)} cells")

    ef = [c for c in cells if c.get("ef_pass")]
    prom = [c for c in cells if c.get("promote")]
    print(f"\nE/F PASS: {len(ef)}/{len(cells)}   promote-candidates: {len(prom)}")
    if len(ef) != len(cells):
        print("!! parity regression — do not promote")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
