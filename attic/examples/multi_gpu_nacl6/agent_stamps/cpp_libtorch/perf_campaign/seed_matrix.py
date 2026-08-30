#!/usr/bin/env python3
"""Seed / update MATRIX.json for settings-matrix campaign."""
from __future__ import annotations

import json
from pathlib import Path

CAMP = Path(__file__).resolve().parent
ENG = CAMP.parents[4] / "uma-engine"  # .../ML-UMA/uma-engine — adjust if wrong
# CAMP = .../multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign
# parents: perf_campaign, cpp_libtorch, agent_stamps, multi_gpu_nacl6, examples, ML-UMA
ML_UMA = CAMP.parents[3]
ENG = ML_UMA / "uma-engine"

ART = {
    "gen": str(ENG / "artifacts" / "uma-s-1p2-omat-f64"),
    "gmerge": str(ENG / "artifacts" / "uma-s-1p2-omat-f64-merge"),
    "ufast": str(ENG / "artifacts" / "uma-s-1p2-omat-f64-fast"),
}

# Seed known cells: path, system, tag, ngpu -> record
cells = []

def add(system, path, tag, ngpu, *, ms, status, job=None, note=None, metric=None):
    cells.append(
        {
            "system": system,
            "path": path,
            "tag": tag,
            "ngpu": ngpu,
            "ms": ms,
            "status": status,
            "job": job,
            "note": note,
            "metric": metric,
        }
    )

# Illegal
for system in ("nacl6", "water888"):
    for path in ("ase", "fc", "uma"):
        for ngpu in (1, 2, 4):
            add(system, path, "ufast_nomole", ngpu, ms=None, status="SKIP_ILLEGAL",
                note="umas_fast_pytorch requires merge_mole=True")

# FC merge crash
for system in ("nacl6", "water888"):
    for tag in ("gmerge", "ufast"):
        for ngpu in (1, 2, 4):
            add(system, "fc", tag, ngpu, ms=None, status="SKIP_KNOWN_CRASH",
                note="FC+merge_mole+FP64 Float/Double in merge_MOLE")

# Locked gen ASE/FC
for ngpu, ase, fc in ((1, 396.5, 345.5), (2, 193.9, 193.2), (4, 115.2, 118.0)):
    add("nacl6", "ase", "gen", ngpu, ms=ase, status="REUSED", metric="ms_per_eval_python")
    add("nacl6", "fc", "gen", ngpu, ms=fc, status="REUSED", metric="ms_per_eval_python")
for ngpu, ase, fc in ((1, 382.09, 359.40), (2, 198.19, 200.54), (4, 117.98, 118.94)):
    add("water888", "ase", "gen", ngpu, ms=ase, status="REUSED", metric="nvt_pair_ms_per_step")
    add("water888", "fc", "gen", ngpu, ms=fc, status="REUSED", metric="nvt_pair_ms_per_step")

# NaCl ASE matching
add("nacl6", "ase", "gmerge", 2, ms=195.7, status="DONE", job="20989338", metric="ms_per_eval_python")
add("nacl6", "ase", "gmerge", 4, ms=167.9, status="DONE", job="20989346", metric="ms_per_eval_python")
add("nacl6", "ase", "ufast", 2, ms=191.6, status="DONE", job="20989339", metric="ms_per_eval_python")
add("nacl6", "ase", "ufast", 4, ms=164.5, status="DONE", job="20989347", metric="ms_per_eval_python")

# Uma W7 ufast + Tier0 gen @2/@4
add("nacl6", "uma", "ufast", 2, ms=159.4, status="DONE", job="20989184", metric="ms_per_eval_python")
add("nacl6", "uma", "ufast", 4, ms=92.37, status="DONE", job="20989185", metric="ms_per_eval_python")
add("water888", "uma", "ufast", 2, ms=165.10, status="DONE", job="20989186", metric="nvt_pair_ms_per_step")
add("water888", "uma", "ufast", 4, ms=96.84, status="DONE", job="20989187", metric="nvt_pair_ms_per_step")
add("nacl6", "uma", "gen", 2, ms=172.9, status="REUSED", note="Tier0", metric="ms_per_eval_python")
add("nacl6", "uma", "gen", 4, ms=100.2, status="REUSED", note="Tier0", metric="ms_per_eval_python")
add("water888", "uma", "gen", 2, ms=178.3, status="REUSED", note="Tier0", metric="nvt_pair_ms_per_step")
add("water888", "uma", "gen", 4, ms=104.2, status="REUSED", note="Tier0", metric="nvt_pair_ms_per_step")
add("nacl6", "uma", "gmerge", 2, ms=170.0, status="REUSED", note="Tier1c", metric="ms_per_eval_python")

best = {
    "nacl6": {
        "1": {"ase": {"tag": "gen", "ms": 396.5}, "fc": {"tag": "gen", "ms": 345.5}, "uma": None},
        "2": {"ase": {"tag": "ufast", "ms": 191.6}, "fc": {"tag": "gen", "ms": 193.2},
              "uma": {"tag": "ufast", "ms": 159.4}, "floor": "PASS", "margin_ase": 32.2, "margin_fc": 33.8},
        "4": {"ase": {"tag": "ufast", "ms": 164.5}, "fc": {"tag": "gen", "ms": 118.0},
              "uma": {"tag": "ufast", "ms": 92.37}, "floor": "PASS", "margin_ase": 72.1, "margin_fc": 25.6},
    },
    "water888": {
        "1": {"ase": {"tag": "gen", "ms": 382.09}, "fc": {"tag": "gen", "ms": 359.40}, "uma": None},
        "2": {"ase": {"tag": "gen", "ms": 198.19, "note": "ufast pending"}, "fc": {"tag": "gen", "ms": 200.54},
              "uma": {"tag": "ufast", "ms": 165.10}, "floor": "PASS_vs_gen", "margin_ase": 33.09, "margin_fc": 35.44},
        "4": {"ase": {"tag": "gen", "ms": 117.98, "note": "ufast pending"}, "fc": {"tag": "gen", "ms": 118.94},
              "uma": {"tag": "ufast", "ms": 96.84}, "floor": "PASS_vs_gen", "margin_ase": 21.14, "margin_fc": 22.1},
    },
}

out = {
    "stamp": "2026-08-09T20:26:00",
    "policy": "BEST_BARS are minimum floor; max-push to hard ceiling",
    "artifacts": ART,
    "best_bars": best,
    "cells": cells,
}
(CAMP / "MATRIX.json").write_text(json.dumps(out, indent=2) + "\n")
print("WROTE", CAMP / "MATRIX.json", "n_cells", len(cells))
