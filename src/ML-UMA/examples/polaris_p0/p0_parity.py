#!/usr/bin/env python3
"""Phase-P0 parity + report: LAMMPS-UMA (devices 1/2/4) vs ASE-FairChem FP64.

For each system and each device count, reads <sys>_d<N>_uma.{json,npz} and the
per-system oracle <sys>_ase.{json,npz}, computes energy/force deltas against the
gates, and writes P0_RESULTS.json + P0_REPORT.md. Exit 0 iff every cell passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p0_common import (
    GATE_ABS_DE,
    GATE_MAX_DF,
    GATE_REL_DE,
    SYSTEMS,
    results_dir,
)


def load_npz_json(d: Path, stem: str):
    j = json.loads((d / f"{stem}.json").read_text())
    npz = np.load(d / f"{stem}.npz")
    return j, npz["forces"].astype(np.float64), float(npz["energy_eV"])


def recipe_for_devices(devices: int) -> str:
    # devices 1 = traced general-mode model; devices>1 = MP shards (fast+merge).
    return "general" if devices == 1 else "fastmerge"


def one_cell(d: Path, sysname: str, devices: int, ase_f, ase_e, recipe: str) -> dict:
    uma_j, uma_f, uma_e = load_npz_json(d, f"{sysname}_d{devices}_uma")
    dE = abs(uma_e - ase_e)
    relE = dE / max(abs(ase_e), 1e-30)
    energy_pass = (dE <= GATE_ABS_DE) or (relE <= GATE_REL_DE)
    if uma_f.shape == ase_f.shape:
        mag = np.linalg.norm(uma_f - ase_f, axis=1)
        max_df, mean_df = float(mag.max()), float(mag.mean())
        force_pass = max_df <= GATE_MAX_DF
    else:
        max_df = mean_df = None
        force_pass = False
    return {
        "system": sysname,
        "devices": devices,
        "recipe": recipe,
        "natoms": uma_j["natoms"],
        "energy_lammps_eV": uma_e,
        "abs_dE": dE,
        "rel_dE": relE,
        "energy_pass": energy_pass,
        "force_max_per_atom": max_df,
        "force_mean_per_atom": mean_df,
        "force_pass": force_pass,
        "nvt_ms_step": uma_j.get("nvt_ms_per_step"),
        "pass": bool(energy_pass and force_pass),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="*", default=sorted(SYSTEMS))
    ap.add_argument("--devices", nargs="*", type=int, default=[1, 2, 4])
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    d = Path(args.results_dir) if args.results_dir else results_dir()

    recs = []
    for s in args.systems:
        oracle = {}
        for rcp in ("general", "fastmerge"):
            try:
                _, of, oe = load_npz_json(d, f"{s}_ase_{rcp}")
                oracle[rcp] = (of, oe)
            except FileNotFoundError:
                pass
        for dev in args.devices:
            rcp = recipe_for_devices(dev)
            if rcp not in oracle:
                recs.append({"system": s, "devices": dev, "pass": False,
                             "error": f"missing oracle recipe {rcp}"})
                continue
            of, oe = oracle[rcp]
            try:
                recs.append(one_cell(d, s, dev, of, oe, rcp))
            except FileNotFoundError as e:
                recs.append({"system": s, "devices": dev, "pass": False,
                             "error": f"missing: {e}"})

    overall = all(r.get("pass") for r in recs)
    (d / "P0_RESULTS.json").write_text(json.dumps(
        {"gates": {"abs_dE": GATE_ABS_DE, "rel_dE": GATE_REL_DE, "max_dF": GATE_MAX_DF},
         "cells": recs, "P0_PASS": overall}, indent=2) + "\n")

    lines = [
        "# Phase P0 - Polaris single-node validation (1/2/4 GPUs)",
        "",
        "LAMMPS + LibTorch-UMA (precision double, devices 1/2/4) vs ASE-FairChem FP64 (task=omat).",
        "",
        f"Gates: |dE| <= {GATE_ABS_DE:g} eV (or rel <= {GATE_REL_DE:g}); "
        f"per-atom max|dF| <= {GATE_MAX_DF:g} eV/A.",
        "",
        "Each LAMMPS path is compared to the ASE-FC FP64 oracle in the MATCHING "
        "recipe: devices 1 = general (traced model); devices 2/4 = "
        "umas_fast_pytorch + merge_mole (MP shards).",
        "",
        "| system | N | GPUs | recipe | E_lammps (eV) | |dE| | max|dF| | mean|dF| | NVT ms/step | speedup | verdict |",
        "|---|---:|---:|:--:|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    base_ms = {}
    for r in recs:
        if "error" in r:
            g = r.get("devices", "-")
            lines.append(f"| {r['system']} | - | {g} | - | - | - | - | - | - | - | FAIL ({r['error']}) |")
            continue
        if r["devices"] == 1 and r.get("nvt_ms_step"):
            base_ms[r["system"]] = r["nvt_ms_step"]
        ms = r["nvt_ms_step"]
        sp = ("-" if (ms is None or r["system"] not in base_ms or not ms)
              else f"{base_ms[r['system']]/ms:.2f}x")
        mdf = "-" if r["force_max_per_atom"] is None else f"{r['force_max_per_atom']:.3e}"
        adf = "-" if r["force_mean_per_atom"] is None else f"{r['force_mean_per_atom']:.3e}"
        msf = "-" if ms is None else f"{ms:.2f}"
        lines.append(
            f"| {r['system']} | {r['natoms']} | {r['devices']} | {r['recipe']} | "
            f"{r['energy_lammps_eV']:.9f} | {r['abs_dE']:.3e} | {mdf} | {adf} | "
            f"{msf} | {sp} | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    lines += ["", f"**P0 {'PASS' if overall else 'FAIL'}**", ""]
    (d / "P0_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {d / 'P0_REPORT.md'} and {d / 'P0_RESULTS.json'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
