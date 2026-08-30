#!/usr/bin/env python3
"""Compiled energy + per-atom force + timing report across all four paths.

Paths
-----
  1. ASE FC FP64            (frozen; locked bars reused)
  2. FC LAMMPS              (frozen; locked bars reused)
  3. LibTorch UMA LAMMPS    (product W8nk + live V7 optimization waves)
  4. ALCHEMI (nvalchemi)    (external toolkit)

Built from on-disk artifacts only. Anything not measured is printed as "-"
rather than inferred, and every number carries its provenance.

  python report_all_paths.py              # markdown report to stdout
  python report_all_paths.py --write      # also write REPORT_ALL_PATHS.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

CAMP = Path(__file__).resolve().parent
EXAMPLES = CAMP.parents[3]
ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")

ORACLE = {
    "nacl6": (CAMP / "oracle_ase_umas_fast_merge.npz", -5830.9237413382, 1728),
    "water888": (CAMP / "oracle_ase_water_merge.npz", -3143.3893774722696, 648),
}
DE_TOL, DF_TOL = 1e-6, 1e-5

# Frozen baselines (STATE.json). Metric differs per path, so it is labelled.
LOCKED = {
    ("ASE FC FP64 (general)", "nacl6"): {1: 396.5, 2: 193.9, 4: 115.2},
    ("ASE FC FP64 (general)", "water888"): {1: 382.09, 2: 198.19, 4: 117.98},
    ("ASE FC FP64 (ufast+merge)", "nacl6"): {1: 350.0, 2: 191.6, 4: 164.5},
    ("ASE FC FP64 (ufast+merge)", "water888"): {2: 165.5, 4: 94.5},
    ("FC LAMMPS (general)", "nacl6"): {1: 345.5, 2: 193.2, 4: 118.0},
    ("FC LAMMPS (general)", "water888"): {1: 359.4, 2: 200.54, 4: 118.94},
}
W8NK = {("nacl6", 1): 296.48, ("nacl6", 2): 161.94, ("nacl6", 4): 92.097,
        ("water888", 2): 164.82, ("water888", 4): 95.744}
# LibTorch UMA runs with saved forces (product W8nk).
W8NK_RUNS = {
    ("nacl6", 1): EXAMPLES / "multi_gpu_nacl6/results/uma_ngpu1_21011180",
    ("nacl6", 2): EXAMPLES / "multi_gpu_nacl6/results/uma_ngpu2_21010252",
    ("nacl6", 4): EXAMPLES / "multi_gpu_nacl6/results/uma_ngpu4_21010253",
    ("water888", 2): EXAMPLES / "water888/results/uma_ngpu2_21010254",
    ("water888", 4): EXAMPLES / "water888/results/uma_ngpu4_21010255",
}


def parity(forces_npz: Path, energy, sysname: str) -> dict:
    """Per-atom force + energy parity vs the merge oracle."""
    out: dict = {}
    onpz, oe, _ = ORACLE[sysname]
    if energy is not None:
        out["dE"] = abs(float(energy) - oe)
    if forces_npz.is_file() and onpz.is_file():
        f = np.load(forces_npz)["forces"]
        fr = np.load(onpz)["forces"]
        if f.shape == fr.shape:
            mag = np.linalg.norm(f - fr, axis=1)
            out["fmax"] = float(mag.max())
            out["fmean"] = float(mag.mean())
            out["nover"] = int((mag > DF_TOL).sum())
            out["natoms"] = int(f.shape[0])
    if "dE" in out and "fmax" in out:
        out["pass"] = bool(out["dE"] <= DE_TOL and out["fmax"] <= DF_TOL)
    return out


def collect() -> dict:
    data: dict = {"nacl6": {}, "water888": {}}

    # --- LibTorch UMA LAMMPS (product W8nk) ---
    for (s, ng), d in W8NK_RUNS.items():
        tj = d / "timing.json"
        if not tj.is_file():
            continue
        t = json.loads(tj.read_text())
        row = {"ms": t.get("nvt_pair_ms_per_step"), "sp_ms": t.get("sp_ms"),
               "metric": "LAMMPS Pair ms/step", "job": t.get("jobid"),
               "energy": t.get("energy_eV")}
        row.update(parity(d / "forces.npz", t.get("energy_eV"), s))
        data[s].setdefault("LibTorch UMA LAMMPS (W8nk)", {})[ng] = row

    # --- LibTorch UMA LAMMPS, live V7 waves ---
    for g in sorted(CAMP.glob("gate_v7_*.json")):
        try:
            c = json.loads(g.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        s, ng, wave = c.get("sys"), c.get("ngpu"), c.get("ver")
        if s not in data or ng is None:
            continue
        row = {
            "ms": c.get("nvt_pair_ms_per_step"),
            "sp_ms": c.get("sp_ms"),
            "shell_ms": c.get("shell_nvt_ms_per_step_shell"),
            "metric": "LAMMPS Pair ms/step (+shell)",
            "job": c.get("job"),
            "energy": c.get("energy_eV"),
            "dE": c.get("dE_vs_merge_ase"),
            "fmax": c.get("force_max_per_atom"),
            "fmean": c.get("force_mean_per_atom"),
            "nover": c.get("n_atoms_over_tol"),
            "natoms": c.get("n_atoms") or c.get("natoms"),
            "pass": c.get("ef_pass"),
            "delta_w8nk": c.get("delta_vs_w8nk_ms"),
            "resid": c.get("accounting_residual_ms"),
        }
        data[s].setdefault(f"LibTorch UMA V7 [{wave}]", {})[ng] = row

    # --- ALCHEMI ---
    nvdir = EXAMPLES / "nvalchemi_path/results"
    for d in sorted(nvdir.glob("*_ngpu*_*")):
        tj = d / "timing.json"
        if not tj.is_file():
            continue
        t = json.loads(tj.read_text())
        s, ng = t.get("sys"), t.get("ngpu")
        if s not in data or ng is None:
            continue
        row = {"ms": t.get("nvt_ms_per_step"), "sp_ms": t.get("sp_ms"),
               "metric": "in-code NVT ms/step (warmup excl.)",
               "job": t.get("job"), "energy": t.get("energy_eV"),
               "dE": t.get("dE_vs_merge_ase"),
               "fmax": t.get("force_max_per_atom"),
               "fmean": t.get("force_mean_per_atom"),
               "nover": t.get("n_atoms_over_tol"),
               "natoms": t.get("natoms"), "pass": t.get("ef_pass"),
               "warm": bool(t.get("cold_start_excluded"))}
        prev = data[s].setdefault("ALCHEMI (nvalchemi)", {}).get(ng)
        if prev is None or (row.get("warm") and not prev.get("warm")):
            data[s]["ALCHEMI (nvalchemi)"][ng] = row
    return data


def fmt(v, spec: str, w: int) -> str:
    if v is None:
        return "-".rjust(w)
    try:
        return format(v, spec).rjust(w)
    except (TypeError, ValueError):
        return str(v).rjust(w)


def render(data: dict) -> str:
    L: list[str] = []
    L.append("# Four-path comparison — energy, per-atom force, timing")
    L.append("")
    L.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    L.append("**Precision:** FP64 · **Ensemble:** NVT 300 K · "
             "**Oracle:** ASE FP64 `umas_fast_pytorch`+`merge_mole`")
    L.append("")
    L.append("Parity gate: `|dE| <= 1e-6 eV` **and** per-atom "
             "`max|dF| <= 1e-5 eV/A`. Net force `|sum F|` is deliberately not "
             "used: it is bit-identical under sign inversion.")
    L.append("")

    for s in ("nacl6", "water888"):
        _, oe, nat = ORACLE[s]
        L.append(f"## {s} ({nat} atoms)")
        L.append("")
        L.append(f"Oracle energy: `{oe:.9f}` eV")
        L.append("")
        L.append("### Energy + per-atom force parity")
        L.append("")
        L.append("| Path | GPUs | Energy (eV) | \\|dE\\| | max/atom \\|dF\\| | "
                 "mean \\|dF\\| | >tol | Verdict |")
        L.append("|---|---:|---:|---:|---:|---:|---:|:---:|")
        any_row = False
        for path in sorted(data[s]):
            for ng in sorted(data[s][path]):
                r = data[s][path][ng]
                if r.get("dE") is None and r.get("fmax") is None:
                    continue
                any_row = True
                v = "PASS" if r.get("pass") else ("FAIL" if r.get("pass") is False else "-")
                L.append(
                    f"| {path} | {ng} | "
                    f"{fmt(r.get('energy'), '.6f', 1).strip() if r.get('energy') else '-'} | "
                    f"{fmt(r.get('dE'), '.2e', 1).strip()} | "
                    f"{fmt(r.get('fmax'), '.2e', 1).strip()} | "
                    f"{fmt(r.get('fmean'), '.2e', 1).strip()} | "
                    f"{r.get('nover', '-')} | **{v}** |")
        if not any_row:
            L.append("| _(no force data yet)_ | | | | | | | |")
        L.append("")
        L.append("### Timing (ms/step or ms/eval)")
        L.append("")
        L.append("| Path | @1 | @2 | @4 | Metric |")
        L.append("|---|---:|---:|---:|---|")
        for name in ("ASE FC FP64 (general)", "ASE FC FP64 (ufast+merge)",
                     "FC LAMMPS (general)"):
            bars = LOCKED.get((name, s))
            if bars:
                L.append(f"| {name} | {fmt(bars.get(1), '.1f', 1).strip()} | "
                         f"{fmt(bars.get(2), '.1f', 1).strip()} | "
                         f"{fmt(bars.get(4), '.1f', 1).strip()} | "
                         f"locked baseline (frozen) |")
        for path in sorted(data[s]):
            cells = data[s][path]
            vals = []
            for ng in (1, 2, 4):
                r = cells.get(ng) or {}
                vals.append(fmt(r.get("ms"), ".1f", 1).strip())
            metric = next((c.get("metric") for c in cells.values()
                           if c.get("metric")), "-")
            L.append(f"| {path} | {vals[0]} | {vals[1]} | {vals[2]} | {metric} |")
        L.append("")

    # V7 worker breakdown, if present
    v7 = [json.loads(g.read_text()) for g in sorted(CAMP.glob("gate_v7_*.json"))]
    v7 = [c for c in v7 if c.get("tick_ms_accounted")]
    if v7:
        L.append("## V7 worker step breakdown (median ms/rank)")
        L.append("")
        L.append("| wave | sys | GPUs | h2d | shard | pad | prep | fwd | "
                 "barrier | bwd | post | accounted | wait | resid |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in sorted(v7, key=lambda x: (x.get("sys", ""), x.get("ngpu", 0))):
            L.append(
                f"| {c.get('ver')} | {c.get('sys')} | {c.get('ngpu')} | "
                + " | ".join(fmt(c.get(k), '.2f', 1).strip() for k in (
                    "tick_ms_h2d", "tick_ms_wshard", "tick_ms_pad",
                    "tick_ms_prep", "tick_ms_fwd", "tick_ms_bar_pre_bwd",
                    "tick_ms_bwd", "tick_ms_post", "tick_ms_accounted",
                    "parent_ms_wait_workers", "accounting_residual_ms"))
                + " |")
        L.append("")

    L.append("## Notes and caveats")
    L.append("")
    L.append("- **ASE FC / FC LAMMPS are frozen.** Their code is unchanged, so "
             "locked bars are reused rather than re-run.")
    L.append("- **FC LAMMPS has no FP64+`merge_mole` row**: `merge_MOLE` "
             "raises a Float/Double error, so the matching-settings bar is "
             "blocked upstream.")
    L.append("- **Metrics are not interchangeable.** LAMMPS paths report the "
             "internal Pair timer; V7 additionally reports a shell-level "
             "differential measured in the SLURM script, external to LAMMPS; "
             "ALCHEMI reports an in-code timer. Compare within a column.")
    L.append("- **ALCHEMI multi-GPU** is spatial domain decomposition (halo); "
             "LibTorch UMA is model-parallel with NCCL. A speed gap between "
             "them is not a like-for-like implementation comparison.")
    L.append("- **ALCHEMI water888 fails parity** (dE 6.4e-3, all 648 atoms "
             "over tol) while nacl6 passes at 1.7e-14. Isolated to nvalchemi's "
             "own input adaptation: plain fairchem on the identical structure "
             "passes at 1.4e-7, and TF32 was ruled out. Upstream defect.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    txt = render(collect())
    print(txt)
    if a.write:
        (CAMP / "REPORT_ALL_PATHS.md").write_text(txt + "\n")
        print(f"\n[written] {CAMP / 'REPORT_ALL_PATHS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
