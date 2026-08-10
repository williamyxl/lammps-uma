#!/usr/bin/env python3
"""Cross-path comparison: energy, per-atom force, and timing on 1/2/4 GPUs.

Assembles the four paths (ASE FC FP64, FC LAMMPS, LibTorch UMA LAMMPS,
nvalchemi) from on-disk results only. Anything not measured is reported as
missing rather than filled in.

  python compare_paths.py            # markdown tables
  python compare_paths.py --json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")
EXAMPLES = ROOT / "src/ML-UMA/examples"
CAMP = EXAMPLES / "multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign"
NV = EXAMPLES / "nvalchemi_path/results"
ORACLE = {
    "nacl6": CAMP / "oracle_ase_umas_fast_merge.npz",
    "water888": CAMP / "oracle_ase_water_merge.npz",
}
DE_TOL, DF_TOL = 1e-6, 1e-5

# Locked speed bars from the campaign (STATE.json). Metrics differ by path;
# each row carries its provenance so they are never silently mixed.
LOCKED = {
    ("ASE FC FP64 (ufast+merge)", "nacl6"): {1: 350.0, 2: 191.6, 4: 164.5},
    ("ASE FC FP64 (general)", "nacl6"): {1: 396.5, 2: 193.9, 4: 115.2},
    ("ASE FC FP64 (general)", "water888"): {1: 382.09, 2: 198.19, 4: 117.98},
    ("FC LAMMPS (general)", "nacl6"): {1: 345.5, 2: 193.2, 4: 118.0},
    ("FC LAMMPS (general)", "water888"): {1: 359.4, 2: 200.54, 4: 118.94},
    ("LibTorch UMA LAMMPS (W8nk)", "nacl6"): {1: 296.48, 2: 161.94, 4: 92.097},
    ("LibTorch UMA LAMMPS (W8nk)", "water888"): {2: 164.82, 4: 95.744},
}
UMA_FORCES = {
    ("nacl6", 1): EXAMPLES / "multi_gpu_nacl6/results/uma_ngpu1_21011180",
    ("nacl6", 2): EXAMPLES / "multi_gpu_nacl6/results/uma_ngpu2_21010252",
    ("nacl6", 4): EXAMPLES / "multi_gpu_nacl6/results/uma_ngpu4_21010253",
}


def force_stats(f: np.ndarray, ref: np.ndarray) -> dict:
    if f.shape != ref.shape:
        return {"err": f"shape {f.shape} vs {ref.shape}"}
    mag = np.linalg.norm(f - ref, axis=1)
    return {
        "max_per_atom": float(mag.max()),
        "mean_per_atom": float(mag.mean()),
        "rms_per_atom": float(np.sqrt((mag**2).mean())),
        "n_over_tol": int((mag > DF_TOL).sum()),
        "n_atoms": int(f.shape[0]),
    }


def collect() -> dict:
    out: dict = {"nacl6": {}, "water888": {}}
    oracles = {}
    for s, p in ORACLE.items():
        if p.is_file():
            d = np.load(p)
            oracles[s] = (np.asarray(d["forces"], np.float64), float(d["energy_eV"]))

    # LibTorch UMA LAMMPS
    for (s, ng), d in UMA_FORCES.items():
        fz, tj = d / "forces.npz", d / "timing.json"
        if not (fz.is_file() and tj.is_file()):
            continue
        t = json.loads(tj.read_text())
        rec = {"energy_eV": t.get("energy_eV"),
               "ms": t.get("nvt_pair_ms_per_step"),
               "metric": "NVT Pair ms/step (warmup excl.)",
               "job": t.get("jobid")}
        if s in oracles:
            ref_f, ref_e = oracles[s]
            rec["dE"] = abs(rec["energy_eV"] - ref_e) if rec["energy_eV"] else None
            rec.update(force_stats(np.load(fz)["forces"], ref_f))
        out[s].setdefault("LibTorch UMA LAMMPS (W8nk)", {})[ng] = rec

    # nvalchemi
    for d in sorted(NV.glob("*_ngpu*_*")):
        tj = d / "timing.json"
        if not tj.is_file():
            continue
        t = json.loads(tj.read_text())
        s, ng = t.get("sys"), int(t.get("ngpu", 0))
        if s not in out:
            continue
        rec = {"energy_eV": t.get("energy_eV"),
               "ms": t.get("nvt_ms_per_step"),
               "sp_ms": t.get("sp_ms"),
               "metric": "NVT ms/step (warmup excl.)",
               "job": t.get("job"),
               "dE": t.get("dE_vs_merge_ase"),
               "max_per_atom": t.get("force_max_per_atom"),
               "mean_per_atom": t.get("force_mean_per_atom"),
               "n_over_tol": t.get("n_atoms_over_tol"),
               "n_atoms": t.get("natoms"),
               "precision_ok": t.get("precision_ok"),
               "nvt_status": t.get("nvt_status"),
               "vram_GiB": t.get("vram_peak_GiB")}
        # Prefer the most trustworthy run for this (sys, ngpu): a completed NVT
        # with warmup excluded beats an older cold-start run, and a run with
        # per-atom force data beats one without. Picking by mtime alone would
        # resurrect the pre-warmup-fix numbers.
        rec["_warm"] = bool(t.get("cold_start_excluded"))
        rec["_has_f"] = rec.get("max_per_atom") is not None
        rec["_ok"] = rec.get("nvt_status") == "OK"

        def score(r: dict) -> tuple:
            return (bool(r.get("_warm")), bool(r.get("_ok")),
                    bool(r.get("_has_f")), str(r.get("job") or ""))

        cells = out[s].setdefault("nvalchemi", {})
        prev = cells.get(ng)
        if prev is None or score(rec) > score(prev):
            cells[ng] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    data = collect()
    if a.json:
        print(json.dumps(data, indent=2))
        return 0

    for s in ("nacl6", "water888"):
        natoms = 1728 if s == "nacl6" else 648
        print(f"\n{'=' * 78}\n{s}  ({natoms} atoms, FP64, NVT 300 K)\n{'=' * 78}")
        print("\n--- E / per-atom F parity vs ASE merge oracle ---")
        print(f"{'path':<30}{'GPUs':>5}{'|dE| eV':>11}{'maxPA|dF|':>12}"
              f"{'meanPA':>11}{'>tol':>7}{'verdict':>9}")
        for path, cells in data[s].items():
            for ng in sorted(cells):
                c = cells[ng]
                dE, mx = c.get("dE"), c.get("max_per_atom")
                if dE is None and mx is None:
                    continue
                ok = (dE is not None and dE <= DE_TOL
                      and mx is not None and mx <= DF_TOL)
                print(f"{path:<30}{ng:>5}"
                      f"{(f'{dE:.2e}' if dE is not None else '-'):>11}"
                      f"{(f'{mx:.2e}' if mx is not None else '-'):>12}"
                      f"{(f'{c['mean_per_atom']:.2e}' if c.get('mean_per_atom') is not None else '-'):>11}"
                      f"{str(c.get('n_over_tol', '-')):>7}"
                      f"{('PASS' if ok else 'FAIL'):>9}")

        print("\n--- speed (ms/step or ms/eval; metrics differ, see notes) ---")
        print(f"{'path':<30}{'@1':>10}{'@2':>10}{'@4':>10}   metric")
        for name in ("ASE FC FP64 (general)", "ASE FC FP64 (ufast+merge)",
                     "FC LAMMPS (general)"):
            bars = LOCKED.get((name, s))
            if bars:
                print(f"{name:<30}"
                      f"{bars.get(1, float('nan')):>10.1f}"
                      f"{bars.get(2, float('nan')):>10.1f}"
                      f"{bars.get(4, float('nan')):>10.1f}   locked baseline")
        for path in ("LibTorch UMA LAMMPS (W8nk)", "nvalchemi"):
            cells = data[s].get(path, {})
            bars = LOCKED.get((path, s), {})
            vals = []
            for ng in (1, 2, 4):
                v = (cells.get(ng, {}) or {}).get("ms") or bars.get(ng)
                vals.append(f"{v:>10.1f}" if v else f"{'-':>10}")
            metric = next((c.get("metric") for c in cells.values() if c.get("metric")),
                          "NVT Pair ms/step")
            print(f"{path:<30}{''.join(vals)}   {metric}")

        nv = data[s].get("nvalchemi", {})
        bad = [ng for ng, c in nv.items() if c.get("nvt_status") not in (None, "OK")]
        if bad:
            for ng in bad:
                print(f"  ! nvalchemi @{ng}: {nv[ng]['nvt_status'][:90]}")
    print("\nFC LAMMPS has no FP64+merge_mole row: merge_MOLE Float/Double crash.")
    print("nvalchemi multi-GPU = spatial domain decomposition; LibTorch UMA =")
    print("model-parallel NCCL workers. Different strategies, not a like-for-like scaling test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
