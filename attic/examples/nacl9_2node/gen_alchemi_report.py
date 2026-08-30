#!/usr/bin/env python3
"""Build the ALCHEMI 4-GPU vs 8-GPU report on NaCl 8x8x8 (4096 atoms).

Energy, per-atom force, and timing, both vs each other and vs the LibTorch
4-GPU ground truth -- but only when ALCHEMI ran the SAME box (UMA_XYZ pointed
at nacl_nsweep nacl8). If the boxes differ, cross-engine energy is NOT reported.

Reads results only; writes ALCHEMI_8x8x8_REPORT.md. Prints "PENDING" for cells
whose jobs have not produced a timing.json yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EX = Path(__file__).resolve().parent
NSWEEP = EX.parent / "nacl_nsweep"
GT_CELL = NSWEEP / "results/n8_21026029"          # LibTorch 4-GPU ground truth
GT_XYZ = NSWEEP / "structures/nacl8_rattle.extxyz"
OUT = EX / "ALCHEMI_8x8x8_REPORT.md"
DF_TOL = 1e-5


def gt() -> dict:
    t = json.loads((GT_CELL / "cell.json").read_text())
    f = np.load(GT_CELL / "forces.npz")["forces"]
    return {"energy_eV": t["energy_eV"], "forces": f,
            "nvt_ms": t.get("nvt_pair_ms_per_step"), "job": "21026029"}


def alchemi_runs() -> list[dict]:
    out = []
    for d in sorted(EX.glob("results/alchemi_*_ngpu*")):
        tj = d / "timing.json"
        if not tj.is_file():
            continue
        t = json.loads(tj.read_text())
        # Only the ground-truth box (nacl8, 4096) is comparable.
        if t.get("sys") != "nacl8" or t.get("natoms") != 4096:
            continue
        rec = {
            "ngpu": t.get("ngpu"), "nnodes": t.get("nnodes"),
            "energy_eV": t.get("energy_eV"),
            "sp_ms": t.get("sp_ms"), "nvt_ms": t.get("nvt_ms_per_step"),
            "vram_GiB": t.get("vram_peak_GiB"),
            "sp_status": t.get("sp_status"), "nvt_status": t.get("nvt_status"),
            "dir": d,
        }
        fz = d / "forces.npz"
        rec["forces"] = np.load(fz)["forces"] if fz.is_file() else None
        out.append(rec)
    return out


def fmt(v, spec=".4f"):
    return "-" if v is None else format(v, spec)


def main() -> int:
    L = ["# ALCHEMI on NaCl 8×8×8 (4096 atoms) — 4 GPU vs 8 GPU",
         "",
         "FP64, `graph_partition`, single point + NVT 300 K. Ground truth is the "
         "LibTorch UMA 4-GPU run (job 21026029) on the **same** box "
         "(`nacl8_rattle`).",
         ""]
    g = gt()
    L += [f"**Ground truth (LibTorch, 4 GPU):** E = `{g['energy_eV']:.9f}` eV, "
          f"NVT `{fmt(g['nvt_ms'], '.1f')}` ms/step, "
          f"forces (4096,3), |F|max `{np.abs(g['forces']).max():.4f}`", ""]

    runs = alchemi_runs()
    if not runs:
        L += ["_No ALCHEMI run on the nacl8 ground-truth box yet "
              "(jobs 21038856 / 21038857 pending)._"]
        OUT.write_text("\n".join(L) + "\n")
        print("PENDING: no comparable ALCHEMI results yet")
        return 0

    # ---- energy ----
    L += ["## Energy", "",
          "| Engine | GPUs | Nodes | Energy (eV) | ΔE vs GT | status |",
          "|---|---:|---:|---:|---:|---|"]
    L.append(f"| LibTorch (GT) | 4 | 1 | {g['energy_eV']:.6f} | — | ref |")
    for r in runs:
        dE = (abs(r["energy_eV"] - g["energy_eV"])
              if r["energy_eV"] is not None else None)
        L.append(f"| ALCHEMI | {r['ngpu']} | {r['nnodes']} | "
                 f"{fmt(r['energy_eV'], '.6f')} | {fmt(dE, '.2e')} | "
                 f"{r['sp_status']} |")

    # ---- per-atom force ----
    L += ["", "## Per-atom force vs ground truth", "",
          "| Engine | GPUs | max|ΔF| | mean|ΔF| | >tol | verdict |",
          "|---|---:|---:|---:|---:|:---:|"]
    for r in runs:
        if r["forces"] is not None and r["forces"].shape == g["forces"].shape:
            mag = np.linalg.norm(r["forces"] - g["forces"], axis=1)
            over = int((mag > DF_TOL).sum())
            L.append(f"| ALCHEMI | {r['ngpu']} | {mag.max():.2e} | "
                     f"{mag.mean():.2e} | {over} | "
                     f"{'PASS' if mag.max() <= DF_TOL else 'FAIL'} |")
        else:
            L.append(f"| ALCHEMI | {r['ngpu']} | - | - | - | no forces |")

    # ---- 4 vs 8 GPU force self-consistency ----
    r4 = next((r for r in runs if r["ngpu"] == 4 and r["forces"] is not None), None)
    r8 = next((r for r in runs if r["ngpu"] == 8 and r["forces"] is not None), None)
    if r4 and r8 and r4["forces"].shape == r8["forces"].shape:
        m = np.linalg.norm(r4["forces"] - r8["forces"], axis=1)
        L += ["", "## ALCHEMI 4-GPU vs 8-GPU self-consistency", "",
              f"max per-atom |ΔF| between rank counts: `{m.max():.2e}` "
              f"(should be ~round-off if decomposition is exact)"]

    # ---- timing ----
    L += ["", "## Timing", "",
          "| Engine | GPUs | Nodes | SP ms | NVT ms/step | VRAM GiB |",
          "|---|---:|---:|---:|---:|---:|"]
    L.append(f"| LibTorch (GT) | 4 | 1 | - | {fmt(g['nvt_ms'], '.1f')} | 24.4 |")
    for r in runs:
        L.append(f"| ALCHEMI | {r['ngpu']} | {r['nnodes']} | "
                 f"{fmt(r['sp_ms'], '.1f')} | {fmt(r['nvt_ms'], '.1f')} | "
                 f"{fmt(r['vram_GiB'], '.1f')} |")

    L += ["", "## Notes",
          "- ALCHEMI multi-GPU is spatial domain decomposition (graph_partition); "
          "LibTorch is edge-parallel MP. Timing is not a like-for-like "
          "implementation comparison.",
          "- Timing metrics differ (ALCHEMI in-code vs LibTorch LAMMPS Pair "
          "timer); compare within an engine, not across."]
    OUT.write_text("\n".join(L) + "\n")
    print(f"[written] {OUT}  ({len(runs)} ALCHEMI cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
