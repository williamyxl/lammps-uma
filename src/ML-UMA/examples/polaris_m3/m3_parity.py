"""M3 parity + timing: 2-node/8-GPU vs 1-node/4-GPU ground truth, NaCl 8x8x8.

NaCl 8x8x8 (4096 atoms) OOMs on <4 GPUs, so the ground truth is the 4-GPU/1-node
run (the minimum that fits), NOT a serial run. Scheme A replicated means the
8-GPU/2-node result must be bit-identical in E and per-atom F to the 4-GPU one.
Timing (NVT-300K ms/step) is compared as scaling context.

Reads m3_r4.{json,npz} (ground truth) and m3_r8.{json,npz} (2-node test) from
the results dir; writes M3_REPORT.md / M3_RESULTS.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Scheme A is exact by construction -> require bit-identical E/F.
GATE_DE = 1e-9      # eV, 8-GPU vs 4-GPU
GATE_DF = 1e-9      # eV/A, per-atom max


def load(d: Path, stem: str):
    j = json.loads((d / f"{stem}.json").read_text())
    npz = np.load(d / f"{stem}.npz")
    return j, npz["forces"].astype(np.float64), float(npz["energy_eV"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()
    d = Path(args.results_dir)

    gt = load(d, "m3_gt4")    # 4-GPU/1-node ground truth (minimum that fits)
    test = load(d, "m3_r8")   # 8-GPU/2-node test
    gj, gf, ge = gt
    tj, tf, te = test

    dE = abs(te - ge)
    dF = float(np.linalg.norm(tf - gf, axis=1).max()) if tf.shape == gf.shape else None
    meanF = float(np.linalg.norm(tf - gf, axis=1).mean()) if tf.shape == gf.shape else None
    e_ok = dE <= GATE_DE
    f_ok = dF is not None and dF <= GATE_DF
    gt_units = gj.get("devices", gj.get("ranks"))   # gt4: 4 GPUs (1 node, devices 4)
    test_ranks = tj.get("ranks")                     # r8: 8 MPI ranks (2 nodes)
    ranks_ok = gt_units == 4 and test_ranks == 8

    ms4 = gj.get("nvt_ms_per_step")
    ms8 = tj.get("nvt_ms_per_step")
    speedup = (ms4 / ms8) if (ms4 and ms8) else None

    overall = bool(e_ok and f_ok and ranks_ok)
    rec = {
        "system": "nacl4096",
        "ground_truth": {"ranks": gt_units, "energy_eV": ge,
                         "nvt_ms_per_step": ms4, "force_abs_max": gj.get("force_abs_max")},
        "test_2node": {"ranks": tj.get("ranks"), "energy_eV": te,
                       "nvt_ms_per_step": ms8, "force_abs_max": tj.get("force_abs_max")},
        "abs_dE_8v4": dE, "max_dF_8v4": dF, "mean_dF_8v4": meanF,
        "energy_pass": e_ok, "force_pass": f_ok, "ranks_pass": ranks_ok,
        "nvt_speedup_8v4": speedup,
        "gates": {"dE": GATE_DE, "dF": GATE_DF},
        "M3_PASS": overall,
    }
    (d / "M3_RESULTS.json").write_text(json.dumps(rec, indent=2) + "\n")

    def fmt(x, f="{:.3e}"):
        return "-" if x is None else f.format(x)

    lines = [
        "# M3 - multi-node parity + timing (Scheme A), NaCl 8x8x8 = 4096 atoms",
        "",
        "Ground truth = 4 GPUs / 1 node (minimum that fits; 4096 OOMs on <4 GPUs).",
        "Test = 8 GPUs / 2 nodes. Scheme A replicated -> E/F must be bit-identical;",
        "NVT-300K ms/step compared as scaling context (Scheme A is O(N)/rank, so a",
        "step-time win is NOT expected -- this proves cross-node correctness).",
        "",
        f"Accuracy gates (8-GPU vs 4-GPU): |dE| <= {GATE_DE:g} eV, max|dF| <= {GATE_DF:g} eV/A.",
        "",
        "| config | ranks | E (eV) | NVT ms/step | force |F|max |",
        "|---|---:|---:|---:|---:|",
        f"| ground truth (1 node) | {gt_units} | {ge:.9f} | {fmt(ms4,'{:.2f}')} | {fmt(gj.get('force_abs_max'))} |",
        f"| 2-node test          | {tj.get('ranks')} | {te:.9f} | {fmt(ms8,'{:.2f}')} | {fmt(tj.get('force_abs_max'))} |",
        "",
        "| metric (8-GPU vs 4-GPU) | value | gate | verdict |",
        "|---|---:|---:|:--:|",
        f"| abs |dE| | {fmt(dE)} | {GATE_DE:g} | {'PASS' if e_ok else 'FAIL'} |",
        f"| max |dF| per atom | {fmt(dF)} | {GATE_DF:g} | {'PASS' if f_ok else 'FAIL'} |",
        f"| mean |dF| per atom | {fmt(meanF)} | - | - |",
        f"| rank counts (4 & 8) | {gt_units} & {tj.get('ranks')} | 4 & 8 | {'PASS' if ranks_ok else 'FAIL'} |",
        f"| NVT speedup 8 vs 4 | {fmt(speedup,'{:.2f}')}x | context | - |",
        "",
        f"**M3 {'PASS' if overall else 'FAIL'}**",
        "",
    ]
    (d / "M3_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
