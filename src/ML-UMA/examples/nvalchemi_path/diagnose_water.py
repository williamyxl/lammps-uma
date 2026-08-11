#!/usr/bin/env python3
"""Isolate the ALCHEMI water888 E/F discrepancy.

nvalchemi water888 @1 disagrees with the campaign merge oracle by
dE=6.4e-3 eV / max per-atom dF=1.0e-2 (all 648 atoms over tol), while nacl6
agrees to 1.7e-10 with the same harness. Already ruled out on CPU:
  * structure identical (extxyz vs .data agree to 5e-9 A, same order/species)
  * task=omat on both sides
  * FP64 confirmed on params and outputs (precision_ok=true)
  * both boxes >> 2x cutoff, so not a minimum-image artifact
Remaining difference found by diffing settings: our nvalchemi harness leaves
execution_mode=None; the oracle uses "umas_fast_pytorch".

This runs plain fairchem (NOT nvalchemi) on water and nacl, toggling only
execution_mode, to answer: is the gap caused by that setting, i.e. by us, or is
it inside nvalchemi's own input adaptation?

Reference (campaign merge oracles):
  water888  E = -3143.3893774722696
  nacl6     E = -5830.9237413382
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

EX = Path(__file__).resolve().parent
ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")
ENGINE = ROOT / "src/ML-UMA/uma-engine"
CKPT = Path(os.environ.get(
    "UMA_CHECKPOINT", "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"))
CAMP = ROOT / "src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign"

CASES = {
    "water888": {
        "xyz": ROOT / "src/ML-UMA/examples/water888/water_nvt_300K.extxyz",
        "oracle_E": -3143.3893774722696,
        "oracle_f": CAMP / "oracle_ase_water_merge.npz",
    },
    "nacl6": {
        "xyz": ROOT / "src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz",
        "oracle_E": -5830.9237413382,
        "oracle_f": CAMP / "oracle_ase_umas_fast_merge.npz",
    },
}


def run_fairchem(xyz: Path, execution_mode, task="omat"):
    """Plain fairchem ASE calculator -- no nvalchemi in the loop."""
    import torch
    from ase.io import read
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    sys.path.insert(0, str(ENGINE / "python"))
    from common import inference_settings_with_dtype

    os.environ["FAIRCHEM_WORKERS"] = "1"
    s = inference_settings_with_dtype("float64")
    s.external_graph_gen = False
    s.activation_checkpointing = False
    s.merge_mole = True
    s.execution_mode = execution_mode

    atoms = read(str(xyz))
    pred = load_predict_unit(str(CKPT), device="cuda",
                             inference_settings=s, workers=1)
    atoms.calc = FAIRChemCalculator(pred, task_name=task)
    e = float(atoms.get_potential_energy())
    f = np.asarray(atoms.get_forces(), dtype=np.float64)
    torch.cuda.synchronize()
    del pred, atoms
    torch.cuda.empty_cache()
    return e, f


def main() -> int:
    out = {}
    for name, cfg in CASES.items():
        ref_e = cfg["oracle_E"]
        ref_f = None
        if Path(cfg["oracle_f"]).is_file():
            ref_f = np.load(cfg["oracle_f"])["forces"]
        out[name] = {}
        for mode in ("umas_fast_pytorch", None):
            key = mode or "none"
            try:
                e, f = run_fairchem(cfg["xyz"], mode)
                rec = {"energy_eV": e, "dE_vs_oracle": abs(e - ref_e)}
                if ref_f is not None and ref_f.shape == f.shape:
                    mag = np.linalg.norm(f - ref_f, axis=1)
                    rec["force_max_per_atom"] = float(mag.max())
                    rec["n_over_tol"] = int((mag > 1e-5).sum())
                out[name][key] = rec
            except Exception as exc:  # noqa: BLE001
                out[name][key] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            print(f"{name:>10} execution_mode={key:<20} "
                  f"{json.dumps(out[name][key])}", flush=True)

    (EX / "diagnose_water.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n=== VERDICT ===")
    for name in CASES:
        a = out[name].get("umas_fast_pytorch", {}).get("dE_vs_oracle")
        b = out[name].get("none", {}).get("dE_vs_oracle")
        if a is not None and b is not None:
            print(f"{name}: ufast dE={a:.3e}   none dE={b:.3e}   "
                  f"execution_mode {'MATTERS' if abs(a - b) > 1e-8 else 'is irrelevant'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
