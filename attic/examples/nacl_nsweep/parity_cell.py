#!/usr/bin/env python3
"""ASE FP64 single-GPU ground truth for one N, compared to the 4-GPU LAMMPS run.

Oracle = ASE FairChem FP64, execution_mode=umas_fast_pytorch, merge_mole=True
(the campaign's merge oracle — matching settings, not `general`).

This is single-GPU and WILL OOM at much smaller N than the 4-GPU sweep. That is
expected and recorded as parity="NO_ORACLE_OOM": above that N the capacity
ceiling is still measurable but is no longer parity-verified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

EX = Path(__file__).resolve().parent
ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")
ENGINE = ROOT / "src/ML-UMA/uma-engine"
CKPT = Path(os.environ.get(
    "UMA_CHECKPOINT", "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"))

DE_TOL = 1e-6      # eV, absolute
DF_TOL = 1e-5      # eV/A, max abs component


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--cell-json", required=True,
                    help="cell.json from the 4-GPU run")
    a = ap.parse_args()
    n = a.n
    cell = json.loads(Path(a.cell_json).read_text())
    out = Path(a.cell_json).parent
    rec: dict = {"n": n, "natoms": 8 * n**3, "oracle": "ase_fp64_ufast_merge"}

    xyz = EX / "structures" / f"nacl{n}_rattle.extxyz"
    if not xyz.is_file():
        rec["parity"] = "NO_STRUCTURE"
        (out / "parity.json").write_text(json.dumps(rec, indent=2) + "\n")
        return 0

    try:
        import torch
        from ase.io import read
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit

        sys.path.insert(0, str(ENGINE / "python"))
        from common import inference_settings_with_dtype

        os.environ["FAIRCHEM_WORKERS"] = "1"   # never ParallelMLIPPredictUnit
        settings = inference_settings_with_dtype("float64")
        settings.external_graph_gen = False
        settings.activation_checkpointing = False
        settings.execution_mode = "umas_fast_pytorch"
        settings.merge_mole = True

        atoms = read(str(xyz))
        predictor = load_predict_unit(str(CKPT), device="cuda",
                                      inference_settings=settings, workers=1)
        calc = FAIRChemCalculator(predictor, task_name="omat")
        atoms.calc = calc
        e_ref = float(atoms.get_potential_energy())
        f_ref = np.asarray(atoms.get_forces(), dtype=np.float64)
        torch.cuda.synchronize()
        rec["oracle_E_eV"] = e_ref
        rec["oracle_peak_GiB"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 2)
        np.savez(out / "oracle_forces.npz", forces=f_ref,
                 energy_eV=np.array(e_ref))
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        oom = "out of memory" in msg.lower() or "OutOfMemory" in msg
        rec["parity"] = "NO_ORACLE_OOM" if oom else "NO_ORACLE_ERR"
        rec["error"] = msg[:400]
        rec["traceback_tail"] = traceback.format_exc()[-800:]
        (out / "parity.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps({k: rec[k] for k in ("n", "parity", "error")}, indent=2))
        return 0

    fz = out / "forces.npz"
    if not fz.is_file() or "energy_eV" not in cell:
        rec["parity"] = "NO_UMA_RESULT"
    else:
        d = np.load(fz)
        f_uma = d["forces"]
        e_uma = float(cell["energy_eV"])
        rec["uma_E_eV"] = e_uma
        rec["dE_abs"] = abs(e_uma - e_ref)
        rec["dE_per_atom"] = rec["dE_abs"] / (8 * n**3)
        if f_uma.shape == f_ref.shape:
            diff = np.abs(f_uma - f_ref)
            rec["force_max_abs"] = float(diff.max())
            rec["force_mae"] = float(diff.mean())
            denom = np.linalg.norm(f_uma) * np.linalg.norm(f_ref)
            rec["cosine"] = float((f_uma * f_ref).sum() / denom) if denom else None
            rec["parity"] = ("PASS" if (rec["dE_abs"] <= DE_TOL
                                        and rec["force_max_abs"] <= DF_TOL)
                             else "FAIL")
        else:
            rec["parity"] = "SHAPE_MISMATCH"
    rec["tolerances"] = {"dE_abs": DE_TOL, "force_max_abs": DF_TOL}
    (out / "parity.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
