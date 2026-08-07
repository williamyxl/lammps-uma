#!/usr/bin/env python3
"""Record ASE FairChem FP64 workers=1 ground truth once (no ParallelMLIPPredictUnit).

Writes:
  results/gp_round/oracle_ase_fp64_w1.json
  results/gp_round/oracle_ase_fp64_w1.npz

Reuse these artifacts for all later parity gates. Re-run only if geometry or
checkpoint changes (delete the files or pass --force).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

EX = Path(__file__).resolve().parent
ROOT = EX.parents[2]  # .../lammps-uma  (examples -> ML-UMA -> src -> root? )
# EX = .../examples/multi_gpu_nacl6
# parents[0]=examples, [1]=ML-UMA, [2]=src, [3]=lammps-uma
ROOT = EX.parents[3]
sys.path.insert(0, str(EX))

from load_geometry import load_nacl6_fixed  # noqa: E402
from run_multigpu import fp64_settings, teardown_predict_unit  # noqa: E402

OUT_DIR = EX / "results" / "gp_round"
JSON_PATH = OUT_DIR / "oracle_ase_fp64_w1.json"
NPZ_PATH = OUT_DIR / "oracle_ase_fp64_w1.npz"
CKPT = Path(
    os.environ.get(
        "UMA_CHECKPOINT", "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"
    )
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Overwrite existing oracle")
    ap.add_argument("--n-timing", type=int, default=int(os.environ.get("N_TIMING", "3")))
    ap.add_argument(
        "--promote-legacy",
        action="store_true",
        help="Copy from results/ngpu1 ASE FP64 artifacts (no GPU recompute)",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if JSON_PATH.exists() and NPZ_PATH.exists() and not args.force:
        print(f"oracle already exists: {JSON_PATH} (pass --force to recompute)")
        meta = json.loads(JSON_PATH.read_text())
        print(f"  energy_eV={meta.get('energy_eV')}  natoms={meta.get('natoms')}")
        return 0

    if args.promote_legacy:
        return promote_legacy()

    # Hard rule: workers=1 only — never ParallelMLIPPredictUnit
    workers = 1
    os.environ["FAIRCHEM_WORKERS"] = "1"

    atoms = load_nacl6_fixed()
    print(f"geometry natoms={len(atoms)}  ckpt={CKPT}  workers={workers}  FP64")

    import torch
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    settings = fp64_settings(external_graph=True)
    t0 = time.perf_counter()
    predictor = load_predict_unit(
        str(CKPT),
        device="cuda",
        inference_settings=settings,
        workers=workers,
    )
    calc = FAIRChemCalculator(predictor, task_name="omat")
    a = atoms.copy()
    a.calc = calc
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(), dtype=np.float64)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    times = []
    for _ in range(max(1, args.n_timing)):
        if hasattr(a.calc, "results"):
            a.calc.results.clear()
        a.positions = a.positions.copy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e = float(a.get_potential_energy())
        f = np.asarray(a.get_forces(), dtype=np.float64)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t1)

    a.calc = None
    del calc, a
    teardown_predict_unit(predictor, "ASE-FP64-oracle-w1")

    return write_oracle(
        energy_eV=e,
        forces=f,
        atoms=atoms,
        ms_per_eval=float(np.mean(times) * 1e3),
        load_s=load_s,
        source="ase_fairchem_fp64_workers1_live",
        workers=workers,
    )


def promote_legacy() -> int:
    legacy_p = EX / "results" / "ngpu1" / "parity.json"
    legacy_f = EX / "results" / "ngpu1" / "forces.npz"
    if not legacy_p.exists() or not legacy_f.exists():
        print("legacy ngpu1 ASE artifacts missing", file=sys.stderr)
        return 1
    rows = json.loads(legacy_p.read_text()).get("rows") or []
    ase = next((r for r in rows if r.get("key") == "ase"), None)
    if ase is None:
        print("no ASE row in legacy parity.json", file=sys.stderr)
        return 1
    if int(ase.get("workers") or 0) != 1:
        print(f"legacy ASE workers={ase.get('workers')} != 1", file=sys.stderr)
        return 1
    z = np.load(legacy_f)
    atoms = load_nacl6_fixed()
    return write_oracle(
        energy_eV=float(ase["energy_eV"]),
        forces=np.asarray(z["forces_ase"], dtype=np.float64),
        atoms=atoms,
        ms_per_eval=float(ase.get("ms_per_eval") or 0.0),
        load_s=float(ase.get("load_s") or 0.0),
        source="promoted_from_results_ngpu1",
        workers=1,
        job_note=str(ase.get("multi_gpu_note") or ""),
    )


def write_oracle(
    *,
    energy_eV: float,
    forces: np.ndarray,
    atoms,
    ms_per_eval: float,
    load_s: float,
    source: str,
    workers: int,
    job_note: str = "",
) -> int:
    forces = np.asarray(forces, dtype=np.float64)
    assert forces.shape == (len(atoms), 3), forces.shape
    meta = {
        "oracle": "ase_fairchem_fp64_workers1",
        "parallel_predictor": False,
        "workers": workers,
        "precision": "float64",
        "task_name": "omat",
        "checkpoint": str(CKPT),
        "geometry": str(
            EX / "structures" / "nacl6_rattle_fixed.extxyz"
        ),
        "natoms": int(len(atoms)),
        "energy_eV": float(energy_eV),
        "ms_per_eval": float(ms_per_eval),
        "load_s": float(load_s),
        "force_rms": float(np.sqrt(np.mean(forces * forces))),
        "force_max_abs": float(np.max(np.abs(forces))),
        "source": source,
        "job_note": job_note,
        "npz": str(NPZ_PATH.name),
    }
    np.savez_compressed(
        NPZ_PATH,
        energy_eV=np.float64(energy_eV),
        forces=forces,
        numbers=np.asarray(atoms.get_atomic_numbers(), dtype=np.int32),
        positions=np.asarray(atoms.get_positions(), dtype=np.float64),
        cell=np.asarray(atoms.get_cell(), dtype=np.float64),
    )
    JSON_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {NPZ_PATH}")
    print(f"energy_eV={energy_eV:.10f}  ms={ms_per_eval:.2f}  force_max_abs={meta['force_max_abs']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
