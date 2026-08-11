#!/usr/bin/env python3
"""Second-stage diagnosis: is the nvalchemi water error precision or a bug?

Stage 1 established: plain fairchem PASSES water (dE 1.4e-7); nvalchemi's
UMAWrapper on the same box/task/dtype gives dE 6.4e-3 and misses on all 648
atoms. So the error enters in nvalchemi, not our settings.

Error structure (CPU analysis of the saved forces):
  * relative error ~4.35e-4, close to TF32 eps (4.88e-4), far from FP32
    (1.2e-7) and FP64 (2.2e-16)
  * corr(|F_ref|, |dF|) = 0.09 -> NOT proportional round-off
  * species-dependent signed bias (O and H offsets are opposite, ~2:1)
  * net sum dF = 0 exactly

Round-off would scale with magnitude and be unbiased per species. A truncated
or altered neighbor graph would produce exactly this: a physical-looking but
different force field, still momentum-conserving.

This runs the two candidate explanations directly:
  A) TF32 contamination      -> pin fp32/TF32 off and see if the error moves
  B) neighbor-graph mismatch -> compare nvalchemi's neighbor count and cutoff
     against what fairchem's internal builder produces for the same box
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

EX = Path(__file__).resolve().parent
ROOT = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma")
CKPT = Path(os.environ.get(
    "UMA_CHECKPOINT", "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"))
CAMP = ROOT / "src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign"
CASES = {
    "water888": (ROOT / "src/ML-UMA/examples/water888/water_nvt_300K.extxyz",
                 CAMP / "oracle_ase_water_merge.npz", -3143.3893774722696),
    "nacl6": (ROOT / "src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz",
              CAMP / "oracle_ase_umas_fast_merge.npz", -5830.9237413382),
}


def build(dtype="float64"):
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from nvalchemi.models.uma import UMAWrapper
    s = InferenceSettings(tf32=False, activation_checkpointing=False,
                          merge_mole=True, compile=False, external_graph_gen=False)
    s.base_precision_dtype = getattr(torch, dtype)
    w = UMAWrapper.from_checkpoint(str(CKPT), task_name="omat",
                                   device="cuda", inference_settings=s)
    w.model_config.active_outputs = {"energy", "forces"}
    return w


def run(name, tf32_off: bool):
    from ase.io import read
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import DynamicsStage
    from nvalchemi.hooks import HookContext

    xyz, oracle_npz, oracle_E = CASES[name]
    if tf32_off:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    atoms = read(str(xyz))
    model = build()
    data = AtomicData.from_atoms(atoms, dtype=torch.float64)
    batch = Batch.from_data_list([data]).to("cuda")

    hooks = list(model.make_neighbor_hooks())
    ctx = HookContext(batch=batch, model=model, global_rank=0, workflow=None)
    for h in hooks:
        h(ctx, DynamicsStage.BEFORE_COMPUTE)

    # What graph did nvalchemi hand the model?
    ei = getattr(batch, "edge_index", None)
    nl = {
        "nvalchemi_edges": int(ei.shape[1]) if ei is not None and ei.numel() else 0,
        "declared_cutoff": float(getattr(model, "cutoff", -1) or -1),
        "neighbor_config_cutoff": float(
            getattr(model.model_config.neighbor_config, "cutoff", -1)),
        "neighbor_config_max_neighbors": getattr(
            model.model_config.neighbor_config, "max_neighbors", None),
    }

    out = model(batch)
    e = float(out["energy"].reshape(-1)[0].item())
    f = out["forces"].detach().to(torch.float64).cpu().numpy()

    ref = np.load(oracle_npz)["forces"]
    mag = np.linalg.norm(f - ref, axis=1)
    rec = {
        "case": name, "tf32_off": tf32_off,
        "energy_eV": e, "dE_vs_oracle": abs(e - oracle_E),
        "force_max_per_atom": float(mag.max()),
        "force_mean_per_atom": float(mag.mean()),
        "n_over_tol": int((mag > 1e-5).sum()),
        "param_dtype": str(next(p.dtype for p in model.predict_unit.model.parameters()
                                if p.is_floating_point())),
        "forces_dtype": str(out["forces"].dtype),
    }
    rec.update(nl)

    # Reference edge count from fairchem's own builder at the model cutoff.
    try:
        from ase.neighborlist import neighbor_list as ase_nl
        cut = nl["declared_cutoff"] if nl["declared_cutoff"] > 0 else 6.0
        i, _ = ase_nl("ij", atoms, cut)
        rec["ase_edges_at_cutoff"] = int(len(i))
        rec["max_neighbors_observed"] = int(np.bincount(i).max())
    except Exception as exc:  # noqa: BLE001
        rec["ase_nl_error"] = str(exc)[:150]

    del model
    torch.cuda.empty_cache()
    return rec


def main() -> int:
    res = []
    for name in ("water888", "nacl6"):
        for tf32_off in (False, True):
            try:
                r = run(name, tf32_off)
            except Exception as exc:  # noqa: BLE001
                r = {"case": name, "tf32_off": tf32_off,
                     "error": f"{type(exc).__name__}: {exc}"[:300]}
            res.append(r)
            print(json.dumps(r), flush=True)

    (EX / "diagnose_water2.json").write_text(json.dumps(res, indent=2) + "\n")
    print("\n=== VERDICT ===")
    for name in ("water888", "nacl6"):
        rs = [r for r in res if r.get("case") == name and "dE_vs_oracle" in r]
        if len(rs) == 2:
            a, b = rs[0]["dE_vs_oracle"], rs[1]["dE_vs_oracle"]
            print(f"{name}: tf32_on dE={a:.3e}  tf32_off dE={b:.3e}  "
                  f"-> TF32 {'IS the cause' if abs(a - b) > 0.1 * max(a, b) else 'is NOT the cause'}")
        for r in rs:
            if "nvalchemi_edges" in r:
                print(f"   graph: nvalchemi_edges={r['nvalchemi_edges']} "
                      f"ase_edges_at_cutoff={r.get('ase_edges_at_cutoff')} "
                      f"cutoff={r.get('declared_cutoff')} "
                      f"max_neigh_cfg={r.get('neighbor_config_max_neighbors')} "
                      f"max_neigh_obs={r.get('max_neighbors_observed')}")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
