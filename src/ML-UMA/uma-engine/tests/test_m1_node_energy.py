#!/usr/bin/env python3
"""M1 gate: per-atom energies sum to the total, and forces are unchanged.

Plan gate (multinode_mpi_plan.md sec 9, M1):
  |sum node_e - E_total| <= 1e-12 relative, on NaCl6 and water888.
  Serial forces unchanged.

Both properties are prerequisites for any spatial decomposition:
  * E = sum_ranks sum_{i in local} e_i is only exact if the identity holds.
  * forces come from grad(E, pos), so the hook must not break the graph.

Run on a GPU node:
  python test_m1_node_energy.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ENGINE = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/uma-engine")
EXAMPLES = Path("/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples")
CKPT = Path(os.environ.get(
    "UMA_CHECKPOINT", "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"))
sys.path.insert(0, str(ENGINE / "python"))

CASES = {
    "nacl6": EXAMPLES / "multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz",
    "water888": EXAMPLES / "water888/water_nvt_300K.extxyz",
}
REL_TOL = 1e-12
FORCE_TOL = 1e-10


def build(dataset="omat"):
    from common import inference_settings_with_dtype
    from fairchem.core.units.mlip_unit import load_predict_unit

    s = inference_settings_with_dtype("float64")
    s.external_graph_gen = False
    s.activation_checkpointing = False
    s.execution_mode = "umas_fast_pytorch"
    s.merge_mole = True
    os.environ["FAIRCHEM_WORKERS"] = "1"
    return load_predict_unit(str(CKPT), device="cuda",
                             inference_settings=s, workers=1)


def run_case(name: str, xyz: Path) -> dict:
    from ase.io import read
    from export_wrapper import (
        make_node_energy_export_wrapper,
        make_traced_export_wrapper,
    )
    from fairchem.core.datasets.atomic_data import AtomicData

    atoms = read(str(xyz))
    pred = build()

    # prepare_for_inference installs the umas_fast_pytorch conversions
    # (SO2 conv rewrite + model._unified_radial_mlp). Reaching into
    # pred.model.module before that runs yields a model whose forward raises
    # AttributeError: 'eSCNMDMoeBackbone' object has no attribute
    # '_unified_radial_mlp'. Drive one real predict first so the model is in the
    # same prepared state the export path uses.
    a0 = AtomicData.from_ase(atoms, task_name="omat").to("cuda")
    pred.predict(a0)
    model = pred.model.module if hasattr(pred.model, "module") else pred.model

    base = make_traced_export_wrapper(model, "omat").to("cuda").eval()
    node = make_node_energy_export_wrapper(model, "omat").to("cuda").eval()

    a = AtomicData.from_ase(atoms, task_name="omat")
    a = a.to("cuda")
    args = base.example_inputs_from_data(a)
    args = tuple(x.to("cuda") for x in args)

    pos = args[0].clone().to(torch.float64).requires_grad_(True)
    rest = args[1:]

    e_base = base(pos, *rest)
    f_base = torch.autograd.grad(e_base.sum(), pos, retain_graph=False)[0]

    pos2 = args[0].clone().to(torch.float64).requires_grad_(True)
    node_e, e_node = node(pos2, *rest)
    f_node = torch.autograd.grad(e_node.sum(), pos2, retain_graph=False)[0]

    tot = float(e_node.reshape(-1)[0].item())
    s_node = float(node_e.sum().item())
    rel = abs(s_node - tot) / max(abs(tot), 1e-30)
    dF = float((f_node - f_base).abs().max().item())

    rec = {
        "case": name,
        "natoms": int(len(atoms)),
        "node_energy_shape": list(node_e.shape),
        "E_total": tot,
        "sum_node_e": s_node,
        "abs_diff": abs(s_node - tot),
        "rel_diff": rel,
        "sum_identity_pass": rel <= REL_TOL,
        "max_dF_vs_energy_only": dF,
        "forces_unchanged_pass": dF <= FORCE_TOL,
        "node_energy_requires_grad": bool(node_e.requires_grad),
    }
    rec["pass"] = bool(rec["sum_identity_pass"] and rec["forces_unchanged_pass"]
                       and rec["node_energy_requires_grad"]
                       and rec["node_energy_shape"] == [len(atoms)])
    del pred, model, base, node
    torch.cuda.empty_cache()
    return rec


def main() -> int:
    out = []
    for name, xyz in CASES.items():
        if not xyz.is_file():
            out.append({"case": name, "error": f"missing {xyz}"})
            continue
        try:
            r = run_case(name, xyz)
        except Exception as exc:  # noqa: BLE001
            r = {"case": name, "error": f"{type(exc).__name__}: {exc}"[:400]}
        out.append(r)
        print(json.dumps(r, indent=2), flush=True)

    Path(ENGINE / "tests/m1_node_energy_result.json").write_text(
        json.dumps(out, indent=2) + "\n")
    ok = all(r.get("pass") for r in out)
    print(f"\nM1 GATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
