#!/usr/bin/env python3
"""Step 21 (N5.2). Reference harness for one rank's chunk.

Saves NaCl and ZIF-8 fixtures under pbs/out (not committed), then runs the
stock eager chunk against itself and against a no-op fused arm that calls the
same function. The no-op arm must match bitwise and time within the relative
noise of the NaCl step.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ENG = HERE.parents[1] / "src" / "ML-UMA" / "uma-engine" / "python"
sys.path.insert(0, str(ENG))
sys.path.insert(0, str(HERE / "n5"))
sys.path.insert(0, str(HERE))

from profile_n51 import edge_geometry, rank0_chunk  # noqa: E402

FIX = Path(os.environ.get(
    "N5_FIXTURES",
    HERE.parents[1] / "pbs" / "out" / "nitro_n5" / "fixtures",
))
# NaCl step MDE is 6.27 ms / 825 ms.
REL_MDE = 6.27 / 825.0


def sync():
    torch.xpu.synchronize()


def mad(xs, med):
    return statistics.median([abs(x - med) for x in xs])


def time_ms(fn, reps=30):
    for _ in range(2):
        fn()
    sync()
    samples = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    med = statistics.median(samples)
    return med, mad(samples, med)


def zif8_atoms():
    import make_zif8
    from ase import Atoms

    cif = os.environ.get("ZIF8_CIF", make_zif8.CIF_DEFAULT)
    syms, pos, box = make_zif8.read_cif(cif)
    nrep = max(1, int(round(box[0] / make_zif8.CELL_A)))
    syms, pos, cell = make_zif8.conventional_cell(syms, pos, box, nrep)
    n = 3
    rs, rp = [], []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                off = (ix * cell[0], iy * cell[1], iz * cell[2])
                for s, p in zip(syms, pos):
                    rs.append(s)
                    rp.append([p[0] + off[0], p[1] + off[1], p[2] + off[2]])
    import numpy as np
    pos = np.array(rp, float)
    pos = pos + np.random.default_rng(0).normal(0, 0.05, pos.shape)
    atoms = Atoms(symbols=rs, positions=pos, cell=[n * cell[0], n * cell[1], n * cell[2]], pbc=True)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0
    return atoms


def load_backbone():
    os.environ.setdefault("UMA_EXPORT_CELL_LIST", "1")
    os.environ.setdefault("FXPU_WIGNER_PREP_CHUNK", "65536")
    os.environ.setdefault("FXPU_WIGNER_PREP_CHUNK_MODE", "both")
    from common import inference_settings_with_dtype
    from export_blocks_xpu import build_nacl
    from model_loader import load_prepared_hydra_model
    from xpu_prepare_wigner import apply_xpu_prepare_wigner_chunking

    try:
        from fairchem_xpu_device import patch_fairchem_xpu_device
        patch_fairchem_xpu_device()
    except Exception as exc:  # noqa: BLE001
        print(f"WARN patch_fairchem_xpu_device: {exc}", flush=True)
    print(apply_xpu_prepare_wigner_chunking(), flush=True)
    settings = inference_settings_with_dtype("float64")
    settings.external_graph_gen = True
    settings.activation_checkpointing = False
    settings.execution_mode = "general"
    settings.merge_mole = False
    sample_atoms = build_nacl(14)
    from common import atoms_to_atomic_data
    sample = atoms_to_atomic_data(sample_atoms, task_name="omat", settings=settings)
    model, _, _ = load_prepared_hydra_model(
        os.environ["UMA_CHECKPOINT"], sample, settings=settings, device="xpu",
    )
    model = model.to(device="xpu", dtype=torch.float64).eval()
    backbone = model.backbone
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone, settings


def prepare(backbone, atoms, settings):
    from common import atoms_to_atomic_data
    from export_blocks_xpu import _ChunkCore

    dev = torch.device("xpu")
    data = atoms_to_atomic_data(atoms, task_name="omat", settings=settings)
    ei, co, n_local, n_real, n_pad = rank0_chunk(data.edge_index, data.cell_offsets, len(atoms))
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=dev)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float64, device=dev)
    ei = ei.to(dev)
    co = co.to(dev)
    vec, dist = edge_geometry(pos, cell, ei, co)
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=dev)
    batch = torch.zeros(len(atoms), dtype=torch.long, device=dev)
    csd = backbone.csd_embedding(
        charge=data.charge.to(dev), spin=data.spin.to(dev), dataset=["omat"],
    )
    backbone.set_MOLE_coefficients(atomic_numbers, batch, csd)
    backbone.set_MOLE_sizes(int(csd.shape[0]), batch, ei)
    x_full = torch.randn(
        (len(atoms), (int(backbone.lmax) + 1) ** 2, int(backbone.sphere_channels)),
        dtype=torch.float64, device=dev,
    )
    core = _ChunkCore(backbone, backbone.blocks[0], backbone.blocks[0].edge_wise, 0, n_local).to(dev)
    return {
        "core": core,
        "x_full": x_full,
        "vec": vec,
        "dist": dist,
        "atomic_numbers": atomic_numbers,
        "edge_index": ei,
        "n_local": n_local,
        "n_real": n_real,
        "n_pad": n_pad,
    }


def run_core(pack, x, vec):
    return pack["core"](x, vec, pack["dist"], pack["atomic_numbers"], pack["edge_index"])


def stock(pack):
    with torch.no_grad():
        return run_core(pack, pack["x_full"], pack["vec"])


def fused_stock(pack):
    return stock(pack)


def forward_grad(pack):
    x = pack["x_full"].detach().requires_grad_(True)
    v = pack["vec"].detach().requires_grad_(True)
    y = run_core(pack, x, v)
    y.sum().backward()
    return y.detach(), x.grad.detach(), v.grad.detach()


def traffic_bytes(pack, y):
    e = pack["edge_index"].shape[1]
    x = pack["x_full"]
    # Two gathers of x, wigner and envelope read inside the chunk, x_edge, write y.
    # Wigner and x_edge are rebuilt inside the core; count the tensors we can see.
    feat = x.shape[1] * x.shape[2]
    return (2 * e * feat + y.numel() + e * 9 * 9 * 2) * 8


def check_system(name, pack):
    y1 = stock(pack)
    y2 = stock(pack)
    yf = fused_stock(pack)
    # index_add_ is atomic, so stock versus stock is not bitwise. The no-op
    # arm is accepted when it stays inside that repeat difference.
    y_self = (y1 - y2).abs().max().item()
    y_fused = (y1 - yf).abs().max().item()
    g1, gx1, gv1 = forward_grad(pack)
    g2, gx2, gv2 = forward_grad(pack)
    gf, gxf, gvf = forward_grad(pack)
    g_self = max(
        (g1 - g2).abs().max().item(),
        (gx1 - gx2).abs().max().item(),
        (gv1 - gv2).abs().max().item(),
    )
    g_fused = max(
        (g1 - gf).abs().max().item(),
        (gx1 - gxf).abs().max().item(),
        (gv1 - gvf).abs().max().item(),
    )
    match = y_fused <= y_self and g_fused <= g_self
    med_s, mad_s = time_ms(lambda: stock(pack))
    med_f, mad_f = time_ms(lambda: fused_stock(pack))
    ratio = med_f / med_s
    band = max(REL_MDE, 3.0 * mad_s / med_s, 3.0 * mad_f / med_f)
    ratio_ok = abs(ratio - 1.0) <= band
    gbps = (traffic_bytes(pack, y1) / 1e9) / (med_s / 1e3)
    print(
        f"N52_ROW {name} edges={pack['edge_index'].shape[1]} real={pack['n_real']} "
        f"pad={pack['n_pad']} n_local={pack['n_local']} y_self={y_self:.3e} "
        f"y_fused={y_fused:.3e} g_self={g_self:.3e} g_fused={g_fused:.3e} "
        f"stock_ms={med_s:.4f} mad={mad_s:.4f} "
        f"fused_ms={med_f:.4f} mad={mad_f:.4f} ratio={ratio:.6f} band={band:.6f} "
        f"GBs={gbps:.2f}",
        flush=True,
    )
    FIX.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x_full": pack["x_full"].detach().cpu(),
            "edge_index": pack["edge_index"].detach().cpu(),
            "vec": pack["vec"].detach().cpu(),
            "dist": pack["dist"].detach().cpu(),
            "y": y1.detach().cpu(),
            "grad_x": gx1.detach().cpu(),
            "grad_vec": gv1.detach().cpu(),
            "n_local": pack["n_local"],
            "n_real": pack["n_real"],
            "n_pad": pack["n_pad"],
        },
        FIX / f"{name}.pt",
    )
    print(f"N52_SAVED {FIX / (name + '.pt')}", flush=True)
    return match and ratio_ok


def main():
    from export_blocks_xpu import build_nacl

    print("N52_ENTER", flush=True)
    backbone, settings = load_backbone()
    systems = {"nacl": build_nacl(14), "zif8": zif8_atoms()}
    ok = True
    for name, atoms in systems.items():
        pack = prepare(backbone, atoms, settings)
        ok = check_system(name, pack) and ok
    if ok:
        print("N52_PASS", flush=True)
        return
    print("N52_FAIL", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
