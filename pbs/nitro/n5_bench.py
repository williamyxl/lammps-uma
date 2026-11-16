#!/usr/bin/env python3
"""Step 21 (N5.2). Reference harness for one rank's chunk.

Saves NaCl and ZIF-8 fixtures under pbs/out (not committed), then runs the
stock eager chunk against itself and against a no-op fused arm that calls the
same function. index_add_ is atomic, so stock is not bitwise against itself
(N5.d). The no-op arm passes when its difference stays within the larger of
the 1e-12 relative bar and twice stock's own repeat, and the interleaved
timing ratio stays within that system's relative MDE.
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
# Step MDEs, as a fraction of the step median. NaCl 6.27/825, ZIF-8 3.11/876.1.
REL_MDE = {"nacl": 6.27 / 825.0, "zif8": 3.11 / 876.1}


def sync():
    torch.xpu.synchronize()


def mad(xs, med):
    return statistics.median([abs(x - med) for x in xs])


def time_pair(fn_a, fn_b, reps=30):
    for _ in range(2):
        fn_a()
        fn_b()
    sync()
    left, right = [], []
    for i in range(reps):
        order = ((fn_a, left), (fn_b, right))
        if i % 2:
            order = ((fn_b, right), (fn_a, left))
        for fn, bucket in order:
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            bucket.append((time.perf_counter() - t0) * 1e3)
    med_a = statistics.median(left)
    med_b = statistics.median(right)
    return med_a, mad(left, med_a), med_b, mad(right, med_b)


def max_abs(a, b):
    return (a - b).abs().max().item()


def rel_of(diff, ref):
    return diff / max(ref.abs().max().item(), 1e-30)


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


def fused_grad(pack):
    return forward_grad(pack)


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


def repeat_floor(draw, n=8):
    """Max abs and relative difference over n draws of (left, right, ref)."""
    abs_max = 0.0
    rel_max = 0.0
    for _ in range(n):
        left, right, ref = draw()
        diff = max_abs(left, right)
        abs_max = max(abs_max, diff)
        rel_max = max(rel_max, rel_of(diff, ref))
    return abs_max, rel_max


def within_repeat(abs_fused, rel_fused, abs_self, rel_self):
    # 1e-12 is the N5 block bar. Twice the stock repeat covers one atomic
    # sample landing slightly above another of the same sum.
    return rel_fused <= max(1e-12, rel_self) or abs_fused <= 2.0 * abs_self


def grad_pair(pack, other):
    left = forward_grad(pack)
    right = other()
    diff = max(max_abs(left[i], right[i]) for i in range(3))
    ref = torch.cat([left[i].reshape(-1) for i in range(3)])
    return diff, rel_of(diff, ref), left


def check_system(name, pack):
    def y_self_draw():
        left = stock(pack)
        return left, stock(pack), left

    def y_fused_draw():
        left = stock(pack)
        return left, fused_stock(pack), left

    y_self, y_self_rel = repeat_floor(y_self_draw)
    y_fused, y_fused_rel = repeat_floor(y_fused_draw)
    g_self, g_self_rel, saved = 0.0, 0.0, None
    for _ in range(8):
        diff, rel, saved = grad_pair(pack, lambda: forward_grad(pack))
        g_self, g_self_rel = max(g_self, diff), max(g_self_rel, rel)
    g_fused, g_fused_rel = 0.0, 0.0
    for _ in range(8):
        diff, rel, _ = grad_pair(pack, lambda: fused_grad(pack))
        g_fused, g_fused_rel = max(g_fused, diff), max(g_fused_rel, rel)
    match = within_repeat(y_fused, y_fused_rel, y_self, y_self_rel) and within_repeat(
        g_fused, g_fused_rel, g_self, g_self_rel
    )
    med_s, mad_s, med_f, mad_f = time_pair(lambda: stock(pack), lambda: fused_stock(pack))
    ratio = med_f / med_s
    rel = REL_MDE[name]
    band = max(rel, 3.0 * mad_s / med_s, 3.0 * mad_f / med_f)
    ratio_ok = abs(ratio - 1.0) <= band
    y_save = stock(pack)
    gbps = (traffic_bytes(pack, y_save) / 1e9) / (med_s / 1e3)
    print(
        f"N52_ROW {name} edges={pack['edge_index'].shape[1]} real={pack['n_real']} "
        f"pad={pack['n_pad']} n_local={pack['n_local']} y_self={y_self:.3e} "
        f"y_fused={y_fused:.3e} y_rel={y_fused_rel:.3e} g_self={g_self:.3e} "
        f"g_fused={g_fused:.3e} g_rel={g_fused_rel:.3e} "
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
            "y": y_save.detach().cpu(),
            "grad_x": saved[1].detach().cpu(),
            "grad_vec": saved[2].detach().cpu(),
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
