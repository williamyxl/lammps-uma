#!/usr/bin/env python3
"""Step 20 (N5.1). One tile. Stock eager forward_chunk, FP64.

Uses one rank of a NaCl 14^3 W=12 edge shard, padded to the 65536-edge chunk.
Times the three checkpoint passes on each of the four layers and groups the
work into the six blocks in NITRO_PLAN.md section 5.1. Writes N5_PROFILE.md.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import torch

ENG = Path(__file__).resolve().parents[2] / "src" / "ML-UMA" / "uma-engine" / "python"
sys.path.insert(0, str(ENG))

OUT = Path(os.environ.get(
    "N5_PROFILE",
    Path(__file__).resolve().parents[2] / "pbs" / "out" / "nitro_n5" / "N5_PROFILE.md",
))
CHUNK = 65536
WORLD = 12
TARGET_MS = 659.0


def sync():
    torch.xpu.synchronize()


def median(xs):
    return statistics.median(xs)


def time_ms(fn, reps=3):
    fn()
    sync()
    samples = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return median(samples), samples


def rank0_chunk(edge_index, cell_offsets, n_atoms):
    """Edges whose center is owned by rank 0, padded to one chunk."""
    n_local = (n_atoms + WORLD - 1) // WORLD
    center = edge_index[1]
    keep = center < n_local
    ei = edge_index[:, keep]
    co = cell_offsets[keep]
    order = torch.argsort(ei[1], stable=True)
    ei = ei[:, order]
    co = co[order]
    n_real = int(ei.shape[1])
    n_pad = CHUNK - n_real
    if n_pad < 0:
        ei = ei[:, :CHUNK]
        co = co[:CHUNK]
        n_real = CHUNK
        n_pad = 0
    elif n_pad > 0:
        pad_e = torch.zeros((2, n_pad), dtype=ei.dtype)
        pad_c = torch.zeros((n_pad, 3), dtype=co.dtype)
        pad_c[:, 0] = 2.0
        ei = torch.cat([pad_e, ei], dim=1)
        co = torch.cat([pad_c, co], dim=0)
    return ei, co, n_local, n_real, n_pad


def edge_geometry(pos, cell, edge_index, cell_offsets):
    nbr = edge_index[0]
    ctr = edge_index[1]
    shift = cell_offsets.to(pos.dtype) @ cell
    vec = pos[nbr] + shift - pos[ctr]
    dist = torch.linalg.vector_norm(vec, dim=-1)
    return vec, dist


def instrumented(edgewise, x_full, natoms, x_edge, edge_index, wigner, wigner_inv, node_offset):
    """Same math as Edgewise.forward_chunk, split at the section 5.1 boundaries."""
    from fairchem.core.models.uma.escn_md_block import set_mole_ac_start_index

    set_mole_ac_start_index(edgewise, 0)
    with torch.profiler.record_function("blk_gather"):
        x_message = edgewise.backend.node_to_edge_wigner_permute(x_full, edge_index, wigner)
    conv1 = edgewise.so2_conv_1
    saved = conv1.rad_func
    with torch.profiler.record_function("blk_radial"):
        radial = saved(x_edge) if saved is not None else x_edge
    conv1.rad_func = None
    try:
        with torch.profiler.record_function("blk_conv1"):
            x_message, gating = conv1(x_message, radial)
    finally:
        conv1.rad_func = saved
    with torch.profiler.record_function("blk_gate"):
        x_message = edgewise.act(gating, x_message)
    with torch.profiler.record_function("blk_conv2"):
        x_message = edgewise.so2_conv_2(x_message)
    with torch.profiler.record_function("blk_scatter"):
        out = edgewise.backend.permute_wigner_inv_edge_to_node(
            x_message, wigner_inv, edge_index, natoms, node_offset,
        )
    set_mole_ac_start_index(edgewise, 0)
    return out


def main():
    os.environ.setdefault("UMA_EXPORT_CELL_LIST", "1")
    os.environ.setdefault("FXPU_WIGNER_PREP_CHUNK", "65536")
    os.environ.setdefault("FXPU_WIGNER_PREP_CHUNK_MODE", "both")
    ckpt = os.environ["UMA_CHECKPOINT"]
    dev = torch.device("xpu")
    print("N51_ENTER", flush=True)

    from common import atoms_to_atomic_data, inference_settings_with_dtype
    from export_blocks_xpu import _ChunkCore, build_nacl
    from model_loader import load_prepared_hydra_model
    from xpu_prepare_wigner import apply_xpu_prepare_wigner_chunking

    try:
        from fairchem_xpu_device import patch_fairchem_xpu_device
        patch_fairchem_xpu_device()
    except Exception as exc:  # noqa: BLE001
        print(f"WARN patch_fairchem_xpu_device: {exc}", flush=True)
    print(apply_xpu_prepare_wigner_chunking(), flush=True)

    atoms = build_nacl(14)
    settings = inference_settings_with_dtype("float64")
    settings.external_graph_gen = True
    settings.activation_checkpointing = False
    settings.execution_mode = "general"
    settings.merge_mole = False
    sample = atoms_to_atomic_data(atoms, task_name="omat", settings=settings)
    model, _, _ = load_prepared_hydra_model(ckpt, sample, settings=settings, device="xpu")
    model = model.to(device=dev, dtype=torch.float64).eval()
    backbone = model.module.backbone if hasattr(model, "module") else model.backbone
    n_params = sum(p.numel() for p in backbone.parameters())
    n_grad = sum(p.requires_grad for p in backbone.parameters())
    for p in backbone.parameters():
        p.requires_grad_(False)

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
        charge=data.charge.to(dev),
        spin=data.spin.to(dev),
        dataset=data.dataset.to(dev) if torch.is_tensor(getattr(data, "dataset", None)) else None,
    )
    backbone.set_MOLE_coefficients(atomic_numbers, batch, csd)
    backbone.set_MOLE_sizes(int(csd.shape[0]), batch, ei)
    print(
        "N51_MOLE",
        tuple(backbone.global_mole_tensors.expert_mixing_coefficients.shape),
        backbone.global_mole_tensors.mole_sizes,
        flush=True,
    )
    x_full = torch.randn(
        (len(atoms), (int(backbone.lmax) + 1) ** 2, int(backbone.sphere_channels)),
        dtype=torch.float64, device=dev,
    )

    core0 = _ChunkCore(backbone, backbone.blocks[0], backbone.blocks[0].edge_wise, 0, n_local)
    with torch.no_grad():
        wigner, wigner_inv, x_edge = core0._recompute_chunk_edge_tensors(
            vec, dist, atomic_numbers, ei,
        )
    shapes = {
        "atoms": len(atoms),
        "n_local": n_local,
        "edges_real": n_real,
        "edges_pad": n_pad,
        "edge_index": list(ei.shape),
        "edge_distance_vec": list(vec.shape),
        "x_full": list(x_full.shape),
        "wigner": list(wigner.shape),
        "wigner_inv_envelope": list(wigner_inv.shape),
        "x_edge": list(x_edge.shape),
        "lmax": int(backbone.lmax),
        "mmax": int(backbone.mmax),
        "sphere_channels": int(backbone.sphere_channels),
        "num_layers": int(backbone.num_layers),
        "params": n_params,
        "params_requiring_grad_at_load": n_grad,
    }
    print("N51_SHAPES", shapes, flush=True)

    names = ["wigner", "gather", "radial", "conv1", "gate", "conv2", "scatter"]
    share = {k: {"nograd": [], "fwd": [], "bwd": []} for k in names}
    whole = {"nograd": [], "fwd": [], "bwd": []}

    def pieces(core, x, v, d, z, e):
        with torch.profiler.record_function("blk_wigner"):
            wig, inv, xe = core._recompute_chunk_edge_tensors(v, d, z, e)
        return instrumented(
            core.edge_wise, x, core.n_local, xe, e, wig, inv, core.node_offset,
        )

    for layer, block in enumerate(backbone.blocks):
        core = _ChunkCore(backbone, block, block.edge_wise, 0, n_local).to(dev)
        x = x_full.detach()
        v = vec.detach()
        d = dist.detach()

        def nograd(core=core, x=x, v=v, d=d):
            with torch.no_grad():
                return core(x, v, d, atomic_numbers, ei)

        def fwd(core=core, x=x, v=v, d=d):
            xv = x.detach().requires_grad_(True)
            vv = v.detach().requires_grad_(True)
            return core(xv, vv, d, atomic_numbers, ei)

        def bwd(core=core, x=x, v=v, d=d):
            xv = x.detach().requires_grad_(True)
            vv = v.detach().requires_grad_(True)
            y = core(xv, vv, d, atomic_numbers, ei)
            y.sum().backward()

        for key, fn in (("nograd", nograd), ("fwd", fwd), ("bwd", bwd)):
            ms, _ = time_ms(fn, reps=3)
            whole[key].append(ms)
            print(f"N51_WHOLE layer={layer} {key} {ms:.3f}", flush=True)

        def split_nograd(core=core, x=x, v=v, d=d):
            with torch.no_grad():
                pieces(core, x, v, d, atomic_numbers, ei)

        def split_fwd(core=core, x=x, v=v, d=d):
            pieces(core, x.detach().requires_grad_(True), v.detach().requires_grad_(True),
                   d, atomic_numbers, ei)

        def split_bwd(core=core, x=x, v=v, d=d):
            y = pieces(core, x.detach().requires_grad_(True), v.detach().requires_grad_(True),
                       d, atomic_numbers, ei)
            y.sum().backward()

        # One synchronized pass per block name would miss the split. Time each
        # record_function by running the split forward under the profiler once
        # per pass and reading self device time. The whole-chunk times above
        # are the calibration.
        for key, fn in (("nograd", split_nograd), ("fwd", split_fwd), ("bwd", split_bwd)):
            activities = [torch.profiler.ProfilerActivity.CPU]
            if hasattr(torch.profiler.ProfilerActivity, "XPU"):
                activities.append(torch.profiler.ProfilerActivity.XPU)
            fn()
            sync()
            with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                fn()
                sync()
            buckets = {n: 0.0 for n in names}
            for evt in prof.key_averages():
                label = None
                for n in names:
                    if f"blk_{n}" in evt.key:
                        label = n
                        break
                if label is None:
                    continue
                us = 0.0
                for attr in ("xpu_time_total", "device_time_total", "cpu_time_total"):
                    if hasattr(evt, attr):
                        us = float(getattr(evt, attr))
                        break
                buckets[label] += us / 1000.0
            for n in names:
                share[n][key].append(buckets[n])
            print(f"N51_SHARE layer={layer} {key} " +
                  " ".join(f"{n}={buckets[n]:.3f}" for n in names), flush=True)

    # share[block][key] is a list with one entry per layer. Sum over layers.
    lines = []
    lines.append("# N5.1 op profile")
    lines.append("")
    lines.append(f"Job host measured on one XPU tile. Target compute row: {TARGET_MS:.0f} ms.")
    lines.append("")
    lines.append("## Shapes")
    lines.append("")
    for k, v in shapes.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Weights were loaded with "
                 f"{n_grad} of {sum(1 for _ in backbone.parameters())} parameter tensors "
                 "requiring grad. All parameter gradients were then turned off, matching "
                 "inference. The backward timed here is only through activations.")
    lines.append("")
    lines.append("## Whole chunk (stock core: wigner recompute + forward_chunk)")
    lines.append("")
    lines.append("| Layer | no-grad ms | grad forward ms | backward ms |")
    lines.append("|---:|---:|---:|---:|")
    for i in range(len(whole["nograd"])):
        lines.append(f"| {i} | {whole['nograd'][i]:.2f} | {whole['fwd'][i]:.2f} | {whole['bwd'][i]:.2f} |")
    total = sum(whole["nograd"]) + sum(whole["fwd"]) + sum(whole["bwd"])
    lines.append("")
    lines.append(f"Four layers, three passes, one chunk: **{total:.1f} ms**.")
    gap = total - TARGET_MS
    pct = 100.0 * total / TARGET_MS
    lines.append(f"Against the 659 ms compute row: {pct:.1f}% ({gap:+.1f} ms).")
    lines.append("")
    if abs(pct - 100.0) > 15.0:
        lines.append("The gap is outside ±15%. This profile is eager `forward_chunk` plus the "
                     "per-chunk wigner recompute only. The 659 ms compute row also includes the "
                     "edge-degree prologue, the atom-wise block, the two layer norms, "
                     "balance, and the runtime padding and copies around the traced module. "
                     "Those are not inside `forward_chunk`. The traced engine also fuses the "
                     "checkpoint wrapper differently from this eager call.")
    else:
        lines.append("The sum is within ±15% of the 659 ms compute row.")
    lines.append("")
    lines.append("## Block shares")
    lines.append("")
    lines.append("Sum over the four layers of one profiler pass per layer. Inclusive time of "
                 "each record_function, device time when the profiler reports it, otherwise "
                 "CPU time. Nothing is dropped.")
    lines.append("")
    rows = []
    for n in names:
        ms = (sum(share[n]["nograd"]) + sum(share[n]["fwd"]) + sum(share[n]["bwd"]))
        rows.append((n, ms, sum(share[n]["nograd"]), sum(share[n]["fwd"]), sum(share[n]["bwd"])))
    rows.sort(key=lambda r: r[1], reverse=True)
    lines.append("| Block | no-grad | grad forward | backward | three passes |")
    lines.append("|---|---:|---:|---:|---:|")
    gemm_names = {"conv1", "conv2"}
    gemm = 0.0
    for n, ms, a, b, c in rows:
        lines.append(f"| {n} | {a:.2f} | {b:.2f} | {c:.2f} | {ms:.2f} |")
        if n in gemm_names:
            gemm += ms
    lines.append("")
    lines.append("Ordered by share (priority only, nothing removed): "
                 + ", ".join(n for n, *_ in rows) + ".")
    lines.append("")
    lines.append(f"Measured GEMM-side floor (conv1 + conv2, all three passes, four layers): "
                 f"**{gemm:.1f} ms**.")
    lines.append("")
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print("N51_TOTAL", f"{total:.1f}", flush=True)
    print("N51_WROTE", OUT, flush=True)
    print("N51_DONE", flush=True)


if __name__ == "__main__":
    main()
