#!/usr/bin/env python3
"""Fast VRAM sweep: TorchScript UMA + vesin CUDA NL (same path as uma-engine).

Finds max NxNxN NaCl that fits on the current GPU via torch.cuda.max_memory_allocated.
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk

_EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXAMPLES))
from _repo import find_uma_engine_root, find_uma_lmp_root  # noqa: E402

ROOT = find_uma_lmp_root()
ENGINE = find_uma_engine_root()


def make_nacl(n: int, a: float = 5.64):
    return bulk("NaCl", "rocksalt", a=a, cubic=True) * (n, n, n)


def build_graph_vesin(pos, cell, pbc, cutoff, max_neighbors, dtype):
    """Mirror uma/vesin_nl.h + FairChem edge flip."""
    from vesin.torch import NeighborList

    # vesin cell_list wants float64
    pos64 = pos.to(torch.float64)
    cell64 = cell.to(torch.float64)
    inv = torch.linalg.inv(cell64)
    frac = pos64 @ inv
    mask = pbc.to(torch.float64).view(1, 3)
    frac = frac - torch.floor(frac) * mask
    pos_w = (frac @ cell64).contiguous()

    nl = NeighborList(cutoff=cutoff, full_list=True, sorted=False, algorithm="cell_list")
    i, j, S = nl.compute(pos_w, cell64, pbc, quantities="ijS", copy=True)
    i = i.to(torch.int64).reshape(-1)
    j = j.to(torch.int64).reshape(-1)
    S = S.to(torch.int32)
    if max_neighbors > 0 and i.numel() > 0:
        shifts_cart = S.to(dtype) @ cell.to(dtype)
        pi = pos_w.to(dtype).index_select(0, i)
        pj = pos_w.to(dtype).index_select(0, j) + shifts_cart
        dist = (pj - pi).norm(dim=1)
        # per-center cap on CPU (small cost vs model)
        i_cpu = i.detach().cpu().numpy()
        d_cpu = dist.detach().cpu().numpy()
        keep = []
        from collections import defaultdict

        buckets = defaultdict(list)
        for e, (ic, d) in enumerate(zip(i_cpu, d_cpu)):
            buckets[int(ic)].append((float(d), e))
        for pairs in buckets.values():
            pairs.sort()
            for _, e in pairs[:max_neighbors]:
                keep.append(e)
        keep = torch.tensor(sorted(keep), device=i.device, dtype=torch.int64)
        i = i.index_select(0, keep)
        j = j.index_select(0, keep)
        S = S.index_select(0, keep)
    # FairChem: row0=neighbor, row1=center
    edge_index = torch.stack([j, i], 0).contiguous()
    cell_offsets = S.to(dtype).contiguous()
    return pos_w.to(dtype), edge_index, cell_offsets


def load_artifact(art: Path, device):
    meta = json.loads((art / "metadata.json").read_text())
    pos_dtype = meta.get("position_dtype", "float32")
    dtype = torch.float64 if pos_dtype == "float64" else torch.float32
    et = meta.get("energy_task") or {}
    meta = {
        **meta,
        "cutoff": float(meta["cutoff"]),
        "max_neighbors": int(meta.get("max_neighbors", 300)),
        "normalizer_mean": float(et.get("normalizer_mean", 0.0)),
        "normalizer_rmsd": float(et.get("normalizer_rmsd", 1.0)),
        "element_references": et.get("element_references"),
    }
    mod = torch.jit.load(str(art / "model_traced.pt"), map_location=device)
    mod.eval()
    refs = None
    if meta["element_references"]:
        refs = torch.tensor(meta["element_references"], dtype=dtype, device=device)
    return mod, meta, dtype, refs


def eval_one(mod, meta, dtype, refs, atoms, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    pos = torch.tensor(atoms.get_positions(), dtype=dtype, device=device)
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int64, device=device)
    cell = torch.tensor(atoms.cell.array, dtype=dtype, device=device)
    pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=device)
    cutoff = float(meta["cutoff"])
    max_neighbors = int(meta["max_neighbors"])

    pos_w, edge_index, cell_offsets = build_graph_vesin(
        pos, cell, pbc, cutoff, max_neighbors, dtype
    )
    pos_grad = pos_w.detach().clone().requires_grad_(True)
    charge = torch.zeros((), dtype=torch.int64, device=device)
    spin = torch.zeros((), dtype=torch.int64, device=device)

    t0 = time.perf_counter()
    with torch.enable_grad():
        normed = mod(pos_grad, Z, cell, pbc, edge_index, cell_offsets, charge, spin)
        energy = normed.to(dtype) * meta["normalizer_rmsd"] + meta["normalizer_mean"]
        energy = energy.reshape(-1)[0]
        if refs is not None:
            energy = energy + refs[Z].sum()
        forces = -torch.autograd.grad(energy, pos_grad, create_graph=False)[0]
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
    return {
        "ok": True,
        "energy": float(energy.detach().cpu()),
        "force_norm": float(forces.detach().norm().cpu()),
        "wall_s": wall,
        "peak_alloc_MiB": peak,
        "peak_reserved_MiB": reserved,
        "n_edges": int(edge_index.size(1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=6)
    ap.add_argument("--n-max", type=int, default=12)
    ap.add_argument(
        "--artifact",
        type=Path,
        default=ENGINE / "artifacts" / "uma-s-1p2-omat",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "lammps" / "src" / "ML-UMA" / "examples" / "nacl_supercell_sweep" / "sweep_vram_torch.json",
    )
    args = ap.parse_args()

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    total_mib = props.total_memory / (1024**2)
    print(f"GPU: {torch.cuda.get_device_name(0)}  total={total_mib:.0f} MiB")
    print(f"artifact={args.artifact}")

    mod, meta, dtype, refs = load_artifact(args.artifact, device)
    # warm-up tiny
    warm = make_nacl(2)
    eval_one(mod, meta, dtype, refs, warm, device)
    torch.cuda.empty_cache()

    results = []
    max_fit = None
    for n in range(args.n_min, args.n_max + 1):
        atoms = make_nacl(n)
        natoms = len(atoms)
        print(f"\n=== N={n} natoms={natoms} ===", flush=True)
        try:
            r = eval_one(mod, meta, dtype, refs, atoms, device)
            r.update({"N": n, "natoms": natoms, "oom": False})
            print(
                f"  OK E={r['energy']:.4f}  wall={r['wall_s']:.2f}s  "
                f"peak_alloc={r['peak_alloc_MiB']:.0f} MiB  "
                f"reserved={r['peak_reserved_MiB']:.0f} MiB",
                flush=True,
            )
            max_fit = n
        except RuntimeError as e:
            msg = str(e)
            oom = "out of memory" in msg.lower() or "cuda" in msg.lower()
            print(f"  FAIL{' OOM' if oom else ''}: {msg[:200]}", flush=True)
            r = {
                "N": n,
                "natoms": natoms,
                "ok": False,
                "oom": oom,
                "error": msg[:500],
            }
            torch.cuda.empty_cache()
        results.append(r)

    report = {
        "method": "TorchScript artifact + vesin.torch + autograd (engine-equivalent)",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_MiB": total_mib,
        "precision": str(dtype),
        "max_N_fit": max_fit,
        "max_natoms_fit": None if max_fit is None else 8 * max_fit**3,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("\n==== SUMMARY ====")
    print(f"max N fit: {max_fit}  natoms={report['max_natoms_fit']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
