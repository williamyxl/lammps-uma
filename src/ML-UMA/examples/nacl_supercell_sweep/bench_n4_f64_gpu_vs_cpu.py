#!/usr/bin/env python3
"""N=4 NaCl FP64 timing: GPU (CUDA) vs CPU for UMA TorchScript + NL.

Matches uma-engine path (external graph, autograd forces, double precision).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
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


def make_nacl(n: int = 4, a: float = 5.64):
    return bulk("NaCl", "rocksalt", a=a, cubic=True) * (n, n, n)


def build_graph(pos, cell, pbc, cutoff, max_neighbors, dtype, device):
    """Vesin on CUDA; vesin CPU tensors on CPU (same API)."""
    from vesin.torch import NeighborList

    pos64 = pos.to(device=device, dtype=torch.float64)
    cell64 = cell.to(device=device, dtype=torch.float64)
    pbc_t = pbc.to(device=device, dtype=torch.bool)
    inv = torch.linalg.inv(cell64)
    frac = pos64 @ inv
    frac = frac - torch.floor(frac) * pbc_t.to(torch.float64).view(1, 3)
    pos_w = (frac @ cell64).contiguous()

    # cell_list on CUDA; auto/brute on CPU is fine
    algo = "cell_list" if device.type == "cuda" else "auto"
    nl = NeighborList(cutoff=cutoff, full_list=True, sorted=False, algorithm=algo)
    i, j, S = nl.compute(pos_w, cell64, pbc_t, quantities="ijS", copy=True)
    i = i.to(torch.int64).reshape(-1)
    j = j.to(torch.int64).reshape(-1)
    S = S.to(torch.int32)
    if max_neighbors > 0 and i.numel() > 0:
        shifts_cart = S.to(dtype) @ cell.to(device=device, dtype=dtype)
        pi = pos_w.to(dtype).index_select(0, i)
        pj = pos_w.to(dtype).index_select(0, j) + shifts_cart
        dist = (pj - pi).norm(dim=1)
        i_cpu = i.detach().cpu().numpy()
        d_cpu = dist.detach().cpu().numpy()
        buckets = defaultdict(list)
        for e, (ic, d) in enumerate(zip(i_cpu, d_cpu)):
            buckets[int(ic)].append((float(d), e))
        keep = []
        for pairs in buckets.values():
            pairs.sort()
            for _, e in pairs[:max_neighbors]:
                keep.append(e)
        keep_t = torch.tensor(sorted(keep), device=device, dtype=torch.int64)
        i = i.index_select(0, keep_t)
        j = j.index_select(0, keep_t)
        S = S.index_select(0, keep_t)
    edge_index = torch.stack([j, i], 0).contiguous()
    cell_offsets = S.to(dtype).contiguous()
    return pos_w.to(dtype), edge_index, cell_offsets


def load_f64(art: Path, device: torch.device):
    meta = json.loads((art / "metadata.json").read_text())
    et = meta["energy_task"]
    dtype = torch.float64
    mod = torch.jit.load(str(art / "model_traced.pt"), map_location=device)
    mod.eval()
    refs = torch.tensor(et["element_references"], dtype=dtype, device=device)
    return mod, {
        "cutoff": float(meta["cutoff"]),
        "max_neighbors": int(meta["max_neighbors"]),
        "mean": float(et["normalizer_mean"]),
        "rmsd": float(et["normalizer_rmsd"]),
    }, dtype, refs


def eval_once(mod, meta, dtype, refs, atoms, device: torch.device):
    pos = torch.tensor(atoms.get_positions(), dtype=dtype, device=device)
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int64, device=device)
    cell = torch.tensor(atoms.cell.array, dtype=dtype, device=device)
    pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=device)
    pos_w, edge, off = build_graph(
        pos, cell, pbc, meta["cutoff"], meta["max_neighbors"], dtype, device
    )
    pos_g = pos_w.detach().clone().requires_grad_(True)
    charge = torch.zeros((), dtype=torch.int64, device=device)
    spin = charge.clone()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.enable_grad():
        e = mod(pos_g, Z, cell, pbc, edge, off, charge, spin).to(dtype).reshape(-1)[0]
        e = e * meta["rmsd"] + meta["mean"] + refs[Z].sum()
        f = -torch.autograd.grad(e, pos_g, create_graph=False)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return float(e.detach().cpu()), float(f.detach().norm().cpu()), dt


def bench_device(label, device, art, atoms, n_warm, n_timed):
    print(f"\n=== {label} ({device}) ===", flush=True)
    mod, meta, dtype, refs = load_f64(art, device)
    times = []
    energy = force_norm = None
    for i in range(n_warm + n_timed):
        e, fn, dt = eval_once(mod, meta, dtype, refs, atoms, device)
        energy, force_norm = e, fn
        if i >= n_warm:
            times.append(dt)
            print(f"  timed[{i - n_warm}] {dt * 1e3:.1f} ms", flush=True)
        else:
            print(f"  warm[{i}] {dt * 1e3:.1f} ms", flush=True)
    arr = np.asarray(times, dtype=np.float64)
    peak = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    return {
        "label": label,
        "device": str(device),
        "energy": energy,
        "force_norm": force_norm,
        "n_warm": n_warm,
        "n_timed": n_timed,
        "times_s": times,
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std()),
        "min_s": float(arr.min()),
        "ms_mean": float(arr.mean() * 1e3),
        "peak_alloc_MiB": peak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--n-warm", type=int, default=2)
    ap.add_argument("--n-timed", type=int, default=5)
    ap.add_argument(
        "--artifact",
        type=Path,
        default=ENGINE / "artifacts" / "uma-s-1p2-omat-f64",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "examples"
        / "nacl_supercell_sweep"
        / "bench_n4_f64_gpu_vs_cpu.json",
    )
    args = ap.parse_args()

    atoms = make_nacl(args.n)
    print(
        f"NaCl {args.n}x{args.n}x{args.n}  natoms={len(atoms)}  "
        f"FP64 artifact={args.artifact}"
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for GPU leg")

    gpu = bench_device(
        "GPU CUDA", torch.device("cuda"), args.artifact, atoms, args.n_warm, args.n_timed
    )
    torch.cuda.empty_cache()
    cpu = bench_device(
        "CPU", torch.device("cpu"), args.artifact, atoms, args.n_warm, args.n_timed
    )

    speedup = cpu["mean_s"] / gpu["mean_s"]
    report = {
        "system": f"NaCl {args.n}x{args.n}x{args.n} rocksalt",
        "natoms": len(atoms),
        "precision": "double / float64",
        "path": "TorchScript UMA + vesin NL + autograd forces",
        "gpu": gpu,
        "cpu": cpu,
        "speedup_gpu_over_cpu": speedup,
        "abs_energy_diff": abs(gpu["energy"] - cpu["energy"]),
        "gpu_name": torch.cuda.get_device_name(0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("\n==== SUMMARY ====")
    print(f"GPU  {gpu['ms_mean']:.1f} ± {gpu['std_s']*1e3:.1f} ms/eval")
    print(f"CPU  {cpu['ms_mean']:.1f} ± {cpu['std_s']*1e3:.1f} ms/eval")
    print(f"GPU speedup vs CPU: {speedup:.2f}x")
    print(f"|E_gpu - E_cpu| = {report['abs_energy_diff']:.3e} eV")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
