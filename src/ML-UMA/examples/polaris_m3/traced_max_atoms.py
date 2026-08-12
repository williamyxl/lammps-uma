#!/usr/bin/env python3
"""Max-atoms sweep using the TRACED artifact (model_traced.pt) exactly as the
LAMMPS engine drives it: traced module forward + autograd force. Tells us whether
activation checkpointing baked into the export actually lowers memory on the
product path (vs the FairChem-model sweep which uses eager checkpointing)."""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk

ENG = Path(__file__).resolve().parents[1].parent / "uma-engine"
sys.path.insert(0, str(ENG / "python"))
from common import atoms_to_atomic_data, inference_settings_with_dtype  # noqa: E402
from metadata import ExportMetadata, denorm_from_metadata, undo_refs_from_metadata  # noqa: E402

TASK = "omat"


def try_size(module, meta, dev, nrep, settings) -> dict:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((nrep, nrep, nrep))
    atoms.pbc = True
    n = len(atoms)
    torch.cuda.empty_cache()
    _stats_ok = True
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        _stats_ok = False  # pluggable (managed) allocator has no peak stats
    rec = {"nrep": nrep, "natoms": n, "ok": False, "peak_gib": None}
    try:
        data = atoms_to_atomic_data(atoms, TASK, settings).to(dev)
        pos = data.pos.detach().to(torch.float64).requires_grad_(True)
        cell = data.cell.squeeze(0).to(dev, torch.float64)
        co = data.cell_offsets.to(dev, torch.float64)
        ch = data.charge.to(dev); sp = data.spin.to(dev)
        if ch.numel() == 1 and ch.dim() == 1: ch = ch.squeeze(0)
        if sp.numel() == 1 and sp.dim() == 1: sp = sp.squeeze(0)
        normed = module(pos, data.atomic_numbers.to(dev), cell, data.pbc.squeeze(0).to(dev),
                        data.edge_index.to(dev), co, ch, sp)
        e = denorm_from_metadata(normed.to(torch.float64), meta)
        batch = torch.zeros(n, dtype=torch.long, device=dev)
        e = undo_refs_from_metadata(meta, data.atomic_numbers.to(dev), batch, e)
        (g,) = torch.autograd.grad(e.sum(), pos)
        torch.cuda.synchronize()
        # timed second pass (proxy for MD step cost; matters for A1 thrash)
        t0 = time.perf_counter()
        pos2 = data.pos.detach().to(torch.float64).requires_grad_(True)
        normed = module(pos2, data.atomic_numbers.to(dev), cell, data.pbc.squeeze(0).to(dev),
                        data.edge_index.to(dev), co, ch, sp)
        e2 = denorm_from_metadata(normed.to(torch.float64), meta)
        e2 = undo_refs_from_metadata(meta, data.atomic_numbers.to(dev), batch, e2)
        (g2,) = torch.autograd.grad(e2.sum(), pos2)
        torch.cuda.synchronize()
        rec["ms"] = (time.perf_counter() - t0) * 1e3
        rec["ok"] = True
        rec["peak_gib"] = (torch.cuda.max_memory_allocated()/1024**3) if _stats_ok else None
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            rec["dev_used_gib"] = (total_b - free_b) / 1024**3
        except Exception:
            pass
    except RuntimeError as ex:
        rec["err"] = "OOM" if "out of memory" in str(ex).lower() else str(ex)[:150]
        rec["peak_gib"] = (torch.cuda.max_memory_allocated()/1024**3) if _stats_ok else None
    finally:
        gc.collect(); torch.cuda.empty_cache()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--sizes", type=int, nargs="*", default=[6, 8, 10, 12, 14, 16])
    ap.add_argument("--managed-alloc", default=None,
                    help="path to libmanaged_alloc.so for CUDA managed-memory "
                         "oversubscription (A1); spills allocations to host RAM")
    args = ap.parse_args()

    # A1: plug a cudaMallocManaged allocator BEFORE any CUDA allocation.
    if args.managed_alloc:
        from torch.cuda.memory import CUDAPluggableAllocator
        alloc = CUDAPluggableAllocator(args.managed_alloc, "managed_malloc", "managed_free")
        torch.cuda.memory.change_current_allocator(alloc)
        print(f"# managed allocator active: {args.managed_alloc}", flush=True)

    dev = torch.device("cuda")
    art = Path(args.artifact)
    meta = ExportMetadata.load(art / "metadata.json")
    module = torch.jit.load(str(art / "model_traced.pt"), map_location=dev); module.eval()
    settings = inference_settings_with_dtype("float64")
    settings.external_graph_gen = True
    settings.execution_mode = "general"; settings.merge_mole = False

    print(f"# traced-artifact sweep: {art.name}")
    max_ok = 0
    for nrep in args.sizes:
        r = try_size(module, meta, dev, nrep, settings)
        ms = r.get("ms")
        mss = f"{ms:8.1f} ms" if ms else "      -    "
        dev = r.get("dev_used_gib")
        devs = f"dev={dev:5.1f}G" if dev is not None else ""
        print(f"  NaCl {nrep:2d}^3 N={r['natoms']:7d} peak={r['peak_gib'] or 0:6.2f}G {devs} "
              f"{mss} {'OK' if r['ok'] else 'FAIL('+r.get('err','?')+')'}", flush=True)
        if r["ok"]: max_ok = r["natoms"]
        elif r.get("err") == "OOM": break
    print(f"# TRACED_MAX_ATOMS_FIT = {max_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
