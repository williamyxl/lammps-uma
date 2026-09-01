#!/usr/bin/env python3
"""Phase-6 offline shard export for XPU graph-parallel pair_style uma.

Exports per-rank TorchScript shards model_mp_w{W}_n{N}_r{R}.pt (and legacy
model_mp_w{W}_r{R}.pt for the default) that embed uma_peer collective ops. The
C++ runtime (XcclPeer) implements the collectives; no Python at runtime.

Build-time only (FairChem Python). Traces on CPU (device-agnostic; the traced
module loads onto XPU at runtime). Carries the single-tile campaign fixes:
  - shape-generic quaternion-Wigner trace patches (Phase 3)
  - FP64 Wigner-prep edge-chunk fix for correct backward at large edges (Phase 1)
  - activation_checkpointing OFF (Phase 3c: avoids baked chunk-list)
  - merge_mole OFF (Phase 3b: energy-exact)

Env:
  UMA_CKPT   (default hen uma-s-1p2.pt)
  UMA_TASK   (default omat)
  OUT        artifact dir (writes shards + metadata.json)
  N_LIST     comma NxNxN sizes to export shards for (default "4")
  W_LIST     comma world sizes (default "1,2")
  FXPU_WIGNER_PREP_CHUNK / _MODE  (Wigner-chunk fix knobs)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# G15/S7: resolve the hen shim/patches root via UMA_HEN_ROOT (no hardcoded path).
from uma_hen import add_hen_to_syspath  # noqa: E402
HEN = add_hen_to_syspath()


def build_nacl(n, rattle=0.05, seed=0):
    from ase import Atoms
    a = 5.64
    na = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], float)
    cl = na + 0.5
    syms, sc = [], []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                o = np.array([ix, iy, iz], float)
                for f in na:
                    syms.append("Na"); sc.append((f + o) / n)
                for f in cl:
                    syms.append("Cl"); sc.append((f + o) / n)
    cell = np.eye(3) * (a * n)
    atoms = Atoms(symbols=syms, scaled_positions=sc, cell=cell, pbc=True)
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(0, rattle, atoms.positions.shape)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0
    return atoms


def main() -> int:
    ckpt = Path(os.environ.get("UMA_CKPT", str(HEN / "uma-cache" / "uma-s-1p2.pt")))
    task = os.environ.get("UMA_TASK", "omat")
    out = Path(os.environ["OUT"]); out.mkdir(parents=True, exist_ok=True)
    n_list = [int(x) for x in os.environ.get("N_LIST", "4").split(",") if x.strip()]
    w_list = [int(x) for x in os.environ.get("W_LIST", "1,2").split(",") if x.strip()]

    # Wigner-chunk fix (correct FP64 backward at large edge counts).
    os.environ.setdefault("FXPU_WIGNER_PREP_CHUNK", "65536")
    os.environ.setdefault("FXPU_WIGNER_PREP_CHUNK_MODE", "both")
    # G4 FIX (audit rev 11 / R3): mirror export_blocks_xpu.py's P5'.5 — the
    # wigner-chunk fix is CORRECTNESS-CRITICAL (the N>=10 FP64 cliff); a silently
    # skipped patch yields an artifact with WRONG forces at N>=10. Fail loudly
    # unless explicitly overridden (UMA_ALLOW_MISSING_PATCHES=1, logged).
    _allow_missing = os.environ.get("UMA_ALLOW_MISSING_PATCHES", "0") == "1"
    try:
        from xpu_prepare_wigner import apply_xpu_prepare_wigner_chunking
        note = apply_xpu_prepare_wigner_chunking()
        print(f"wigner-chunk fix applied: {note}", flush=True)
    except Exception as exc:  # noqa: BLE001
        if _allow_missing:
            print(f"WARN wigner-chunk not applied (UMA_ALLOW_MISSING_PATCHES=1): "
                  f"{exc}", flush=True)
        else:
            raise RuntimeError(
                f"wigner-chunk fix (N>=10 FP64 correctness patch) could not be "
                f"applied: {exc}. Set UMA_ALLOW_MISSING_PATCHES=1 to export anyway "
                f"(NOT recommended — the shard artifact may be numerically wrong)."
            ) from exc

    from common import atoms_to_atomic_data, inference_settings_with_dtype
    from export_wrapper import make_traced_export_wrapper
    from metadata import build_export_metadata
    from model_loader import get_atom_refs, load_prepared_hydra_model
    from trace_patch import apply_trace_patches, restore_trace_patches
    from uma_peer_ops import (install_export_ops, patch_fairchem_gp_utils,
                              set_export_rank)

    dtype = "float64"; torch_dtype = torch.float64
    install_export_ops()
    # Trace device. Default XPU (baked constants live on xpu:0, matching runtime).
    # TRACE_DEV=cpu (path d) traces on CPU (unlimited RAM for large-N shards),
    # then MOVE_TRACED_TO_XPU re-homes buffers to xpu before save.
    trace_dev = os.environ.get("TRACE_DEV", "xpu").strip()
    if trace_dev == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit("shard export (TRACE_DEV=xpu) needs 1 visible XPU tile")
    # XPU device allowlist for MLIPPredictUnit/model load.
    try:
        from fairchem_xpu_parallel import patch_fairchem_xpu_device
        patch_fairchem_xpu_device()
    except Exception as exc:  # noqa: BLE001
        # R3: required to place fairchem tensors on the XPU during tracing on the
        # XPU device path; fail loudly there (override UMA_ALLOW_MISSING_PATCHES=1).
        if trace_dev == "xpu" and not _allow_missing:
            raise RuntimeError(
                f"patch_fairchem_xpu_device (required for TRACE_DEV=xpu shard "
                f"export) failed: {exc}. Set UMA_ALLOW_MISSING_PATCHES=1 to override."
            ) from exc
        print(f"WARN patch_fairchem_xpu_device: {exc}", flush=True)

    # (h) internal activation checkpointing: enable escn's edge-chunk AC so the
    # per-block SO2-conv transient is chunked; the checkpoint call is neutralized
    # to a traceable passthrough (see trace_patch._install_checkpoint_passthrough)
    # so the chunk loop traces, and C++ CheckpointModuleFn frees/recomputes.
    act_ckpt = os.environ.get("ACT_CKPT", "1").strip() in ("1", "true", "yes")
    # The escn edge-AC chunk size is a hardcoded 131072 default; per-rank edge
    # counts at W=12 are below that (=> 1 chunk => no benefit). Override it to a
    # smaller value so the SO2-conv transient is actually chunked. Set BEFORE the
    # model is built (read in escn_md.__init__).
    if act_ckpt:
        import fairchem.core.models.uma.escn_md as _escn
        chunk = int(os.environ.get("EDGE_AC_CHUNK", "16384"))
        _escn.ESCNMD_DEFAULT_EDGE_ACTIVATION_CHECKPOINT_CHUNK_SIZE = chunk
        print(f"edge AC chunk size -> {chunk}", flush=True)
    def settings_for():
        s = inference_settings_with_dtype(dtype)
        s.external_graph_gen = True
        s.activation_checkpointing = act_ckpt
        s.execution_mode = "general"
        s.merge_mole = False
        return s

    # metadata (write once from W=1 build)
    wrote_meta = False
    report = []
    skip_existing = os.environ.get("SKIP_EXISTING", "1").strip() in ("1", "true", "yes")
    for n in n_list:
        atoms = build_nacl(n)
        nat = len(atoms)
        for world in w_list:
            # Load the (rank-independent) prepared model ONCE per (N,W); re-trace
            # per rank with set_export_rank + edge slice. Saves 12x model loads.
            s = settings_for()
            sample = atoms_to_atomic_data(atoms, task_name=task, settings=s)
            model, _, _ = load_prepared_hydra_model(
                str(ckpt), sample, settings=s, device=trace_dev)
            model = model.to(device=trace_dev, dtype=torch_dtype)
            data = atoms_to_atomic_data(atoms, task_name=task, settings=s)
            only_rank = os.environ.get("EXPORT_ONLY_RANK", "").strip()
            ranks_iter = [int(only_rank)] if only_rank else list(range(world))
            generic_name = os.environ.get("GENERIC_NAME", "0").strip() in ("1", "true", "yes")
            for rank in ranks_iter:
                if generic_name:
                    # shape-generic (option b): N-agnostic filename (engine
                    # fallback lookup). Trace at small N, run at any N.
                    out_path = out / f"model_mp_w{world}_r{rank}.pt"
                else:
                    out_path = out / f"model_mp_w{world}_n{nat}_r{rank}.pt"
                if skip_existing and out_path.is_file():
                    print(f"W={world} r={rank} N={n} nat={nat} -> SKIP (exists)", flush=True)
                    report.append({"world": world, "rank": rank, "n": n,
                                   "natoms": nat, "ok": True, "path": str(out_path),
                                   "error": None, "skipped": True})
                    continue
                set_export_rank(rank, world)
                restore_gp = patch_fairchem_gp_utils(world, rank)
                wrapper = make_traced_export_wrapper(model, task).eval().to(trace_dev)
                example = list(wrapper.example_inputs_from_data(data))
                example = [t.to(trace_dev) if torch.is_tensor(t) else t for t in example]
                example[0] = example[0].to(torch_dtype)
                example[2] = example[2].to(torch_dtype)
                example[5] = example[5].to(torch_dtype)
                # Shard edges by NODE PARTITION (must match graph_shard.h + escn
                # GP contract): node_partition = tensor_split(arange(nat), W)[rank];
                # keep edges whose CENTER (edge_index[1]) is in this partition.
                # A naive contiguous edge slice includes out-of-partition centers,
                # which escn's `edge_index[1] - node_offset` drives to index -1.
                eidx = example[4]; coff = example[5]
                nodes = torch.tensor_split(
                    torch.arange(nat, device=eidx.device), world)[rank]
                centers = eidx[1]
                keep = torch.isin(centers, nodes)
                idx = keep.nonzero().squeeze(-1)
                example[4] = eidx.index_select(1, idx).contiguous()
                example[5] = coff.index_select(0, idx).contiguous()
                ok, err = True, None
                try:
                    apply_trace_patches(shape_generic=True,
                                        checkpoint_passthrough=act_ckpt)
                    try:
                        # Option (c): trace under no_grad (NOT inference_mode) to
                        # cut trace-time activation memory ~2x. The traced module
                        # is energy-only; forces come from C++ autograd::grad at
                        # runtime, which re-differentiates the recorded forward
                        # ops -> MD accuracy + performance unchanged. Must be
                        # no_grad (inference_mode would make constants
                        # permanently non-differentiable and break force autograd).
                        with torch.no_grad():
                            _ = wrapper(*example)
                            traced = torch.jit.trace(wrapper, tuple(example),
                                                     strict=False)
                    finally:
                        restore_trace_patches()
                    # Path (d): if traced on CPU (TRACE_DEV=cpu), move the traced
                    # module to XPU and re-save so baked buffers become XPU
                    # tensors (avoids the "xpu:0 and cpu" runtime device clash).
                    if trace_dev == "cpu" and os.environ.get("MOVE_TRACED_TO_XPU",
                                                             "1").strip() in ("1", "true", "yes"):
                        try:
                            traced = traced.to("xpu")
                        except Exception as exc:  # noqa: BLE001
                            print(f"WARN move traced->xpu: {exc}", flush=True)
                    traced.save(str(out_path))
                    torch.jit.load(str(out_path), map_location="cpu")
                except Exception as exc:  # noqa: BLE001
                    ok, err = False, f"{type(exc).__name__}: {exc}"
                    import traceback; traceback.print_exc()
                finally:
                    restore_gp()
                print(f"W={world} r={rank} N={n} nat={nat} -> "
                      f"{'OK '+out_path.name if ok else 'FAIL '+str(err)}", flush=True)
                report.append({"world": world, "rank": rank, "n": n, "natoms": nat,
                               "ok": ok, "path": str(out_path) if ok else None,
                               "error": err})
                # Free per-rank trace state to avoid XPU memory accumulation
                # across ranks (re-tracing in one process leaks graph buffers).
                try:
                    del wrapper, example
                    if "traced" in dir():
                        del traced
                    import gc; gc.collect()
                    if hasattr(torch, "xpu"):
                        torch.xpu.empty_cache()
                except Exception:
                    pass
                if ok and not wrote_meta:
                    meta = build_export_metadata(
                        model=model, model_name="uma-s-1p2", task_name=task,
                        settings=s, export_format="mp_xccl",
                        checkpoint_path=str(ckpt),
                        atom_refs=get_atom_refs("uma-s-1p2"),
                        export_notes=["phase6 GP shard export (XPU/XCCL)"])
                    (out / "metadata.json").write_text(
                        json.dumps(meta.to_dict(), indent=2, default=str))
                    wrote_meta = True
    (out / "shard_export_report.json").write_text(json.dumps(report, indent=2))
    nfail = sum(1 for r in report if not r["ok"])
    print(f"\nDONE shards={len(report)} fail={nfail} -> {out}", flush=True)
    return 2 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
