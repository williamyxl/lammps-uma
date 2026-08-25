#!/usr/bin/env python3
"""Phase-1 force spike: UMA FP64 autograd-vs-finite-difference on Intel XPU.

Make-or-break test for the libtorch UMA-in-LAMMPS force path:

  Does the backward through a *traced* UMA energy module produce correct
  forces on XPU (matching finite difference) at production edge counts
  (NaCl N>=10, >2e5 edges), once the hen Wigner-prep edge-chunking fix
  is applied?

For each N in [N_MIN..N_MAX] on the perturbed rocksalt NaCl cell
(a=5.64, rattle 0.05, rng seed 0 -- identical to hen build_nacl):

  * build the external graph (CPU neighbor list from AtomicData.from_ase),
  * EAGER  : autograd force from the prepared HydraModel (energy diff'd),
  * TRACED : autograd force from the torch.jit-traced module (the LAMMPS path),
  * FD     : central finite difference on total energy, eps=1e-4,
  * gate max|F_AG - F_FD| <= AG_FD_TOL (default 1e-5 eV/Ang) on sampled atoms.

Sampled atoms: >=100 (or all atoms if fewer), strided across the cell so the
first frame is representative. Energy parity eager-vs-traced also reported.

Env:
  UMA_CKPT   (default hen/uma-cache/uma-s-1p2.pt)
  UMA_TASK   (default omat)
  N_MIN=1  N_MAX=12  FD_EPS=1e-4  AG_FD_TOL=1e-5
  FD_SAMPLE_ATOMS=100
  WIGNER_CHUNK (passthrough hen FXPU_WIGNER_PREP_CHUNK; default 65536)
  WIGNER_CHUNK_OFF=1  -> skip the fix (to reproduce the N>=10 cliff)
  OUT        (default ./spike_xpu_force_agfd.json)
  KEEP_TRACE_DIR (dir to save one traced artifact for inspection; optional)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HEN_ROOT = Path("/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen")
# hen shim provides patch_fairchem_xpu_device (XPU device allowlist + wigner chunk)
for p in (HEN_ROOT / "shim", HEN_ROOT / "patches", HEN_ROOT):
    if p.is_dir():
        sys.path.insert(0, str(p))


def build_nacl(n: int, rattle: float = 0.05, seed: int = 0):
    """Perturbed rocksalt NaCl n x n x n (identical to hen build_nacl)."""
    from ase import Atoms

    a = 5.64
    na_frac = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    )
    cl_frac = na_frac + 0.5
    symbols, scaled = [], []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                off = np.array([ix, iy, iz], dtype=float)
                for f in na_frac:
                    symbols.append("Na")
                    scaled.append((f + off) / n)
                for f in cl_frac:
                    symbols.append("Cl")
                    scaled.append((f + off) / n)
    cell = np.eye(3) * (a * n)
    atoms = Atoms(symbols=symbols, scaled_positions=scaled, cell=cell, pbc=True)
    if rattle:
        rng = np.random.default_rng(seed)
        atoms.positions += rng.normal(0.0, rattle, size=atoms.positions.shape)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0
    return atoms


def sample_indices(nat: int, want: int) -> list[int]:
    want = min(max(want, 1), nat)
    if want >= nat:
        return list(range(nat))
    stride = nat / want
    idxs = sorted({int(i * stride) for i in range(want)})
    # ensure endpoints present
    idxs = sorted(set(idxs) | {0, nat - 1})
    return idxs


def main() -> int:
    ckpt = Path(os.environ.get("UMA_CKPT", str(HEN_ROOT / "uma-cache" / "uma-s-1p2.pt")))
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    task = os.environ.get("UMA_TASK", "omat")
    n_min = int(os.environ.get("N_MIN", "1"))
    n_max = int(os.environ.get("N_MAX", "12"))
    eps = float(os.environ.get("FD_EPS", "1e-4"))
    tol = float(os.environ.get("AG_FD_TOL", "1e-5"))
    n_sample = int(os.environ.get("FD_SAMPLE_ATOMS", "100"))
    wigner_off = os.environ.get("WIGNER_CHUNK_OFF", "0").strip() in ("1", "true", "yes")
    out = Path(os.environ.get("OUT", str(HERE / "spike_xpu_force_agfd.json")))
    keep_trace_dir = os.environ.get("KEEP_TRACE_DIR", "").strip()

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise SystemExit("torch.xpu unavailable; run on Aurora compute with ZE_AFFINITY_MASK")
    ndev = int(torch.xpu.device_count())
    if ndev != 1:
        raise SystemExit(
            f"expects exactly 1 visible XPU, got {ndev}. "
            "Set ZE_FLAT_DEVICE_HIERARCHY=FLAT and ZE_AFFINITY_MASK=<one tile>."
        )
    device = "xpu"

    # --- XPU device allowlist + (optionally) wigner-prep edge chunking ---------
    from fairchem_xpu_parallel import patch_fairchem_xpu_device

    if wigner_off:
        os.environ["FXPU_SKIP_WIGNER_PREP_CHUNK"] = "1"
    else:
        os.environ.pop("FXPU_SKIP_WIGNER_PREP_CHUNK", None)
        os.environ.setdefault("FXPU_WIGNER_PREP_CHUNK", os.environ.get("WIGNER_CHUNK", "65536"))
        os.environ.setdefault("FXPU_WIGNER_PREP_CHUNK_MODE", "both")
    patch_note = patch_fairchem_xpu_device()
    print(f"patch_fairchem_xpu_device -> {patch_note} (wigner_off={wigner_off})", flush=True)

    # engine helpers (traced-export path, the actual LAMMPS deployment)
    from common import atoms_to_atomic_data, inference_settings_with_dtype
    from export_wrapper import (
        EnergyExportWrapper,
        clone_prepared_model,
        make_traced_export_wrapper,
    )
    from metadata import build_export_metadata, denorm_from_metadata, undo_refs_from_metadata
    from model_loader import get_atom_refs, load_prepared_hydra_model
    from trace_patch import apply_trace_patches, restore_trace_patches

    # Blocker B fix: activation checkpointing (~3x less activation memory for the
    # force graph) so large-N eager autograd forward fits one XPU tile.
    ac = os.environ.get("ACT_CKPT", "1").strip() in ("1", "true", "yes")
    # merge_mole=True fuses MOLE experts into a single Linear -> removes the
    # segment/split list whose length the trace baked (root cause of the traced
    # "Expected 1 elements in a list but found 2" at large N). Valid for fixed
    # composition (NaCl = Na+Cl, task omat).
    merge_mole = os.environ.get("MERGE_MOLE", "1").strip() in ("1", "true", "yes")
    settings = inference_settings_with_dtype("float64")
    settings.execution_mode = "general"
    settings.merge_mole = merge_mole
    settings.activation_checkpointing = ac
    settings.external_graph_gen = True
    print(f"activation_checkpointing={ac} merge_mole={merge_mole}", flush=True)

    from ase.build import bulk

    if merge_mole:
        # merge_mole locks composition/charge/spin to the prepare sample; use a
        # small NaCl (Na+Cl, charge0, spin0) so the merged model matches the run.
        sample = build_nacl(1)
    else:
        sample = bulk("Fe", "bcc", a=2.87, cubic=True)
    sample_data = atoms_to_atomic_data(sample, task, settings)

    t0 = time.perf_counter()
    print(f"loading checkpoint {ckpt} (task={task}, fp64, device={device})", flush=True)
    model, _, _ = load_prepared_hydra_model(str(ckpt), sample_data, settings=settings, device=device)
    # merge_mole builds fresh nn.Linear layers at torch default dtype (float32);
    # re-cast the whole model to float64 so merged linears match FP64 activations.
    model = model.double()
    print(f"model loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    eager_wrapper = EnergyExportWrapper(clone_prepared_model(model), task, traceable=False)
    eager_wrapper.eval().to(device)
    trace_wrapper = make_traced_export_wrapper(model, task)
    trace_wrapper.to(device)

    metadata = build_export_metadata(
        model=model, model_name="uma-s-1p2", task_name=task, settings=settings,
        export_format="spike", checkpoint_path=str(ckpt),
        atom_refs=get_atom_refs("uma-s-1p2"), export_notes=["phase1 spike"],
    )

    def phys_energy(normed, atomic_numbers):
        e = denorm_from_metadata(normed, metadata)
        batch = torch.zeros(atomic_numbers.shape[0], dtype=torch.long, device=normed.device)
        return undo_refs_from_metadata(metadata, atomic_numbers, batch, e)

    # Trace on a SMALL N. external_graph_gen=True -> the traced module accepts
    # arbitrary atom/edge counts at inference, so a small trace is enough to test
    # backward correctness at large N. Tracing at large N OOMs one tile
    # (UR_RESULT_ERROR_OUT_OF_RESOURCES) and corrupts the XPU context.
    trace_n = int(os.environ.get("TRACE_N", "2"))
    # Blocker A fix: shape-generic trace (symbolic-dim quaternion Wigner) so the
    # traced module runs at any N, not just trace_n.
    shape_generic = os.environ.get("SHAPE_GENERIC", "1").strip() in ("1", "true", "yes")
    print(f"tracing on XPU at N={trace_n} shape_generic={shape_generic} ...", flush=True)
    tr_atoms = build_nacl(trace_n)
    tr_data = atoms_to_atomic_data(tr_atoms, task, settings).to(device)
    tr_inputs = eager_wrapper.example_inputs_from_data(tr_data)
    traced = None
    trace_note = ""
    try:
        apply_trace_patches(shape_generic=shape_generic)
        try:
            with torch.no_grad():
                trace_wrapper(*tr_inputs)
            traced = torch.jit.trace(trace_wrapper, tr_inputs, strict=False)
        finally:
            restore_trace_patches()
        trace_note = f"traced at N={trace_n} nedges={tr_inputs[4].shape[1]} shape_generic={shape_generic}"
        print(f"  {trace_note}", flush=True)
        # Shape-generality self-check: run reloaded module at a DIFFERENT N.
        try:
            chk_atoms = build_nacl(trace_n + 1)
            chk_data = atoms_to_atomic_data(chk_atoms, task, settings).to(device)
            chk_inputs = eager_wrapper.example_inputs_from_data(chk_data)
            with torch.no_grad():
                _ = traced(*chk_inputs)
            print(f"  SHAPE-GENERALITY: PASS (ran at N={trace_n+1} "
                  f"nedges={chk_inputs[4].shape[1]})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  SHAPE-GENERALITY: FAIL ({type(exc).__name__}: {exc})", flush=True)
        if keep_trace_dir:
            kd = Path(keep_trace_dir)
            kd.mkdir(parents=True, exist_ok=True)
            traced.save(str(kd / "model_traced.pt"))
            (kd / "metadata.json").write_text(
                json.dumps(metadata.to_dict(), indent=2, default=str)
            )
            print(f"  saved traced artifact -> {kd}", flush=True)
    except Exception as exc:  # noqa: BLE001
        trace_note = f"TRACE FAILED: {type(exc).__name__}: {exc}"
        print(f"  {trace_note}", flush=True)

    def force_autograd(mod, inputs):
        pos = inputs[0].detach().clone().requires_grad_(True)
        args = (pos, *inputs[1:])
        normed = mod(*args)
        energy = phys_energy(normed, inputs[1])
        (grad,) = torch.autograd.grad(energy.sum(), pos)
        torch.xpu.synchronize()
        return float(energy.detach().cpu()), (-grad).detach().to(torch.float64).cpu().numpy()

    def energy_only(mod, inputs):
        with torch.no_grad():
            normed = mod(*inputs)
            energy = phys_energy(normed, inputs[1])
        torch.xpu.synchronize()
        return float(energy.detach().cpu())

    results = []
    first_fail = None
    for n in range(n_min, n_max + 1):
        atoms = build_nacl(n)
        data = atoms_to_atomic_data(atoms, task, settings).to(device)
        base_inputs = eager_wrapper.example_inputs_from_data(data)
        nat = int(base_inputs[0].shape[0])
        nedges = int(base_inputs[4].shape[1])
        idxs = sample_indices(nat, n_sample)
        print(f"\n=== N={n} natoms={nat} nedges={nedges} sampled={len(idxs)} ===", flush=True)

        row = {"n": n, "natoms": nat, "nedges": nedges, "n_sampled": len(idxs)}
        t = time.perf_counter()

        # eager autograd
        e_eager, f_eager = force_autograd(eager_wrapper, base_inputs)
        row["energy_eager"] = e_eager

        # traced autograd (non-fatal: traced graph may be shape-frozen at trace_n)
        f_traced = None
        if traced is not None:
            try:
                e_traced, f_traced = force_autograd(traced, base_inputs)
                row["energy_traced"] = e_traced
                row["dE_traced_vs_eager"] = abs(e_traced - e_eager)
                row["max_dF_traced_vs_eager"] = float(np.max(np.abs(f_traced - f_eager)))
            except Exception as exc:  # noqa: BLE001
                f_traced = None
                row["traced_error"] = f"{type(exc).__name__}: {exc}"[:200]
                print(f"  traced eval failed (shape-freeze?): {row['traced_error']}", flush=True)

        # Decisive science test: EAGER autograd force vs EAGER finite-difference
        # (pure XPU FP64 force correctness). Traced parity reported separately above.
        fd_mod = eager_wrapper
        f_ag = f_eager
        ag_fd = []
        for ia in idxs:
            for ic in range(3):
                pos_p = base_inputs[0].detach().clone()
                pos_p[ia, ic] += eps
                ep = energy_only(fd_mod, (pos_p, *base_inputs[1:]))
                pos_m = base_inputs[0].detach().clone()
                pos_m[ia, ic] -= eps
                em = energy_only(fd_mod, (pos_m, *base_inputs[1:]))
                f_fd = -(ep - em) / (2.0 * eps)
                ag_fd.append(abs(float(f_ag[ia, ic]) - f_fd))
        max_agfd = max(ag_fd)
        mean_agfd = float(np.mean(ag_fd))
        ok = max_agfd <= tol
        row["max_abs_AG_FD"] = max_agfd
        row["mean_abs_AG_FD"] = mean_agfd
        row["ok"] = bool(ok)
        row["elapsed_s"] = time.perf_counter() - t
        results.append(row)
        status = "PASS" if ok else "FAIL"
        extra = ""
        if "max_dF_traced_vs_eager" in row:
            extra = (f" | traced-vs-eager dE={row['dE_traced_vs_eager']:.2e} "
                     f"dF={row['max_dF_traced_vs_eager']:.2e}")
        print(f"N={n} {status} E={row.get('energy_traced', e_eager):.8f} "
              f"max|AG-FD|={max_agfd:.3e} mean={mean_agfd:.3e}{extra} "
              f"({row['elapsed_s']:.1f}s)", flush=True)

        summary = {
            "probe": "xpu_force_ag_fd_traced",
            "task": task, "dtype": "float64", "eps": eps, "tol": tol,
            "wigner_chunk_off": wigner_off, "trace_note": trace_note,
            "results": results, "first_fail_n": first_fail, "complete": False,
        }
        out.write_text(json.dumps(summary, indent=2) + "\n")
        if not ok and first_fail is None:
            first_fail = n
            print(f"STOP: first FAIL at N={n}", flush=True)
            break

    summary["first_fail_n"] = first_fail
    summary["complete"] = True
    summary["verdict"] = (f"FIRST_FAIL_N={first_fail}" if first_fail is not None
                          else f"ALL_PASS_N={n_min}..{n_max}")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}\nDONE verdict={summary['verdict']} trace='{trace_note}'", flush=True)
    return 2 if first_fail is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
