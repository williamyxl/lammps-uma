#!/usr/bin/env python3
"""Multi-GPU NaCl 6x6x6 parity: ASE / FairChem FC / uma/kk double+mixed.

Environment
-----------
NGPUS=1|2|4              number of GPUs (default 1)
UMA_DEVICES=NGPUS        uma/kk ``devices`` in pair_style (default NGPUS)
N_TIMING=5               timed evals / NVE steps
ONLY_PATHS=<one path>    required under SLURM: ase|fc|uma_double|uma_mixed
                         (multi-path needs ALLOW_MULTI_PATH=1; prefer separate jobs)
MERGE_RESULTS=1          merge into existing parity.json (default for path jobs)
USE_SLURM_TIMING=1       set by _run_common.sh — ms_per_eval comes from SLURM wall
                         (stamp_slurm_timing.py); Python/Pair timers are debug only
N_TIMING=5               eval repeats inside run_multigpu; SLURM ms/eval = wall/N_TIMING
UMA_CHECKPOINT / LMP_UMA / LMP_FC        path overrides (env)

Geometry
--------
Always loads frozen ``structures/nacl6_rattle_fixed.extxyz``
(1728 atoms). Never re-rattles.

Multi-GPU recipes
-----------------
* ASE / FairChem FC: ``load_predict_unit(..., workers=NGPUS)`` → Ray
  ``ParallelMLIPPredictUnit`` (graph-parallel across GPUs). Requires
  ``fairchem-core[extras]`` / Ray. FP64 via custom InferenceSettings.
* uma/kk: single MPI rank + Kokkos ``lmp -k on g NGPUS -sf kk`` and
  ``pair_style uma/kk precision ... devices UMA_DEVICES``. Graph-parallel
  UMA inference shards across devices when WRITE lands engine GP support.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from ase import Atoms
from ase.data import atomic_masses, chemical_symbols

from load_geometry import geometry_meta, load_nacl6_fixed  # noqa: E402
from parity_gates import (  # noqa: E402
    PRECISION_BY_KEY,
    check_gate,
    summarize_gates,
)


def _find_ml_uma_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / "pair_uma.cpp").is_file() and (p / "uma-engine").is_dir():
            return p
    raise RuntimeError(
        "cannot find ML-UMA package root (expected pair_uma.cpp + uma-engine/)"
    )


def find_uma_engine_root(start: Path | None = None) -> Path:
    return _find_ml_uma_root(start) / "uma-engine"


def find_lammps_root(start: Path | None = None) -> Path:
    return _find_ml_uma_root(start).parent.parent


def find_checkpoint() -> Path:
    env = os.environ.get("UMA_CHECKPOINT")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        find_lammps_root().parent / "uma-cache" / "uma-s-1p2.pt",
        Path("/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt"),
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "uma-s-1p2.pt not found; set UMA_CHECKPOINT=/path/to/uma-s-1p2.pt"
    )


def find_uma_lmp_binary() -> Path:
    env = os.environ.get("LMP_UMA")
    if env:
        return Path(env).expanduser().resolve()
    root = find_lammps_root()
    for c in (root / "build-uma" / "lmp", root / "build-uma" / "lmp_kokkos_cuda"):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "local uma/kk LAMMPS binary not found; build build-uma/lmp or set LMP_UMA="
    )


def find_fairchem_lmp_binary() -> Path:
    env = os.environ.get("LMP_FC")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        Path("/u/xyan11/miniforge3-x86_64/envs/uma312/bin/lmp"),
        Path(os.environ["CONDA_PREFIX"]) / "bin" / "lmp"
        if os.environ.get("CONDA_PREFIX")
        else None,
    ]
    for c in candidates:
        if c is not None and c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "FairChem LAMMPS binary not found; set LMP_FC=/path/to/conda/bin/lmp"
    )


ENGINE = find_uma_engine_root()
LAMMPS_ROOT = find_lammps_root()
sys.path.insert(0, str(ENGINE / "python"))
from common import inference_settings_with_dtype  # noqa: E402

OUT = Path(__file__).resolve().parent
RESULTS = OUT / "results"

ART_F64 = Path(
    os.environ.get(
        "UMA_ARTIFACT_DIR",
        str(ENGINE / "artifacts" / "uma-s-1p2-omat-f64"),
    )
)
ART_MIX = ENGINE / "artifacts" / "uma-s-1p2-omat"

ALL_PATHS = ("ase", "fc", "uma_double", "uma_mixed")


def release_cuda(tag: str = "") -> None:
    """Drop Python-held CUDA tensors so the next path can use VRAM."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    free, total = torch.cuda.mem_get_info()
    used = total - free
    label = f" ({tag})" if tag else ""
    print(
        f"GPU VRAM after release{label}: "
        f"used={used / (1024**3):.2f} GiB  free={free / (1024**3):.2f} GiB / "
        f"{total / (1024**3):.2f} GiB",
        flush=True,
    )


def teardown_predict_unit(
    predictor,
    tag: str = "",
    *,
    timeout_s: float | None = None,
    skip_dist_destroy: bool | None = None,
) -> None:
    """Drop FairChem predictor and shut down Ray so the next path can claim GPUs.

    ``dist.destroy_process_group`` has hung after FC (NCCL). Default:
    ``SKIP_DIST_DESTROY=1`` (skip destroy) and ``TEARDOWN_TIMEOUT_S=45`` wall
    timeout around ``ray.shutdown`` via SIGALRM so the parent can still flush
    results. For a fully clean FC@workers>1 path prefer subprocess isolation
    (see ``run_fairchem_lammps``).
    """
    import signal

    if timeout_s is None:
        timeout_s = float(os.environ.get("TEARDOWN_TIMEOUT_S", "45"))
    if skip_dist_destroy is None:
        skip_dist_destroy = os.environ.get("SKIP_DIST_DESTROY", "1") != "0"

    try:
        del predictor
    except Exception:
        pass
    gc.collect()

    timed_out = {"flag": False}

    def _alarm_handler(signum, frame):  # noqa: ARG001
        timed_out["flag"] = True
        raise TimeoutError(f"teardown timed out after {timeout_s}s")

    old_handler = None
    try:
        if hasattr(signal, "SIGALRM") and timeout_s > 0:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_s)

        if not skip_dist_destroy:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    print(
                        f"dist.destroy_process_group ({tag})"
                        if tag
                        else "dist.destroy_process_group",
                        flush=True,
                    )
                    dist.destroy_process_group()
            except TimeoutError:
                raise
            except Exception as exc:
                label = f" ({tag})" if tag else ""
                print(f"dist.destroy_process_group warning{label}: {exc}", flush=True)
        else:
            print(
                f"skip dist.destroy_process_group ({tag})"
                if tag
                else "skip dist.destroy_process_group",
                flush=True,
            )

        try:
            import ray

            if ray.is_initialized():
                print(f"ray.shutdown ({tag})" if tag else "ray.shutdown", flush=True)
                ray.shutdown()
        except TimeoutError:
            raise
        except Exception as exc:
            label = f" ({tag})" if tag else ""
            print(f"ray.shutdown warning{label}: {exc}", flush=True)
    except TimeoutError as exc:
        print(f"teardown TIMEOUT ({tag}): {exc} — continuing", flush=True)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    release_cuda(tag)


def parse_ngpus() -> int:
    raw = os.environ.get("NGPUS", "1").strip()
    try:
        n = int(raw)
    except ValueError as exc:
        raise SystemExit(f"NGPUS must be int 1|2|4, got {raw!r}") from exc
    if n not in (1, 2, 4):
        raise SystemExit(f"NGPUS must be 1, 2, or 4 (got {n})")
    return n


def parse_only_paths() -> list[str]:
    raw = os.environ.get("ONLY_PATHS", "").strip()
    if not raw:
        return list(ALL_PATHS)
    names = [p.strip().lower() for p in raw.split(",") if p.strip()]
    aliases = {
        "ase": "ase",
        "ase_fp64": "ase",
        "fc": "fc",
        "fairchem": "fc",
        "uma_double": "uma_double",
        "uma64": "uma_double",
        "double": "uma_double",
        "uma_mixed": "uma_mixed",
        "mixed": "uma_mixed",
    }
    out: list[str] = []
    for n in names:
        if n not in aliases:
            raise SystemExit(
                f"unknown ONLY_PATHS entry {n!r}; use "
                f"{'|'.join(ALL_PATHS)} (comma-separated)"
            )
        key = aliases[n]
        if key not in out:
            out.append(key)
    return out


def write_data(atoms: Atoms, path: Path, title: str) -> list[str]:
    Z = atoms.get_atomic_numbers()
    uniq = sorted(set(int(z) for z in Z))
    z_to_type = {z: i + 1 for i, z in enumerate(uniq)}
    types = np.array([z_to_type[int(z)] for z in Z], dtype=np.int32)
    symbols = [chemical_symbols[z] for z in uniq]
    cell = atoms.cell.array
    lx, ly, lz = float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])
    pos = atoms.get_positions()
    lines = [
        title,
        "",
        f"{len(atoms)} atoms",
        f"{len(uniq)} atom types",
        "",
        f"0.0 {lx:.16f} xlo xhi",
        f"0.0 {ly:.16f} ylo yhi",
        f"0.0 {lz:.16f} zlo zhi",
        "",
        "Masses",
        "",
    ]
    for i, z in enumerate(uniq, 1):
        lines.append(f"{i} {atomic_masses[z]:.8f}")
    lines += ["", "Atoms # atomic", ""]
    for i, (t, p) in enumerate(zip(types, pos), 1):
        lines.append(f"{i} {t} {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}")
    path.write_text("\n".join(lines) + "\n")
    return symbols


def force_stats(f_ref: np.ndarray, f: np.ndarray) -> dict:
    df = f - f_ref
    return {
        "force_mae": float(np.mean(np.abs(df))),
        "force_rmse": float(np.sqrt(np.mean(df**2))),
        "force_max_abs": float(np.max(np.abs(df))),
        "force_max_norm_per_atom": float(np.max(np.linalg.norm(df, axis=1))),
        "cosine": float(
            np.dot(f_ref.ravel(), f.ravel())
            / (np.linalg.norm(f_ref) * np.linalg.norm(f) + 1e-30)
        ),
        "f_ref_max_abs": float(np.max(np.abs(f_ref))),
        "f_ref_rms": float(np.sqrt(np.mean(f_ref**2))),
    }


def setup_ld_path() -> dict:
    env = os.environ.copy()
    vesin = ENGINE / "third_party" / "vesin" / "lib"
    torch_lib = Path(torch.__path__[0]) / "lib"
    parts = [
        str(vesin),
        str(torch_lib),
        "/usr/local/cuda/lib64",
        "/usr/lib/wsl/lib",
    ]
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    try:
        fc = find_fairchem_lmp_binary()
        env["PATH"] = f"{fc.parent}:{env.get('PATH', '')}"
    except FileNotFoundError:
        pass
    # C++ LibTorch MP worker (process-per-rank). Prefer explicit env, else
    # build-cpp-mp / build-uma sidecar next to the engine.
    if not env.get("UMA_LIBTORCH_MP_WORKER"):
        lammps_root = ENGINE.parent.parent.parent  # .../lammps-uma
        for cand in (
            ENGINE / "build-cpp-mp" / "uma_libtorch_mp_worker",
            lammps_root / "build-uma" / "uma-engine" / "uma_libtorch_mp_worker",
        ):
            if cand.is_file() and os.access(cand, os.X_OK):
                env["UMA_LIBTORCH_MP_WORKER"] = str(cand)
                break
    # NaCl6 (1728) needs n-specific MP shards (baked gp_node_offset).
    if not env.get("UMA_MP_NATOMS"):
        n_atoms = int(os.environ.get("UMA_STRUCTURE_NATOMS", "0") or "0")
        if n_atoms > 0:
            env["UMA_MP_NATOMS"] = str(n_atoms)
    return env


def fairchem_knobs_from_env() -> tuple[str, bool]:
    """Optional ASE/FC InferenceSettings overrides (defaults = product general).

    FAIRCHEM_EXECUTION_MODE: general | umas_fast_pytorch
    FAIRCHEM_MERGE_MOLE: 0|1|true|false (default false)
    """
    mode = os.environ.get("FAIRCHEM_EXECUTION_MODE", "general").strip() or "general"
    if mode not in ("general", "umas_fast_pytorch"):
        raise SystemExit(
            f"FAIRCHEM_EXECUTION_MODE must be general|umas_fast_pytorch, got {mode!r}"
        )
    raw = os.environ.get("FAIRCHEM_MERGE_MOLE", "0").strip().lower()
    merge = raw in ("1", "true", "yes", "on")
    if mode == "umas_fast_pytorch" and not merge:
        raise SystemExit(
            "FAIRCHEM_EXECUTION_MODE=umas_fast_pytorch requires FAIRCHEM_MERGE_MOLE=1"
        )
    return mode, merge


def apply_fairchem_knobs(settings) -> tuple[str, bool]:
    mode, merge = fairchem_knobs_from_env()
    settings.execution_mode = mode
    settings.merge_mole = merge
    return mode, merge


def fp64_settings(*, external_graph: bool):
    settings = inference_settings_with_dtype("float64")
    settings.external_graph_gen = external_graph
    apply_fairchem_knobs(settings)
    return settings


def dtype_settings(dtype: str, *, external_graph: bool):
    settings = inference_settings_with_dtype(dtype)
    settings.external_graph_gen = external_graph
    settings.activation_checkpointing = False
    apply_fairchem_knobs(settings)
    return settings


def run_ase_dtype(
    atoms: Atoms,
    ckpt: Path,
    *,
    dtype: str,
    workers: int = 1,
    n_timing: int = 1,
) -> dict:
    """ASE FairChem at explicit dtype/workers (used as mixed-GP oracle @ float32/w1)."""
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    settings = dtype_settings(dtype, external_graph=(workers <= 1))
    t0 = time.perf_counter()
    predictor = load_predict_unit(
        str(ckpt),
        device="cuda",
        inference_settings=settings,
        workers=workers,
    )
    calc = FAIRChemCalculator(predictor, task_name="omat")
    a = atoms.copy()
    a.calc = calc
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(), dtype=np.float64)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    times = []
    for _ in range(max(1, n_timing)):
        if hasattr(a.calc, "results"):
            a.calc.results.clear()
        a.positions = a.positions.copy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e = float(a.get_potential_energy())
        f = np.asarray(a.get_forces(), dtype=np.float64)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t1)
    out = _pack(
        f"ASE FairChem {dtype} workers={workers}",
        e,
        f,
        float(np.mean(times) * 1e3),
        load_s=load_s,
        workers=workers,
        dtype=dtype,
        multi_gpu_note=f"mixed-GP oracle path: load_predict_unit(workers={workers}, dtype={dtype})",
    )
    a.calc = None
    del calc, a
    teardown_predict_unit(predictor, f"ASE-{dtype}-w{workers}")
    return out


def parse_uma_devices(ngpus: int) -> int:
    raw = os.environ.get("UMA_DEVICES", str(ngpus)).strip()
    try:
        n = int(raw)
    except ValueError as exc:
        raise SystemExit(f"UMA_DEVICES must be int 1|2|4, got {raw!r}") from exc
    if n not in (1, 2, 4):
        raise SystemExit(f"UMA_DEVICES must be 1, 2, or 4 (got {n})")
    return n


def _run_lmp(cmd: list[str], *, cwd: Path, env: dict, tag: str) -> None:
    """Run LAMMPS without capture_output (avoids Ray/stderr pipe deadlock)."""
    out_path = cwd / f"lmp_{tag}.stdout"
    err_path = cwd / f"lmp_{tag}.stderr"
    with out_path.open("w") as out, err_path.open("w") as err:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=out, stderr=err, text=True)
    if proc.returncode != 0:
        out_t = out_path.read_text()[-2000:] if out_path.is_file() else ""
        err_t = err_path.read_text()[-4000:] if err_path.is_file() else ""
        raise RuntimeError(
            f"LAMMPS failed (rc={proc.returncode}) tag={tag}:\n{out_t}\n{err_t}"
        )


def _uma_use_kokkos() -> bool:
    """W8nk product: default off → ``pair_style uma`` (no ``-k``/``-sf kk``).

    Set ``UMA_USE_KOKKOS=1`` only for A/B vs legacy ``uma/kk``.
    """
    v = os.environ.get("UMA_USE_KOKKOS", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def uma_pair_style_line(precision: str, uma_devices: int) -> str:
    """``uma/kk`` when Kokkos on; plain ``uma`` + explicit ``devices N`` otherwise."""
    style = "uma/kk" if _uma_use_kokkos() else "uma"
    return f"pair_style {style} precision {precision} devices {uma_devices}"


def uma_kk_argv(ngpus: int, *extra: str, uma_devices: int | None = None) -> list[str]:
    """LAMMPS argv: with Kokkos ``-k on g N -sf kk``; without, bare ``lmp``."""
    lmp = find_uma_lmp_binary()
    del uma_devices  # pair_style carries devices
    if _uma_use_kokkos():
        return [str(lmp), "-k", "on", "g", str(ngpus), "-sf", "kk", *extra]
    return [str(lmp), *extra]


def load_uma_d1_baseline(results_root: Path) -> dict[str, dict]:
    """Load uma_double / uma_mixed E+F from ngpu1 (devices=1 oracle)."""
    base_dir = results_root / "ngpu1"
    parity_path = base_dir / "parity.json"
    forces_path = base_dir / "forces.npz"
    if not parity_path.is_file() or not forces_path.is_file():
        return {}
    parity = json.loads(parity_path.read_text())
    forces_npz = np.load(forces_path)
    out: dict[str, dict] = {}
    for row in parity.get("rows") or []:
        key = row.get("key")
        if key not in PRECISION_BY_KEY:
            continue
        fkey = f"forces_{key}"
        if fkey not in forces_npz.files:
            continue
        out[key] = {
            "energy_eV": float(row["energy_eV"]),
            "forces": np.asarray(forces_npz[fkey], dtype=np.float64),
            "uma_devices": row.get("uma_devices", 1),
        }
    return out


def _pack(path: str, e: float, f: np.ndarray, ms: float | None, **extra) -> dict:
    return {
        "path": path,
        "energy_eV": float(e),
        "natoms": len(f),
        "forces": np.asarray(f, dtype=np.float64),
        "ms_per_eval": None if ms is None else float(ms),
        **extra,
    }


def run_ase(atoms: Atoms, ckpt: Path, n_timing: int, ngpus: int) -> dict:
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    workers = int(os.environ.get("FAIRCHEM_WORKERS", str(ngpus)))
    # Graph-parallel (workers>1) + external_graph=True trips CUDA index asserts on
    # UMA (seen on NGPUS=2). Use internal graph gen for multi-GPU ASE.
    settings = fp64_settings(external_graph=(workers <= 1))
    mode = settings.execution_mode
    merge = bool(settings.merge_mole)
    t0 = time.perf_counter()
    # workers>1 → ParallelMLIPPredictUnit (Ray graph-parallel). workers=1 → single GPU.
    predictor = load_predict_unit(
        str(ckpt),
        device="cuda",
        inference_settings=settings,
        workers=workers,
    )
    calc = FAIRChemCalculator(predictor, task_name="omat")
    a = atoms.copy()
    a.calc = calc
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(), dtype=np.float64)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    times = []
    for _ in range(n_timing):
        if hasattr(a.calc, "results"):
            a.calc.results.clear()
        a.positions = a.positions.copy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e = float(a.get_potential_energy())
        f = np.asarray(a.get_forces(), dtype=np.float64)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t1)

    out = _pack(
        "ASE FairChem FP64",
        e,
        f,
        float(np.mean(times) * 1e3),
        load_s=load_s,
        workers=workers,
        execution_mode=mode,
        merge_mole=merge,
        multi_gpu_note=(
            f"load_predict_unit(workers={workers}); "
            f"execution_mode={mode} merge_mole={merge}; "
            + (
                "ParallelMLIPPredictUnit / Ray graph-parallel"
                if workers > 1
                else "single MLIPPredictUnit"
            )
        ),
    )
    a.calc = None
    del calc, a
    teardown_predict_unit(predictor, "ASE")
    return out


def run_fairchem_lammps(
    atoms: Atoms, ckpt: Path, n_timing: int, work: Path, title: str, ngpus: int
) -> dict:
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.lammps.lammps_fc import run_lammps_with_fairchem

    settings = fp64_settings(external_graph=False)
    mode = settings.execution_mode
    merge = bool(settings.merge_mole)
    data = work / "data.lmp"
    write_data(atoms, data, title)
    inp = work / "in.fc"
    inp.write_text(
        f"""units metal
atom_style atomic
boundary p p p
read_data {data.name}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo 1
thermo_style custom step pe
run 0
"""
    )

    fc_lmp = find_fairchem_lmp_binary()
    os.environ["PATH"] = f"{fc_lmp.parent}:{os.environ.get('PATH', '')}"

    workers = int(os.environ.get("FAIRCHEM_WORKERS", str(ngpus)))
    t0 = time.perf_counter()
    predictor = load_predict_unit(
        str(ckpt),
        device="cuda",
        inference_settings=settings,
        workers=workers,
    )
    load_s = time.perf_counter() - t0

    cwd = Path.cwd()
    os.chdir(work)
    try:
        t1 = time.perf_counter()
        # LAMMPS itself is a single Python process; multi-GPU is Ray workers
        # inside the predictor (same as ASE path).
        lmp = run_lammps_with_fairchem(predictor, str(inp), "omat")
        first_s = time.perf_counter() - t1

        e = float(lmp.get_thermo("pe"))
        nlocal = lmp.extract_global("nlocal")
        tags = np.array(lmp.numpy.extract_atom("id")[:nlocal]).copy()
        order = np.argsort(tags)
        f = np.array(lmp.numpy.extract_atom("f")[:nlocal], dtype=np.float64).copy()[
            order
        ]

        times = [first_s]
        for _ in range(max(0, n_timing - 1)):
            t2 = time.perf_counter()
            lmp.command("run 0")
            times.append(time.perf_counter() - t2)
            e = float(lmp.get_thermo("pe"))
            f = np.array(
                lmp.numpy.extract_atom("f")[:nlocal], dtype=np.float64
            ).copy()[order]

        steady = times[1:] if len(times) > 1 else times
        if hasattr(lmp, "_predictor"):
            del lmp._predictor
        lmp.close()
        del lmp
    finally:
        os.chdir(cwd)

    out = _pack(
        "FairChem LAMMPS fix external",
        e,
        f,
        float(np.mean(steady) * 1e3),
        load_s=load_s,
        fairchem_lmp=str(fc_lmp),
        workers=workers,
        execution_mode=mode,
        merge_mole=merge,
        note="lammps_fc builds cell in FP32; not a pure FP64 path",
        multi_gpu_note=(
            f"predictor workers={workers}; execution_mode={mode} merge_mole={merge}; "
            "LAMMPS single-process Python bridge"
        ),
    )
    # Persist before teardown — Ray/NCCL cleanup can hang.
    early_json = work / "fc_result_early.json"
    early_npz = work / "fc_result_early.npz"
    early_json.write_text(
        json.dumps({k: v for k, v in out.items() if k != "forces"}, indent=2) + "\n"
    )
    np.savez(early_npz, forces=out["forces"], energy_eV=out["energy_eV"])
    print(f"wrote early FC result {early_json}", flush=True)

    if os.environ.get("HARD_EXIT_AFTER_FC", "0") == "1":
        # Merge FC row into existing parity.json then hard-exit (skip Ray hang).
        out_dir = work.parent if work.name == "fc" else work
        # work is .../ngpu2/work/fc → out_dir = .../ngpu2
        if work.name == "fc":
            out_dir = work.parent.parent
        parity_path = out_dir / "parity.json"
        row = {
            "path": out["path"],
            "key": "fc",
            "energy_eV": out["energy_eV"],
            "ms_per_eval": out["ms_per_eval"],
            "workers": out.get("workers"),
            "note": out.get("note"),
            "multi_gpu_note": out.get("multi_gpu_note"),
            "fairchem_lmp": out.get("fairchem_lmp"),
            "load_s": out.get("load_s"),
        }
        if parity_path.is_file():
            rep = json.loads(parity_path.read_text())
            rows = [r for r in (rep.get("rows") or []) if r.get("key") != "fc"]
            # dE vs ASE if present
            ase_e = None
            for r in rows:
                if r.get("key") == "ase":
                    ase_e = r.get("energy_eV")
                    break
            if ase_e is not None:
                row["abs_dE_vs_ase_f64"] = abs(out["energy_eV"] - ase_e)
            rows.append(row)
            # keep path order ase, fc, uma_double, uma_mixed when possible
            order = {"ase": 0, "fc": 1, "uma_double": 2, "uma_mixed": 3}
            rows.sort(key=lambda r: order.get(r.get("key"), 9))
            rep["rows"] = rows
            rep["paths_run"] = list(dict.fromkeys(list(rep.get("paths_run") or []) + ["fc"]))
            rep["paths_merged"] = True
            parity_path.write_text(json.dumps(rep, indent=2) + "\n")
            print(f"merged FC into {parity_path}", flush=True)
            # merge forces
            forces_path = out_dir / "forces.npz"
            if forces_path.is_file():
                old = dict(np.load(forces_path))
                old["forces_fc"] = out["forces"]
                old["energy_fc_eV"] = np.array(out["energy_eV"])
                np.savez(forces_path, **old)
        else:
            # no prior — write minimal parity
            rep = {
                "ngpus": ngpus,
                "paths_run": ["fc"],
                "rows": [row],
                "note": "FC-only hard-exit write; other paths absent",
            }
            parity_path.write_text(json.dumps(rep, indent=2) + "\n")
            np.savez(
                out_dir / "forces.npz",
                forces_fc=out["forces"],
                energy_fc_eV=np.array(out["energy_eV"]),
            )
            print(f"wrote FC-only {parity_path}", flush=True)
        print("HARD_EXIT_AFTER_FC=1 — exiting before Ray teardown", flush=True)
        os._exit(0)

    teardown_predict_unit(predictor, "FC")
    return out


def run_uma_kk(
    atoms: Atoms,
    *,
    precision: str,
    artifact: Path,
    symbols: list[str],
    n_timing_steps: int,
    work: Path,
    title: str,
    ngpus: int,
    uma_devices: int,
) -> dict:
    if not (artifact / "model_traced.pt").is_file():
        raise FileNotFoundError(
            f"missing {artifact / 'model_traced.pt'} — export with export_artifact.py first"
        )
    data = work / "data.lmp"
    write_data(atoms, data, title)
    dump = work / "forces.dump"
    log_sp = work / "log.sp"
    log_nve = work / "log.nve"
    env = setup_ld_path()
    # MP TorchScript shards bake gp_node_offset for a fixed N.
    if uma_devices > 1 and "UMA_MP_NATOMS" not in env:
        env["UMA_MP_NATOMS"] = str(int(atoms.get_global_number_of_atoms()))
    el = " ".join(symbols)
    pair_style = uma_pair_style_line(precision, uma_devices)

    if _uma_use_kokkos():
        kk_note = (
            f"argv: lmp -k on g {ngpus} -sf kk (single MPI rank). "
            f"pair_style: {pair_style} (LibTorch MP + NCCL when devices>1)."
        )
    else:
        kk_note = (
            f"argv: lmp (no Kokkos; UMA_USE_KOKKOS=0). "
            f"pair_style: {pair_style} (LibTorch MP + NCCL when devices>1)."
        )

    inp_sp = work / "in.sp"
    inp_sp.write_text(
        f"""units metal
atom_style atomic
boundary p p p
newton off
read_data {data.name}
{pair_style}
pair_coeff * * {artifact} {el}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo 1
thermo_style custom step pe fmax fnorm
dump 1 all custom 1 {dump.name} id type x y z fx fy fz
dump_modify 1 sort id
run 0
print "Final PE = $(pe)"
"""
    )
    cmd_sp = uma_kk_argv(
        ngpus, "-in", inp_sp.name, "-log", log_sp.name, uma_devices=uma_devices
    )
    (work / "cmd.sp.txt").write_text(" ".join(cmd_sp) + "\n")
    try:
        _run_lmp(cmd_sp, cwd=work, env=env, tag="sp")
    except RuntimeError as exc:
        raise RuntimeError(f"uma/kk {precision} SP (g={ngpus}) failed:\n{exc}") from exc

    e = None
    for line in log_sp.read_text().splitlines():
        if line.startswith("Final PE"):
            e = float(line.split("=")[1].strip())
    text = dump.read_text().split("ITEM: ATOMS")[-1].strip().splitlines()
    rows = []
    for line in text[1:]:
        parts = line.split()
        if len(parts) >= 8:
            rows.append(
                [float(parts[0]), float(parts[5]), float(parts[6]), float(parts[7])]
            )
    rows = np.array(rows)
    f = rows[np.argsort(rows[:, 0]), 1:4]

    inp_nve = work / "in.nve"
    inp_nve.write_text(
        f"""units metal
atom_style atomic
boundary p p p
newton off
read_data {data.name}
{pair_style}
pair_coeff * * {artifact} {el}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
velocity all create 0.0 1
timestep 0.001
fix 1 all nve
thermo 0
run 0
run {n_timing_steps}
"""
    )
    cmd_nve = uma_kk_argv(
        ngpus, "-in", inp_nve.name, "-log", log_nve.name, uma_devices=uma_devices
    )
    (work / "cmd.nve.txt").write_text(" ".join(cmd_nve) + "\n")
    nve_ok = True
    nve_err = None
    try:
        _run_lmp(cmd_nve, cwd=work, env=env, tag="nve")
    except RuntimeError as exc:
        nve_ok = False
        nve_err = str(exc)
        # Under SLURM, ms_per_eval comes from wall-clock; keep SP E/F for parity.
        if os.environ.get("USE_SLURM_TIMING", "0") != "1":
            raise RuntimeError(
                f"uma/kk {precision} NVE (g={ngpus}) failed:\n{exc}"
            ) from exc
        print(
            f"WARN: uma/kk {precision} NVE (g={ngpus}) failed; "
            "continuing with SP forces (USE_SLURM_TIMING=1):\n"
            f"{exc}",
            flush=True,
        )

    pair_s = None
    if nve_ok and log_nve.is_file():
        for line in log_nve.read_text().splitlines():
            if line.strip().startswith("Pair") and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                try:
                    v = float(parts[2])
                    if v > 0:
                        pair_s = v
                except ValueError:
                    pass
    ms = (pair_s / n_timing_steps) * 1e3 if pair_s else None
    return _pack(
        f"uma/kk precision {precision}",
        e,
        f,
        ms,
        pair_section_s=pair_s,
        lmp_uma=str(find_uma_lmp_binary()),
        artifact=str(artifact),
        kokkos_gpus=ngpus,
        uma_devices=uma_devices,
        pair_style=pair_style,
        multi_gpu_note=kk_note,
        nve_ok=nve_ok,
        nve_error=None if nve_ok else (nve_err or "")[:500],
    )


def main() -> int:
    ngpus = parse_ngpus()
    uma_devices = parse_uma_devices(ngpus)
    n_timing = int(os.environ.get("N_TIMING", "5"))
    paths = parse_only_paths()
    ckpt = find_checkpoint()

    atoms = load_nacl6_fixed()
    meta = geometry_meta(atoms)
    Z = atoms.get_atomic_numbers()
    uniq = sorted(set(int(z) for z in Z))
    symbols = [chemical_symbols[z] for z in uniq]
    title = "nacl6_rattle_fixed multi-gpu compare"

    if os.environ.get("RESULTS_DIR"):
        out_dir = Path(os.environ["RESULTS_DIR"]).expanduser().resolve()
    else:
        out_dir = RESULTS / f"ngpu{ngpus}"

    # Baseline for uma vs devices=1: sibling ngpu1 under same results parent.
    results_parent = out_dir.parent
    uma_d1_baseline = load_uma_d1_baseline(results_parent)

    subset = set(paths) != set(ALL_PATHS)
    merge_existing = subset or os.environ.get("MERGE_RESULTS", "0") == "1"
    prior_report = None
    prior_forces = {}
    if merge_existing and (out_dir / "parity.json").is_file():
        prior_report = json.loads((out_dir / "parity.json").read_text())
        print(f"merge: keeping existing {out_dir / 'parity.json'}", flush=True)
        if (out_dir / "forces.npz").is_file():
            old = np.load(out_dir / "forces.npz")
            for k in old.files:
                if k.startswith("forces_"):
                    prior_forces[k[len("forces_") :]] = old[k]
    elif out_dir.exists() and not merge_existing:
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = out_dir / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log(f"gpu_name={gpu_name}")
    log(f"torch.cuda.device_count()={n_visible}  NGPUS={ngpus}  UMA_DEVICES={uma_devices}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    if n_visible < ngpus:
        raise SystemExit(
            f"Need >= {ngpus} visible CUDA devices, got torch.cuda.device_count()={n_visible}"
        )
    log(f"checkpoint={ckpt}")
    log(f"engine={ENGINE}")
    log(f"lammps_root={LAMMPS_ROOT}")
    log(f"LMP_UMA={find_uma_lmp_binary()}")
    log(f"LMP_FC={find_fairchem_lmp_binary()}")
    log(f"paths={paths}")
    log(f"geometry={meta['source']}  natoms={meta['natoms']}")
    log(f"ART_F64={ART_F64} exists={(ART_F64 / 'model_traced.pt').is_file()}")
    log(f"ART_MIX={ART_MIX} exists={(ART_MIX / 'model_traced.pt').is_file()}")
    log(
        "uma/kk recipe: single MPI rank, "
        f"`lmp -k on g {ngpus} -sf kk`, "
        f"`pair_style uma/kk ... devices {uma_devices}`"
    )
    if uma_d1_baseline:
        log(f"uma devices=1 baseline loaded: {sorted(uma_d1_baseline)}")
    elif uma_devices > 1:
        log("warning: no ngpu1 uma baseline yet — d1 parity metrics will be null")
    if os.environ.get("UMA_KK_LAUNCH"):
        log(f"UMA_KK_LAUNCH={os.environ['UMA_KK_LAUNCH']}")
    if os.environ.get("FAIRCHEM_WORKERS"):
        log(f"FAIRCHEM_WORKERS={os.environ['FAIRCHEM_WORKERS']}")

    results: dict[str, dict] = {}
    forces: dict[str, np.ndarray] = {}
    energies: dict[str, float] = {}

    # Mixed GP (devices>1) uses FairChem eager float32 — traced uma mixed @ devices=1
    # disagrees by ~0.058 eV on NaCl6 while ASE FairChem float32@1 matches GP mixed.
    # Retarget mixed oracle accordingly; double still gates vs uma devices=1 traced.
    if uma_devices > 1 and "uma_mixed" in paths:
        log(
            "mixed GP oracle: ASE FairChem float32 workers=1 "
            "(not traced uma_mixed @ ngpu1; see gp_round/f32_diag.json)"
        )
        ase_f32 = run_ase_dtype(atoms, ckpt, dtype="float32", workers=1, n_timing=1)
        uma_d1_baseline["uma_mixed"] = {
            "energy_eV": ase_f32["energy_eV"],
            "forces": ase_f32["forces"],
            "uma_devices": 1,
            "oracle": "ase_fairchem_float32_workers1",
        }
        forces["ase_mixed_oracle"] = ase_f32["forces"]
        energies["ase_mixed_oracle"] = ase_f32["energy_eV"]
        log(
            f"mixed_oracle E={ase_f32['energy_eV']:.10f}  "
            f"{ase_f32['ms_per_eval']:.1f} ms  (ASE float32 w1)"
        )
        release_cuda("post-mixed-oracle")

    if "ase" in paths:
        ase = run_ase(atoms, ckpt, n_timing, ngpus)
        results["ase"] = ase
        forces["ase"] = ase["forces"]
        energies["ase"] = ase["energy_eV"]
        log(
            f"ASE   E={ase['energy_eV']:.10f}  "
            f"{ase['ms_per_eval']:.1f} ms  workers={ase.get('workers')}"
        )
        release_cuda("pre-next")

    if "fc" in paths:
        (work_root / "fc").mkdir(parents=True, exist_ok=True)
        fc = run_fairchem_lammps(
            atoms, ckpt, n_timing, work_root / "fc", title, ngpus
        )
        results["fc"] = fc
        forces["fc"] = fc["forces"]
        energies["fc"] = fc["energy_eV"]
        log(
            f"FC    E={fc['energy_eV']:.10f}  "
            f"{fc['ms_per_eval']:.1f} ms  workers={fc.get('workers')}"
        )
        release_cuda("pre-next")

    if "uma_double" in paths:
        (work_root / "uma_double").mkdir(parents=True, exist_ok=True)
        uma64 = run_uma_kk(
            atoms,
            precision="double",
            artifact=ART_F64,
            symbols=symbols,
            n_timing_steps=n_timing,
            work=work_root / "uma_double",
            title=title,
            ngpus=ngpus,
            uma_devices=uma_devices,
        )
        results["uma_double"] = uma64
        forces["uma_double"] = uma64["forces"]
        energies["uma_double"] = uma64["energy_eV"]
        log(
            f"uma64 E={uma64['energy_eV']:.10f}  "
            f"{uma64['ms_per_eval']} ms  kokkos_g={ngpus} devices={uma_devices}"
        )
        release_cuda("pre-next")

    if "uma_mixed" in paths:
        (work_root / "uma_mixed").mkdir(parents=True, exist_ok=True)
        umamix = run_uma_kk(
            atoms,
            precision="mixed",
            artifact=ART_MIX,
            symbols=symbols,
            n_timing_steps=n_timing,
            work=work_root / "uma_mixed",
            title=title,
            ngpus=ngpus,
            uma_devices=uma_devices,
        )
        results["uma_mixed"] = umamix
        forces["uma_mixed"] = umamix["forces"]
        energies["uma_mixed"] = umamix["energy_eV"]
        log(
            f"mixed E={umamix['energy_eV']:.10f}  "
            f"{umamix['ms_per_eval']} ms  kokkos_g={ngpus} devices={uma_devices}"
        )
        release_cuda("post-mixed")

    f_ref = forces.get("ase")
    e_ref = energies.get("ase")
    if f_ref is None and prior_report is not None:
        # Recover ASE reference from merged prior for dE / force metrics.
        for pr in prior_report.get("rows") or []:
            if pr.get("key") == "ase" or "ASE" in str(pr.get("path", "")):
                e_ref = pr.get("energy_eV", e_ref)
                break
        if "ase" in prior_forces:
            f_ref = prior_forces["ase"]

    # When ngpu=1 and uma paths ran, they become the devices=1 baseline for siblings.
    if ngpus == 1:
        for uk in ("uma_double", "uma_mixed"):
            if uk in results:
                uma_d1_baseline[uk] = {
                    "energy_eV": results[uk]["energy_eV"],
                    "forces": results[uk]["forces"],
                    "uma_devices": uma_devices,
                }

    gate_rows: list[dict] = []
    rows = []
    for key in paths:
        r = results[key]
        row = {
            "path": r["path"],
            "key": key,
            "energy_eV": r["energy_eV"],
            "ms_per_eval": r["ms_per_eval"],
        }
        for k in (
            "fairchem_lmp",
            "lmp_uma",
            "artifact",
            "note",
            "load_s",
            "pair_section_s",
            "workers",
            "kokkos_gpus",
            "uma_devices",
            "pair_style",
            "multi_gpu_note",
            "execution_mode",
            "merge_mole",
            "ms_per_eval_python",
        ):
            if k in r:
                row[k] = r[k]
        if f_ref is None or key == "ase":
            row.update(
                abs_dE_vs_ase_f64=0.0 if key == "ase" else None,
                force_mae=0.0 if key == "ase" else None,
                force_rmse=0.0 if key == "ase" else None,
                force_max_abs=0.0 if key == "ase" else None,
                force_max_norm_per_atom=0.0 if key == "ase" else None,
                cosine=1.0 if key == "ase" else None,
            )
            if key == "ase" and f_ref is not None:
                row["f_ref_max_abs"] = float(np.max(np.abs(f_ref)))
                row["f_ref_rms"] = float(np.sqrt(np.mean(f_ref**2)))
        else:
            row["abs_dE_vs_ase_f64"] = (
                abs(r["energy_eV"] - e_ref) if e_ref is not None else None
            )
            row.update(force_stats(f_ref, r["forces"]))

        # uma vs devices=1 baseline (primary GP parity gate)
        if key in PRECISION_BY_KEY:
            bl = uma_d1_baseline.get(key)
            if bl is not None and uma_devices != 1:
                d1_stats = force_stats(bl["forces"], r["forces"])
                row["abs_dE_vs_uma_d1"] = abs(r["energy_eV"] - bl["energy_eV"])
                row["force_mae_vs_uma_d1"] = d1_stats["force_mae"]
                row["force_rmse_vs_uma_d1"] = d1_stats["force_rmse"]
                row["force_max_abs_vs_uma_d1"] = d1_stats["force_max_abs"]
                row["cosine_vs_uma_d1"] = d1_stats["cosine"]
                if bl.get("oracle"):
                    row["d1_oracle"] = bl["oracle"]
                row["parity_gate"] = check_gate(
                    key,
                    abs_dE=row["abs_dE_vs_uma_d1"],
                    force_max_abs=row["force_max_abs_vs_uma_d1"],
                    cosine=row["cosine_vs_uma_d1"],
                )
                if bl.get("oracle"):
                    row["parity_gate"]["oracle"] = bl["oracle"]
                gate_rows.append(row["parity_gate"])
            elif uma_devices == 1:
                row["abs_dE_vs_uma_d1"] = 0.0
                row["force_max_abs_vs_uma_d1"] = 0.0
                row["cosine_vs_uma_d1"] = 1.0
                row["parity_gate"] = {
                    "applicable": True,
                    "precision": PRECISION_BY_KEY[key],
                    "passed": True,
                    "note": "devices=1 baseline (self)",
                }
            else:
                row["parity_gate"] = {
                    "applicable": False,
                    "note": "no ngpu1 uma baseline",
                }

        rows.append(row)
        log(
            f"  {key}: E={row['energy_eV']:.10f}  ms={row['ms_per_eval']}  "
            f"dE_ase={row.get('abs_dE_vs_ase_f64')}  "
            f"dE_d1={row.get('abs_dE_vs_uma_d1')}  "
            f"force_max_d1={row.get('force_max_abs_vs_uma_d1')}"
        )

    # Merge prior rows for paths not re-run this job.
    if prior_report is not None:
        by_key = {r.get("key") or r.get("path"): r for r in rows}
        merged = []
        seen = set()
        for pr in prior_report.get("rows") or []:
            k = pr.get("key")
            if k in by_key:
                merged.append(by_key[k])
                seen.add(k)
            elif k not in paths:
                merged.append(pr)
                seen.add(k)
        for r in rows:
            if r.get("key") not in seen:
                merged.append(r)
        rows = merged
        for k, farr in prior_forces.items():
            forces.setdefault(k, farr)
        for pr in prior_report.get("rows") or []:
            k = pr.get("key")
            if k and k not in energies and "energy_eV" in pr:
                energies[k] = pr["energy_eV"]

    npz_kwargs = {
        "numbers": atoms.get_atomic_numbers().astype(np.int32),
        "positions": atoms.get_positions().astype(np.float64),
        "cell": atoms.cell.array.astype(np.float64),
        "ngpus": np.array(ngpus, dtype=np.int32),
    }
    for key, e in energies.items():
        npz_kwargs[f"energy_{key}_eV"] = np.array(e)
    for key, f in forces.items():
        npz_kwargs[f"forces_{key}"] = f

    forces_path = out_dir / "forces.npz"
    np.savez(forces_path, **npz_kwargs)

    report = {
        "ngpus": ngpus,
        "uma_devices": uma_devices,
        "n_timing": n_timing,
        "gpu_name": gpu_name,
        "torch_cuda_device_count": n_visible,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint": str(ckpt),
        "reference": "ASE FAIRChemCalculator FP64",
        "uma_d1_reference": "uma/kk devices=1 @ ngpu1 (same precision)",
        "lmp_uma": str(find_uma_lmp_binary()),
        "lmp_fc": str(find_fairchem_lmp_binary()),
        "artifacts": {"double": str(ART_F64), "mixed": str(ART_MIX)},
        "geometry": meta,
        "fp64_policy": (
            "ASE and uma/kk double use FP64. FairChem FC may build cell in FP32. "
            "uma/kk mixed is an explicit separate path."
        ),
        "multi_gpu": {
            "ase_fc": "load_predict_unit(..., workers=NGPUS) → Ray ParallelMLIPPredictUnit",
            "uma_kk": (
                f"single MPI rank: lmp -k on g {ngpus} -sf kk; "
                f"pair_style uma/kk ... devices {uma_devices}; "
                "no mpirun domain decomp"
            ),
        },
        "parity_gates_summary": summarize_gates(gate_rows),
        "paths_run": paths,
        "paths_merged": merge_existing,
        "rows": rows,
    }
    parity_path = out_dir / "parity.json"
    parity_path.write_text(json.dumps(report, indent=2) + "\n")

    log_path = out_dir / "run.log"
    # Append on merge so prior log is preserved.
    if merge_existing and log_path.is_file():
        prev = log_path.read_text()
        log_path.write_text(prev + "\n--- merge rerun ---\n" + "\n".join(log_lines) + "\n")
    else:
        log_path.write_text("\n".join(log_lines) + "\n")
    log(f"wrote {parity_path}")
    log(f"wrote {forces_path}")
    log(f"wrote {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
