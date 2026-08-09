#!/usr/bin/env python3
"""Shared helpers for water888 single-path timed runs (post-warmup only)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

EX = Path(__file__).resolve().parent
ML_UMA = EX.parents[1]  # .../ML-UMA
ENGINE = ML_UMA / "uma-engine"
ROOT = EX.parents[3]  # .../lammps-uma

DEFAULT_DATA = EX / "water_nvt_300K_atomic_metal.data"
DEFAULT_CKPT = Path(
    os.environ.get(
        "UMA_CHECKPOINT",
        "/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt",
    )
)
ART_F64 = Path(
    os.environ.get(
        "UMA_ARTIFACT_DIR",
        str(ENGINE / "artifacts" / "uma-s-1p2-omat-f64"),
    )
)


def out_dir(path_key: str) -> Path:
    jobid = os.environ.get("SLURM_JOB_ID", "manual")
    ngpus = int(os.environ.get("NGPUS", os.environ.get("FAIRCHEM_WORKERS", "1")))
    d = EX / "results" / f"{path_key}_ngpu{ngpus}_{jobid}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fp64_settings(*, workers: int, external_graph: bool):
    import sys

    sys.path.insert(0, str(ENGINE / "python"))
    from common import inference_settings_with_dtype

    settings = inference_settings_with_dtype("float64")
    settings.external_graph_gen = external_graph
    settings.activation_checkpointing = False
    settings.execution_mode = "general"
    return settings


def find_uma_lmp() -> Path:
    env = os.environ.get("LMP_UMA")
    if env:
        return Path(env).expanduser().resolve()
    cand = ROOT / "build-uma" / "lmp"
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError("set LMP_UMA or build build-uma/lmp")


def find_fc_lmp() -> Path:
    env = os.environ.get("LMP_FC")
    if env:
        return Path(env).expanduser().resolve()
    conda = Path(os.environ.get("CONDA_PREFIX", ""))
    for c in (
        conda / "bin" / "lmp",
        Path("/u/xyan11/miniforge3-x86_64/envs/uma312/bin/lmp"),
    ):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError("FairChem lmp not found; set LMP_FC")


def setup_ld_path(*, natoms: int | None = None, uma_devices: int = 1) -> dict:
    env = os.environ.copy()
    import torch

    vesin = ENGINE / "third_party" / "vesin" / "lib"
    torch_lib = Path(torch.__path__[0]) / "lib"
    parts = [str(vesin), str(torch_lib), "/usr/local/cuda/lib64"]
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    env["UMA_FORBID_RAY_GP"] = "1"
    if uma_devices > 1:
        worker = env.get("UMA_LIBTORCH_MP_WORKER")
        if not worker:
            cand = ENGINE / "build-cpp-mp" / "uma_libtorch_mp_worker"
            if cand.is_file():
                env["UMA_LIBTORCH_MP_WORKER"] = str(cand)
        if natoms is not None:
            env["UMA_MP_NATOMS"] = str(int(natoms))
            env["UMA_STRUCTURE_NATOMS"] = str(int(natoms))
        env.setdefault("UMA_PEER_TRANSPORT", "nccl")
    return env


def parse_lammps_run_blocks(log_text: str) -> list[dict]:
    """Parse each 'Loop time … for N steps' + following Pair row."""
    blocks: list[dict] = []
    loop_re = re.compile(
        r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps"
    )
    pair_re = re.compile(
        r"^Pair\s+\|\s+([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)",
        re.M,
    )
    for m in loop_re.finditer(log_text):
        start = m.end()
        # Pair table appears after this Loop; take first Pair after it, before next Loop
        nxt = loop_re.search(log_text, start)
        chunk = log_text[start : nxt.start() if nxt else len(log_text)]
        pm = pair_re.search(chunk)
        pair_avg = float(pm.group(1)) if pm else None
        nsteps = int(m.group(3))
        loop_s = float(m.group(1))
        blocks.append(
            {
                "loop_s": loop_s,
                "nsteps": nsteps,
                "pair_s": pair_avg,
                "loop_ms_per_step": (loop_s / nsteps) * 1e3 if nsteps else None,
                "pair_ms_per_step": (pair_avg / nsteps) * 1e3
                if pair_avg is not None and nsteps
                else None,
            }
        )
    return blocks


def write_timing(
    out: Path,
    payload: dict,
    forces: np.ndarray | None = None,
) -> None:
    """Write timing.json; optional first-frame forces.npz only."""
    out.mkdir(parents=True, exist_ok=True)
    timing = {k: v for k, v in payload.items() if k != "forces"}
    (out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    if forces is not None and "energy_eV" in payload:
        np.savez(
            out / "forces.npz",
            forces=np.asarray(forces, dtype=np.float64),
            energy_eV=np.array(payload["energy_eV"], dtype=np.float64),
        )
        print(f"wrote {out / 'forces.npz'} shape={tuple(np.asarray(forces).shape)}")
    print(json.dumps(timing, indent=2))
    print(f"wrote {out / 'timing.json'}")


def parse_lammps_dump_frames(dump_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse multi-frame dump with columns: id type x y z fx fy fz (or id fx fy fz).

    Returns forces (nframes, natoms, 3) sorted by atom id per frame, and
    atom ids (natoms,) from the first frame.
    """
    text = dump_path.read_text()
    chunks = text.split("ITEM: ATOMS")
    frames = []
    ids0 = None
    for chunk in chunks[1:]:
        lines = chunk.strip().splitlines()
        if not lines:
            continue
        header = lines[0].split()
        # remaining lines until next ITEM or empty
        body = []
        for line in lines[1:]:
            if line.startswith("ITEM:"):
                break
            if line.strip():
                body.append(line.split())
        if not body:
            continue
        # locate id / fx fy fz columns from header
        # header line is like "id type x y z fx fy fz" after ITEM: ATOMS
        cols = header
        try:
            i_id = cols.index("id")
            i_fx = cols.index("fx")
            i_fy = cols.index("fy")
            i_fz = cols.index("fz")
        except ValueError:
            # fallback: id ... fx fy fz at end
            i_id, i_fx, i_fy, i_fz = 0, -3, -2, -1
        rows = np.array(
            [[float(p[i_id]), float(p[i_fx]), float(p[i_fy]), float(p[i_fz])] for p in body]
        )
        order = np.argsort(rows[:, 0])
        rows = rows[order]
        if ids0 is None:
            ids0 = rows[:, 0]
        frames.append(rows[:, 1:4])
    if not frames:
        raise RuntimeError(f"no frames in {dump_path}")
    return np.stack(frames, axis=0), ids0


def parse_thermo_pe_after_marker(log_text: str, marker: str, nsteps: int) -> np.ndarray:
    """Collect the next ``nsteps`` thermo PE values after a marker line.

    Expects ``thermo_style custom step temp pe ...`` (pe in column index 2).
    """
    lines = log_text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if marker in line:
            start = i + 1
            break
    pes = []
    for line in lines[start:]:
        s = line.strip()
        if not s or s.startswith("Loop") or s.startswith("Performance") or s.startswith("Pair"):
            if len(pes) >= nsteps:
                break
            continue
        parts = s.split()
        # thermo numeric line: step temp pe ...
        try:
            step = int(float(parts[0]))
            pe = float(parts[2])
        except (ValueError, IndexError):
            continue
        if step == 0 and not pes:
            continue  # skip run-boundary step 0 if present
        pes.append(pe)
        if len(pes) >= nsteps:
            break
    if len(pes) < nsteps:
        raise RuntimeError(
            f"expected {nsteps} thermo PE after {marker!r}, got {len(pes)}"
        )
    return np.asarray(pes[:nsteps], dtype=np.float64)
