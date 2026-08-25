#!/usr/bin/env python3
"""Persistent FairChem graph-parallel worker for uma-engine ``devices>1``.

Protocol (dedicated fd — libraries must NOT write there):

  At startup we ``dup`` the original stdout for JSON/binary replies, then
  ``dup2(stderr → stdout)`` so FairChem / Ray / wandb logs cannot corrupt
  the C++ pipe.

  INIT:
    {"cmd":"init","checkpoint":"...","workers":N,"dtype":"float64"|"float32",
     "task":"omat"}
    -> {"ok":true,"backend":"fairchem_eager_python","workers":N,...}

  PREDICT:
    {"cmd":"predict","n":N,"charge":0,"spin":0}
    then little-endian binary: pos[N*3] f64, Z[N] i32, cell[9] f64, pbc[3] i32
    -> {"ok":true,"energy":...}
    then forces[N*3] f64

  SHUTDOWN:
    {"cmd":"shutdown"} -> {"ok":true} then exit 0

NL note: workers>1 forces FairChem internal graph generation
(``external_graph_gen=False``), matching ASE multi-GPU. Parity vs traced
devices=1 is gated on E/F thresholds, not bit-identical edges.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path

# Silence wandb before it can wrap stdout (was causing BrokenPipe on protocol).
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Protocol stream: keep a private dup of the original stdout, then point fd 1
# at stderr so Ray/FairChem INFO lines never land on the C++ pipe.
_PROTO_FD = os.dup(1)
os.dup2(2, 1)  # stdout → stderr
_PROTO = os.fdopen(_PROTO_FD, "wb", buffering=0)

import numpy as np  # noqa: E402  — after stdout redirect


def _write_json(obj: dict) -> None:
    payload = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
    _PROTO.write(payload)
    _PROTO.flush()


def _write_err(msg: str, **extra) -> None:
    _write_json({"ok": False, "error": msg, **extra})


def _read_exact(n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            raise EOFError(f"stdin closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


class WorkerState:
    def __init__(self) -> None:
        self.predictor = None
        self.calc = None
        self.workers = 1
        self.dtype = "float64"
        self.task = "omat"
        self.checkpoint = ""

    def init(
        self,
        checkpoint: str,
        workers: int,
        dtype: str,
        task: str,
        activation_checkpointing: bool = False,
    ) -> dict:
        from ase import Atoms
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core.units.mlip_unit.api.inference import inference_settings_default

        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        if dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be float32|float64, got {dtype}")
        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

        import torch

        # Device selection: Intel XPU (Aurora) > CUDA > CPU. On XPU, apply the
        # hen patches (XPU device allowlist for MLIPPredictUnit + the FP64
        # prepare_wigner edge-chunk fix that restores correct backward forces at
        # large edge counts, NaCl N>=10). See hen/docs/finding_xpu_ag_fd_cliff.
        xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
        if xpu_ok:
            for p in (
                "/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/shim",
                "/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/patches",
                "/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen",
            ):
                if p not in sys.path and Path(p).is_dir():
                    sys.path.insert(0, p)
            from fairchem_xpu_parallel import patch_fairchem_xpu_device
            patch_fairchem_xpu_device()
            device = "xpu"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        settings = inference_settings_default()
        settings.base_precision_dtype = getattr(torch, dtype)
        # Graph-parallel + external_graph trips CUDA index asserts on UMA.
        settings.external_graph_gen = workers <= 1
        # Activation checkpointing (eager only): recompute block activations in
        # backward -> ~3x less activation memory, exact. Cannot be traced, so the
        # eager worker is the way LAMMPS gets this capacity lever.
        settings.activation_checkpointing = bool(activation_checkpointing)
        # Never silent turbo for double / parity paths.
        settings.execution_mode = "general"

        self.predictor = load_predict_unit(
            str(ckpt),
            device=device,
            inference_settings=settings,
            workers=workers,
        )
        # Enforce FP64 module cast on XPU (matches the hen eager path).
        if dtype == "float64":
            for attr in ("model", "module", "_module"):
                mod = getattr(self.predictor, attr, None)
                if mod is not None:
                    mod.double()
                    break
        self.calc = FAIRChemCalculator(self.predictor, task_name=task)
        self._device = device
        self.workers = workers
        self.dtype = dtype
        self.task = task
        self.checkpoint = str(ckpt)
        _ = Atoms
        return {
            "ok": True,
            "backend": "fairchem_eager_python",
            "workers": workers,
            "dtype": dtype,
            "task": task,
            "checkpoint": str(ckpt),
            "external_graph_gen": bool(settings.external_graph_gen),
            "device": device,
            "cuda": bool(torch.cuda.is_available()),
            "xpu": bool(xpu_ok),
            "n_visible": (
                int(torch.xpu.device_count()) if xpu_ok
                else (int(torch.cuda.device_count()) if torch.cuda.is_available() else 0)
            ),
        }

    def predict(
        self,
        n: int,
        charge: int,
        spin: int,
    ) -> tuple[dict, np.ndarray]:
        from ase import Atoms

        if self.calc is None:
            raise RuntimeError("worker not initialized; send init first")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        pos = np.frombuffer(_read_exact(n * 3 * 8), dtype="<f8").reshape(n, 3).copy()
        z = np.frombuffer(_read_exact(n * 4), dtype="<i4").copy()
        cell = np.frombuffer(_read_exact(9 * 8), dtype="<f8").reshape(3, 3).copy()
        pbc_i = np.frombuffer(_read_exact(3 * 4), dtype="<i4")
        pbc = [bool(int(pbc_i[0])), bool(int(pbc_i[1])), bool(int(pbc_i[2]))]

        atoms = Atoms(numbers=z.tolist(), positions=pos, cell=cell, pbc=pbc)
        atoms.info["charge"] = int(charge)
        atoms.info["spin"] = int(spin)
        atoms.calc = self.calc
        if hasattr(self.calc, "results"):
            self.calc.results.clear()

        print("uma_gp_worker: predict start get_potential_energy", file=sys.stderr, flush=True)
        energy = float(atoms.get_potential_energy())
        print(f"uma_gp_worker: energy done E={energy}", file=sys.stderr, flush=True)
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        print(f"uma_gp_worker: forces done shape={forces.shape}", file=sys.stderr, flush=True)
        if forces.shape != (n, 3):
            raise RuntimeError(f"forces shape {forces.shape} != ({n}, 3)")
        return {"ok": True, "energy": energy, "n": n}, forces


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
        force=True,
    )
    state = WorkerState()
    _write_json({"ok": True, "ready": True, "pid": os.getpid()})

    while True:
        # Read the command line in BINARY. Mixing sys.stdin.readline() (text,
        # read-ahead buffered) with sys.stdin.buffer.read() (binary geometry)
        # desyncs the stream: readline() swallows bytes past the newline, so the
        # following binary read misses them. Use buffer.readline() throughout.
        raw = sys.stdin.buffer.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="strict").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_err(f"invalid JSON: {exc}")
            continue

        cmd = msg.get("cmd")
        try:
            if cmd == "init":
                out = state.init(
                    checkpoint=str(msg["checkpoint"]),
                    workers=int(msg.get("workers", 1)),
                    dtype=str(msg.get("dtype", "float64")),
                    task=str(msg.get("task", "omat")),
                    activation_checkpointing=bool(msg.get("activation_checkpointing", False)),
                )
                _write_json(out)
            elif cmd == "predict":
                meta, forces = state.predict(
                    n=int(msg["n"]),
                    charge=int(msg.get("charge", 0)),
                    spin=int(msg.get("spin", 0)),
                )
                _write_json(meta)
                _PROTO.write(np.ascontiguousarray(forces, dtype="<f8").tobytes())
                _PROTO.flush()
            elif cmd == "shutdown":
                _write_json({"ok": True, "shutdown": True})
                return 0
            else:
                _write_err(f"unknown cmd: {cmd!r}")
        except Exception as exc:  # noqa: BLE001 — surface to C++ client
            try:
                _write_err(str(exc), traceback=traceback.format_exc()[-2000:])
            except BrokenPipeError:
                print(f"uma_gp_worker fatal (pipe closed): {exc}", file=sys.stderr)
                return 2
            if cmd in ("init", "predict"):
                return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
