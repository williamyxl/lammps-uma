#!/usr/bin/env python3
"""Native same-node multi-GPU UMA worker (no FairChem Ray).

Protocol matches ``uma_gp_worker.py`` (JSON + binary over dedicated stdout dup).

Backend: N *processes* × MLIPPredictUnit on cuda:0..N-1 with ``kokkos_gp_runtime``
host-staged peer collectives (SharedGatherSlot) — avoids threaded CUDA
``lazy wrapper`` / autograd races. No Ray, no c10d.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")

_PROTO_FD = os.dup(1)
os.dup2(2, 1)
_PROTO = os.fdopen(_PROTO_FD, "wb", buffering=0)

import numpy as np  # noqa: E402


def _write_json(obj: dict) -> None:
    _PROTO.write((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
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


def _rank_process_main(
    rank: int,
    world: int,
    checkpoint: str,
    dtype: str,
    task: str,
    conn,
    gather_slot,
    reduce_slot,
) -> None:
    """Persistent GPU rank process (spawn)."""
    import torch
    from ase import Atoms
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import inference_settings_default

    from common import atoms_to_atomic_data
    import kokkos_gp_runtime as kgp

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    kgp.install_patches(world, gather_slot=gather_slot, reduce_slot=reduce_slot)
    kgp.set_rank(rank)

    settings = inference_settings_default()
    settings.base_precision_dtype = getattr(torch, dtype)
    settings.external_graph_gen = False
    settings.activation_checkpointing = False
    settings.execution_mode = "general"

    with torch.cuda.device(rank):
        unit = load_predict_unit(
            checkpoint,
            device="cuda",
            inference_settings=settings,
            workers=1,
        )
        target = torch.device(f"cuda:{rank}")
        unit.device = target
        if hasattr(unit, "model") and unit.model is not None:
            unit.model.to(target)
        torch.zeros(1, device=target, dtype=getattr(torch, dtype))

    kgp.install_patches(world, gather_slot=gather_slot, reduce_slot=reduce_slot)
    kgp.set_rank(rank)
    conn.send({"ok": True, "rank": rank, "skip_force_reduce": os.environ.get("UMA_SKIP_FORCE_GP_REDUCE", "0")})

    while True:
        msg = conn.recv()
        if msg is None or msg.get("cmd") == "shutdown":
            conn.send({"ok": True})
            break
        if msg.get("cmd") != "predict":
            conn.send({"ok": False, "error": f"unknown cmd {msg}"})
            continue
        try:
            kgp.set_rank(rank)
            n = int(msg["n"])
            pos = np.frombuffer(msg["pos"], dtype="<f8").reshape(n, 3).copy()
            z = np.frombuffer(msg["z"], dtype="<i4").copy()
            cell = np.frombuffer(msg["cell"], dtype="<f8").reshape(3, 3).copy()
            pbc = [bool(x) for x in msg["pbc"]]
            atoms = Atoms(numbers=z.tolist(), positions=pos, cell=cell, pbc=pbc)
            atoms.info["charge"] = int(msg.get("charge", 0))
            atoms.info["spin"] = int(msg.get("spin", 0))
            data = atoms_to_atomic_data(atoms, task_name=task, settings=settings)
            with torch.cuda.device(rank):
                data = data.clone().to(f"cuda:{rank}")
                out = unit.predict(data, undo_element_references=True)
            energy = out["energy"]
            forces = out["forces"]
            if isinstance(energy, torch.Tensor):
                e = float(energy.detach().cpu().reshape(-1)[0].item())
            else:
                e = float(energy)
            if isinstance(forces, torch.Tensor):
                f = forces.detach().cpu().to(torch.float64).numpy()
            else:
                f = np.asarray(forces, dtype=np.float64)
            f = np.ascontiguousarray(f.reshape(n, 3), dtype=np.float64)
            # All ranks must finish for collectives; only rank0 payload used by parent.
            conn.send({"ok": True, "energy": e, "forces": f.tobytes(order="C"), "n": n})
        except Exception as exc:  # noqa: BLE001
            conn.send(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc()[-2000:],
                }
            )


class NativeWorkerState:
    def __init__(self) -> None:
        self.workers = 1
        self.dtype = "float64"
        self.task = "omat"
        self.checkpoint = ""
        self._ctx = None
        self._manager = None
        self._procs: list = []
        self._parents: list = []
        self._gather = None
        self._reduce = None

    def init(self, checkpoint: str, workers: int, dtype: str, task: str) -> dict:
        import torch
        import torch.multiprocessing as mp

        import kokkos_gp_runtime as kgp

        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        if dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be float32|float64, got {dtype}")
        if dtype != "float64":
            logging.warning("UMA policy is FP64-only; got dtype=%s", dtype)
        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        if not torch.cuda.is_available() or torch.cuda.device_count() < workers:
            raise RuntimeError(
                f"need {workers} CUDA devices; available="
                f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}"
            )

        # Spawn context — each rank is a process (thread CUDA is unsafe for UMA).
        try:
            self._ctx = mp.get_context("spawn")
        except RuntimeError:
            self._ctx = mp
        self._manager = self._ctx.Manager()
        self._gather = kgp.SharedGatherSlot(workers, self._manager)
        self._reduce = kgp.SharedGatherSlot(workers, self._manager)

        self._procs = []
        self._parents = []
        for r in range(workers):
            parent, child = self._ctx.Pipe(duplex=True)
            p = self._ctx.Process(
                target=_rank_process_main,
                args=(
                    r,
                    workers,
                    str(ckpt),
                    dtype,
                    task,
                    child,
                    self._gather,
                    self._reduce,
                ),
                name=f"uma-gp-rank{r}",
                daemon=True,
            )
            p.start()
            child.close()
            ready = parent.recv()
            if not ready.get("ok"):
                raise RuntimeError(f"rank {r} init failed: {ready}")
            self._parents.append(parent)
            self._procs.append(p)

        self.workers = workers
        self.dtype = dtype
        self.task = task
        self.checkpoint = str(ckpt)
        return {
            "ok": True,
            "backend": "kokkos_peer_process_gp",
            "workers": workers,
            "dtype": dtype,
            "task": task,
            "checkpoint": str(ckpt),
        }

    def predict(self, n: int, charge: int, spin: int) -> tuple[float, np.ndarray]:
        pos = _read_exact(n * 3 * 8)
        z = _read_exact(n * 4)
        cell = _read_exact(9 * 8)
        pbc_i = np.frombuffer(_read_exact(3 * 4), dtype="<i4")
        pbc = [bool(int(pbc_i[0])), bool(int(pbc_i[1])), bool(int(pbc_i[2]))]

        msg = {
            "cmd": "predict",
            "n": n,
            "charge": int(charge),
            "spin": int(spin),
            "pos": pos,
            "z": z,
            "cell": cell,
            "pbc": pbc,
        }
        for parent in self._parents:
            parent.send(msg)

        replies = [parent.recv() for parent in self._parents]
        for r, rep in enumerate(replies):
            if not rep.get("ok"):
                raise RuntimeError(
                    f"native GP rank {r} failed: {rep.get('error')}\n"
                    f"{rep.get('traceback', '')}"
                )

        rep0 = replies[0]
        e = float(rep0["energy"])
        f = np.frombuffer(rep0["forces"], dtype="<f8").reshape(n, 3).copy()
        return e, np.ascontiguousarray(f, dtype=np.float64)

    def shutdown(self) -> None:
        for parent in self._parents:
            try:
                parent.send({"cmd": "shutdown"})
                parent.recv()
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        self._procs = []
        self._parents = []
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _write_json({"ok": True, "ready": True, "backend": "kokkos_peer_process_gp"})
    state = NativeWorkerState()
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        try:
            msg = json.loads(line.decode("utf-8"))
        except Exception as exc:
            _write_err(f"bad json: {exc}")
            continue
        cmd = msg.get("cmd")
        try:
            if cmd == "init":
                resp = state.init(
                    msg["checkpoint"],
                    int(msg["workers"]),
                    msg.get("dtype", "float64"),
                    msg.get("task", "omat"),
                )
                _write_json(resp)
            elif cmd == "predict":
                e, f = state.predict(
                    int(msg["n"]), int(msg.get("charge", 0)), int(msg.get("spin", 0))
                )
                _write_json({"ok": True, "energy": e})
                _PROTO.write(f.tobytes(order="C"))
                _PROTO.flush()
            elif cmd == "shutdown":
                state.shutdown()
                _write_json({"ok": True})
                return 0
            else:
                _write_err(f"unknown cmd: {cmd}")
        except Exception as exc:
            _write_err(str(exc), traceback=traceback.format_exc()[-2000:])
    state.shutdown()
    return 0


if __name__ == "__main__":
    # Required for CUDA + spawn under fork-from-LAMMPS parent.
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    sys.exit(main())
