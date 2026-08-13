#!/usr/bin/env python3
"""Multi-GPU eager UMA worker using REAL torch.distributed + FairChem gp_utils
(graph parallel), driven by the C++ GraphParallelRuntime stdin/binary protocol.

Why: the host-staged kgp SharedGatherSlot emulation produced partial per-atom
forces (~13% low), while real torch.distributed GP is bit-exact vs serial
(dF ~6e-16, verified). This worker therefore uses the real collectives.

Design: the process the C++ parent spawns is GP rank 0. It forks ranks 1..W-1
as children that join the same torch.distributed group (MASTER_ADDR/PORT via a
private file store). Rank 0 speaks the pipe protocol, broadcasts geometry to all
ranks, every rank runs the eager model collectively (activation_checkpointing
optional), rank 0 returns E + full per-atom forces.

Protocol (unchanged from uma_gp_worker.py):
  ready:   {"ok":true,"ready":true}
  init:    {"cmd":"init","checkpoint":..,"workers":W,"dtype":..,"task":..,
            "activation_checkpointing":bool}  -> {"ok":true,...}
  predict: {"cmd":"predict","n":N,...} then binary pos(f64 n*3) z(i32 n)
            cell(f64 9) pbc(i32 3); reply {"ok":true,"energy":E,"n":N} + f64 n*3
  shutdown:{"cmd":"shutdown"}
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Private protocol stream on a dup of fd1; redirect fd1 -> stderr so library logs
# never corrupt the C++ pipe.
_PROTO_FD = os.dup(1)
os.dup2(2, 1)
_PROTO = os.fdopen(_PROTO_FD, "wb", buffering=0)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402


def _write_json(obj: dict) -> None:
    _PROTO.write((json.dumps(obj, separators=(",", ":")) + "\n").encode())
    _PROTO.flush()


def _read_exact(n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        c = sys.stdin.buffer.read(n - len(buf))
        if not c:
            raise EOFError(f"stdin closed {len(buf)}/{n}")
        buf.extend(c)
    return bytes(buf)


def _build_predictor(checkpoint, dtype, task, ckpt_flag, rank):
    ENG = Path(__file__).resolve().parent
    sys.path.insert(0, str(ENG))
    from common import inference_settings_with_dtype
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    s = inference_settings_with_dtype(dtype)
    s.external_graph_gen = False  # GP path uses internal graph gen
    s.activation_checkpointing = bool(ckpt_flag)
    s.execution_mode = "general"
    s.merge_mole = False
    pred = load_predict_unit(checkpoint, device="cuda", inference_settings=s, workers=1)
    return FAIRChemCalculator(pred, task_name=task)


def _rank_worker(rank, world, store_path, checkpoint, dtype, task, ckpt_flag, conn):
    """Ranks 1..W-1: join the group, loop on broadcasts, run model collectively."""
    try:
        torch.cuda.set_device(0)  # each child pinned to its GPU via CUDA_VISIBLE_DEVICES
    except Exception:
        pass
    store = dist.FileStore(store_path, world)
    dist.init_process_group(backend="nccl", store=store, rank=rank, world_size=world)
    from fairchem.core.common import gp_utils
    gp_utils.setup_graph_parallel_groups(world, "nccl")
    from ase import Atoms
    calc = _build_predictor(checkpoint, dtype, task, ckpt_flag, rank)
    if conn is not None:
        conn.send({"ok": True, "rank": rank})
    while True:
        # geometry is broadcast from rank 0 as a CUDA tensor blob
        stop = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(stop, src=0)
        if int(stop.item()) != 0:
            break
        meta = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=0)
        n = int(meta.item())
        pos = torch.zeros(n, 3, dtype=torch.float64, device="cuda")
        z = torch.zeros(n, dtype=torch.int64, device="cuda")
        cell = torch.zeros(3, 3, dtype=torch.float64, device="cuda")
        dist.broadcast(pos, src=0); dist.broadcast(z, src=0); dist.broadcast(cell, src=0)
        atoms = Atoms(numbers=z.cpu().numpy(), positions=pos.cpu().numpy(),
                      cell=cell.cpu().numpy(), pbc=[True, True, True])
        atoms.calc = calc
        _ = float(atoms.get_potential_energy()); _ = atoms.get_forces()  # collective


def main() -> int:
    ready_sent = False
    _write_json({"ok": True, "ready": True, "pid": os.getpid()})
    ready_sent = True

    state = {"calc": None, "world": 1, "children": [], "store": None}

    def do_init(msg):
        world = int(msg.get("workers", 1))
        dtype = str(msg.get("dtype", "float64"))
        task = str(msg.get("task", "omat"))
        ckpt = str(msg["checkpoint"])
        ckpt_flag = bool(msg.get("activation_checkpointing", False))
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("no CUDA device visible to rank 0")
        store_path = tempfile.mktemp(prefix="uma_dist_store_")
        state["store"] = store_path
        # spawn ranks 1..W-1 (each inherits its own CUDA_VISIBLE_DEVICES via the
        # affinity wrapper only in MPI launch; here single-node fork -> we set it)
        ctx = mp.get_context("spawn")
        for r in range(1, world):
            child_env_gpu = r  # rank r uses GPU r on this node
            parent, child = ctx.Pipe(duplex=True)
            p = ctx.Process(target=_child_entry,
                            args=(r, world, store_path, ckpt, dtype, task, ckpt_flag,
                                  child, child_env_gpu),
                            daemon=True, name=f"uma-dist-rank{r}")
            p.start()
            state["children"].append((p, parent))
        # rank 0 on GPU 0
        torch.cuda.set_device(0)
        store = dist.FileStore(store_path, world)
        dist.init_process_group(backend="nccl", store=store, rank=0, world_size=world)
        from fairchem.core.common import gp_utils
        gp_utils.setup_graph_parallel_groups(world, "nccl")
        state["calc"] = _build_predictor(ckpt, dtype, task, ckpt_flag, 0)
        state["world"] = world
        for _, parent in state["children"]:
            r = parent.recv()
            if not r.get("ok"):
                raise RuntimeError(f"rank child failed: {r}")
        return {"ok": True, "backend": "fairchem_dist_gp", "workers": world,
                "activation_checkpointing": ckpt_flag}

    def do_predict(msg):
        from ase import Atoms
        n = int(msg["n"])
        pos = np.frombuffer(_read_exact(n * 3 * 8), dtype="<f8").reshape(n, 3).copy()
        z = np.frombuffer(_read_exact(n * 4), dtype="<i4").astype(np.int64)
        cell = np.frombuffer(_read_exact(9 * 8), dtype="<f8").reshape(3, 3).copy()
        _ = _read_exact(3 * 4)  # pbc (assume periodic)
        world = state["world"]
        if world > 1:
            dist.broadcast(torch.zeros(1, dtype=torch.int64, device="cuda"), src=0)  # stop=0
            dist.broadcast(torch.tensor([n], dtype=torch.int64, device="cuda"), src=0)
            dist.broadcast(torch.tensor(pos, device="cuda"), src=0)
            dist.broadcast(torch.tensor(z, device="cuda"), src=0)
            dist.broadcast(torch.tensor(cell, device="cuda"), src=0)
        atoms = Atoms(numbers=z, positions=pos, cell=cell, pbc=[True, True, True])
        atoms.calc = state["calc"]
        e = float(atoms.get_potential_energy())
        f = np.ascontiguousarray(np.asarray(atoms.get_forces(), dtype=np.float64).reshape(n, 3))
        return {"ok": True, "energy": e, "n": n}, f

    while True:
        raw = sys.stdin.buffer.readline()
        if not raw:
            break
        line = raw.decode("utf-8", "strict").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _write_json({"ok": False, "error": f"bad json: {e}"}); continue
        cmd = msg.get("cmd")
        try:
            if cmd == "init":
                _write_json(do_init(msg))
            elif cmd == "predict":
                meta, f = do_predict(msg)
                _write_json(meta)
                _PROTO.write(np.ascontiguousarray(f, dtype="<f8").tobytes()); _PROTO.flush()
            elif cmd == "shutdown":
                if state["world"] > 1:
                    try:
                        dist.broadcast(torch.ones(1, dtype=torch.int64, device="cuda"), src=0)
                    except Exception:
                        pass
                _write_json({"ok": True, "shutdown": True})
                return 0
            else:
                _write_json({"ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as exc:  # noqa: BLE001
            _write_json({"ok": False, "error": str(exc),
                         "traceback": traceback.format_exc()[-2000:]})
            if cmd in ("init", "predict"):
                return 2
    return 0


def _child_entry(rank, world, store_path, ckpt, dtype, task, ckpt_flag, conn, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch  # re-import in child
    _rank_worker(rank, world, store_path, ckpt, dtype, task, ckpt_flag, conn)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    raise SystemExit(main())
