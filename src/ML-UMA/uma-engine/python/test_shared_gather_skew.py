#!/usr/bin/env python3
"""MP smoke: consecutive all_reduces with rank skew (SharedGatherSlot gens)."""
from __future__ import annotations

import time

import torch
import torch.multiprocessing as mp

import kokkos_gp_runtime as kgp


def worker(rank: int, world: int, gather, niter: int, queue) -> None:
    kgp.configure(world, gather_slot=gather, reduce_slot=gather)
    kgp.set_rank(rank)
    try:
        for i in range(niter):
            if rank == 1:
                time.sleep(0.002)
            x = torch.full((4,), float(rank + 1 + i * 10), dtype=torch.float64)
            y = kgp.peer_all_reduce_sum(x)
            expect = 3.0 + i * 20
            if not torch.allclose(y, torch.full((4,), expect, dtype=torch.float64)):
                queue.put((rank, i, y.tolist(), expect))
                return
        queue.put((rank, "ok"))
    except Exception as e:  # noqa: BLE001
        queue.put((rank, "err", str(e)))


def main() -> None:
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    gather = kgp.SharedGatherSlot(2, mgr)
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, 2, gather, 50, q)) for r in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=120)
    results = [q.get(timeout=5) for _ in range(2)]
    print(results)
    assert all(r[1] == "ok" for r in results), results
    print("skewed consecutive all_reduce OK")


if __name__ == "__main__":
    main()
