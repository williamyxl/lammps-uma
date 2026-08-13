#!/usr/bin/env python3
"""Minimal cross-node NCCL probe (no UMA/FairChem). Sweeps all_reduce and
reduce_scatter over increasing sizes to find where/if cross-node collectives
hang. Launch with mpiexec; reads PMI_* for rank/world."""
from __future__ import annotations
import os, sys, time
import torch
import torch.distributed as dist

rank = int(os.environ.get("PMI_RANK", "0"))
world = int(os.environ.get("PMI_SIZE", "1"))
local = int(os.environ.get("PMI_LOCAL_RANK", "0"))
os.environ.setdefault("RANK", str(rank)); os.environ.setdefault("WORLD_SIZE", str(world))
torch.cuda.set_device(local % max(1, torch.cuda.device_count()))
dev = torch.device("cuda")

def log(*a):
    if rank == 0:
        print(*a, flush=True)

t0 = time.time()
dist.init_process_group("nccl", rank=rank, world_size=world,
                        timeout=__import__("datetime").timedelta(seconds=60))
log(f"[init ok] world={world} in {time.time()-t0:.2f}s")

# element counts up to the UMA GP reduce_scatter size (~85M)
for nel in [1<<10, 1<<16, 1<<20, 1<<23, 1<<25, 90_000_000]:
    # all_reduce
    x = torch.ones(nel, dtype=torch.float64, device=dev)
    torch.cuda.synchronize(); t=time.time()
    try:
        dist.all_reduce(x); torch.cuda.synchronize()
        log(f"all_reduce  nel={nel:>10d} ({nel*8/1e6:7.1f} MB)  {(time.time()-t)*1e3:8.1f} ms  sum0={x[0].item():.0f}")
    except Exception as e:
        log(f"all_reduce  nel={nel} FAILED: {e}"); break
    # reduce_scatter (the op that hung)
    if nel % world == 0:
        inp = torch.ones(nel, dtype=torch.float64, device=dev)
        out = torch.empty(nel//world, dtype=torch.float64, device=dev)
        torch.cuda.synchronize(); t=time.time()
        try:
            dist.reduce_scatter_tensor(out, inp); torch.cuda.synchronize()
            log(f"reduce_scat nel={nel:>10d} ({nel*8/1e6:7.1f} MB)  {(time.time()-t)*1e3:8.1f} ms")
        except Exception as e:
            log(f"reduce_scat nel={nel} FAILED: {e}"); break

log("[PROBE DONE]")
dist.barrier(); dist.destroy_process_group()
