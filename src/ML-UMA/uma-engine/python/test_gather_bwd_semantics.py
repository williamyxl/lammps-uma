#!/usr/bin/env python3
"""Quick CPU-side unit test for gather sum_grad backward semantics (no CUDA)."""
from __future__ import annotations

import torch
import kokkos_gp_runtime as kgp


def test_gather_sum_grad_bwd_matches_manual():
    """Two ranks, each with local energy depending on full gathered x."""
    torch.manual_seed(0)
    natoms = 5
    world = 2
    sizes = kgp.size_list_fn(natoms, world)  # [3,2]
    # Simulate: true full embedding grads from each rank's local loss
    g0_full = torch.randn(natoms, 4, dtype=torch.float64)
    g1_full = torch.randn(natoms, 4, dtype=torch.float64)
    # Correct local input grads = sum of full grads on each slice
    expect0 = (g0_full + g1_full)[: sizes[0]]
    expect1 = (g0_full + g1_full)[sizes[0] : sizes[0] + sizes[1]]

    # Thread simulation of our backward
    kgp.configure(world)
    results = [None, None]

    def run(rank, g_full):
        kgp.set_rank(rank)
        # Mimic _GatherSumGrad.backward
        gathered = kgp._gather_slot.all_gather(rank, g_full.contiguous())
        summed = gathered[0].clone()
        for t in gathered[1:]:
            summed = summed + t
        start = sum(sizes[:rank])
        local = summed.narrow(0, start, sizes[rank])
        results[rank] = local

    import threading

    t0 = threading.Thread(target=run, args=(0, g0_full))
    t1 = threading.Thread(target=run, args=(1, g1_full))
    t0.start(); t1.start(); t0.join(); t1.join()
    assert torch.allclose(results[0], expect0), (results[0], expect0)
    assert torch.allclose(results[1], expect1), (results[1], expect1)
    print("gather_sum_grad_bwd OK")


def test_autograd_end_to_end_thread():
    """Local x -> gather -> use slice of full in loss -> backward."""
    kgp.install_patches(2)
    natoms = 4
    sizes = kgp.size_list_fn(natoms, 2)
    xs = [
        torch.arange(sizes[0], dtype=torch.float64).unsqueeze(1).requires_grad_(True),
        (torch.arange(sizes[1], dtype=torch.float64) + 10).unsqueeze(1).requires_grad_(True),
    ]
    losses = [None, None]
    grads = [None, None]

    def run(rank):
        kgp.set_rank(rank)
        full = kgp.gather_from_model_parallel_region_sum_grad(xs[rank], natoms)
        # local energy depends on ALL full entries (like message passing)
        e = (full ** 2).sum() * (0.3 if rank == 0 else 0.7)
        losses[rank] = e
        e.backward()
        grads[rank] = xs[rank].grad.detach().clone()

    import threading
    # Sequential barrier collectives need concurrent threads
    t0 = threading.Thread(target=run, args=(0,))
    t1 = threading.Thread(target=run, args=(1,))
    t0.start(); t1.start(); t0.join(); t1.join()

    # Manual: E = 0.3*|full|^2 + 0.7*|full|^2 = |full|^2
    # dE/d(full)=2*full; dE/d(x0)=(2*full)[0:2]; etc.
    # But each rank only backprops its own e_r without summing losses!
    # Rank0: e0=0.3*|full|^2, d(e0)/d(x0)=0.3*2*full[0:2]
    # Rank1: e1=0.7*|full|^2, d(e1)/d(x1)=0.7*2*full[2:4]
    # After our sum_grad bwd: d(e0)/d(x0) uses only rank0's grad_full=0.6*full
    # Wait - rank0's backward: grad_full = 0.6*full, all_gather sum with rank1's 1.4*full = 2*full, slice0 = 2*full[0:2]
    # Rank1: similarly 2*full[2:4]
    # So after gather bwd alone, local grads are for TOTAL E already!
    full = torch.cat([xs[0].detach(), xs[1].detach()], dim=0)
    expect0 = 2 * full[: sizes[0]].unsqueeze(1) if False else (2 * full[: sizes[0]]).unsqueeze(1)
    # xs are already [n,1]
    expect0 = (2 * full[: sizes[0]]).reshape(sizes[0], 1)
    expect1 = (2 * full[sizes[0] :]).reshape(sizes[1], 1)
    print("grad0", grads[0].T, "expect", expect0.T)
    print("grad1", grads[1].T, "expect", expect1.T)
    # If sum_grad gives total E grads per slice, then forces all_reduce would DOUBLE
    if torch.allclose(grads[0], expect0) and torch.allclose(grads[1], expect1):
        print("NOTE: gather bwd already yields TOTAL energy grads — post all_reduce would 2x")
    else:
        # Check partial
        expect0_p = (0.3 * 2 * full[: sizes[0]]).reshape(sizes[0], 1)
        expect1_p = (0.7 * 2 * full[sizes[0] :]).reshape(sizes[1], 1)
        print("partial expect0", expect0_p.T, "partial expect1", expect1_p.T)
        if torch.allclose(grads[0], expect0_p):
            print("gather bwd yields PARTIAL (this-rank-only) — post all_reduce needed")
        else:
            print("MISMATCH")
            raise SystemExit(1)


def test_all_reduce_with_grad_identity_bwd_for_process_gp():
    """Process-GP uses identity bwd on all_reduce_with_grad (see kokkos_gp_runtime)."""
    kgp.install_patches(2)
    xs = [
        torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True),
        torch.tensor([10.0, 20.0], dtype=torch.float64, requires_grad=True),
    ]
    grads = [None, None]

    def run(rank):
        kgp.set_rank(rank)
        y = kgp._AllReduceWithGrad.apply(xs[rank])
        e = (y ** 2).sum() * (0.3 if rank == 0 else 0.7)
        e.backward()
        grads[rank] = xs[rank].grad.detach().clone()

    import threading

    t0 = threading.Thread(target=run, args=(0,))
    t1 = threading.Thread(target=run, args=(1,))
    t0.start(); t1.start(); t0.join(); t1.join()
    y = xs[0].detach() + xs[1].detach()
    assert torch.allclose(grads[0], 0.6 * y), grads[0]
    assert torch.allclose(grads[1], 1.4 * y), grads[1]
    print("all_reduce_with_grad identity-bwd OK (process-GP regime)")
    kgp.uninstall_patches()


if __name__ == "__main__":
    test_gather_sum_grad_bwd_matches_manual()
    test_autograd_end_to_end_thread()
    test_all_reduce_with_grad_identity_bwd_for_process_gp()
    print("all OK")
