#!/usr/bin/env python3
"""Tier 1 (b)+(c): edge-padding cap arithmetic and node-partition contract.

Pure Python + numpy. Encodes the invariants the C++ engine and the Python exporter
must agree on (P2.1 / P5'.4 / P0'.3), so a future drift is caught without XPU:

edge_pad_cap (export_blocks_xpu.py):   cap = (E // chunk + 1) * chunk
  -> cap is a multiple of chunk, cap > E always (>= one guard chunk),
     traced chunk count = ceil(cap / chunk) = E//chunk + 1.

node_partition (graph_shard.h):        tensor_split(arange(n), W)[rank]
  == numpy array_split(arange(n), W)[rank]; the W parts must be a disjoint cover
     of range(n) with sizes differing by <= 1.
"""
import sys
from pathlib import Path

import numpy as np


# --- edge_pad_cap contract ----------------------------------------------------
def edge_pad_cap(E, chunk):
    return (E // chunk + 1) * chunk


def test_cap_is_chunk_multiple():
    for chunk in (256, 8192, 16384, 32768, 65536):
        for E in (0, 1, chunk - 1, chunk, chunk + 1, 3 * chunk, 87392, 262144):
            cap = edge_pad_cap(E, chunk)
            assert cap % chunk == 0, (E, chunk, cap)


def test_cap_strictly_exceeds_E():
    for chunk in (256, 32768, 65536):
        for E in (0, 1, chunk - 1, chunk, chunk + 1, 100000):
            assert edge_pad_cap(E, chunk) > E, (E, chunk)


def test_traced_chunk_count_matches_ceil():
    for chunk in (256, 32768, 65536):
        for E in (1, chunk - 1, chunk, chunk + 1, 200000):
            cap = edge_pad_cap(E, chunk)
            assert cap // chunk == E // chunk + 1
            # ceil(cap/chunk) == cap/chunk exactly since cap is a multiple
            assert -(-cap // chunk) == cap // chunk


def test_guard_chunk_present_at_boundary():
    # exactly on a chunk boundary must still add a full guard chunk (drift safety)
    chunk = 32768
    assert edge_pad_cap(chunk, chunk) == 2 * chunk
    assert edge_pad_cap(2 * chunk, chunk) == 3 * chunk


# --- node_partition contract --------------------------------------------------
def node_partition(n, world, rank):
    # torch.tensor_split(arange(n), world)[rank] == np.array_split(arange(n), world)[rank]
    return np.array_split(np.arange(n), world)[rank]


def test_partition_disjoint_cover():
    for n in (0, 1, 3, 7, 16, 4096, 32768):
        for world in (1, 2, 3, 4, 6, 8, 12):
            parts = [node_partition(n, world, r) for r in range(world)]
            cat = np.concatenate(parts) if parts else np.array([], dtype=int)
            # cover
            assert np.array_equal(np.sort(cat), np.arange(n)), (n, world)
            # disjoint
            assert len(cat) == n == len(set(cat.tolist())), (n, world)


def test_partition_balanced():
    for n in (16, 4096, 32768, 262144):
        for world in (2, 3, 4, 6, 8, 12):
            sizes = [len(node_partition(n, world, r)) for r in range(world)]
            assert max(sizes) - min(sizes) <= 1, (n, world, sizes)
            assert sum(sizes) == n


def test_partition_matches_tensor_split_semantics():
    # first n%world parts get ceil, rest floor (the tensor_split rule)
    n, world = 32768, 12
    q, r = divmod(n, world)
    for rank in range(world):
        expect = q + (1 if rank < r else 0)
        assert len(node_partition(n, world, rank)) == expect


def test_world_gt_natoms_gives_empty_shards():
    # world > n_atoms: some ranks own zero nodes (P0.4 empty-shard case)
    n, world = 3, 8
    sizes = [len(node_partition(n, world, r)) for r in range(world)]
    assert sum(sizes) == n
    assert sizes.count(0) == world - n


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    n_fail = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception:
            n_fail += 1; print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{len(tests)-n_fail}/{len(tests)} passed")
    sys.exit(1 if n_fail else 0)
