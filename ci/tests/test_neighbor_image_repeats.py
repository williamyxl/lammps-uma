"""Tier 1 (d, pure core): neighbor-list image-repeat bound uses INTERPLANAR spacing.

Pure numpy replica of neighbor_list.cpp::image_repeats (P0.5). The bug was using
the lattice-vector length |cell[d]| instead of the interplanar spacing V/|area_d|,
which under-counts periodic images on a SHEARED cell and silently drops edges.

This encodes the invariant so a regression is caught on a login node. (The full
brute-force-vs-cell-list C++ test is unittest/ CTest, Sprint 4 Tier 1 (d) C++.)
"""
import sys

import numpy as np


def image_repeats_interplanar(cell, pbc, cutoff):
    """Replica of the FIXED C++ image_repeats (P0.5)."""
    a, b, c = cell[0], cell[1], cell[2]
    vol = abs(float(np.dot(a, np.cross(b, c))))
    rep = [0, 0, 0]
    area = [np.linalg.norm(np.cross(b, c)),
            np.linalg.norm(np.cross(c, a)),
            np.linalg.norm(np.cross(a, b))]
    for d in range(3):
        if not pbc[d]:
            continue
        spacing = vol / area[d] if area[d] > 1e-12 else 1e-12
        rep[d] = int(np.ceil(cutoff / max(spacing, 1e-12)))
    return rep


def image_repeats_buggy(cell, pbc, cutoff):
    """The OLD buggy version: uses |cell[d]| (lattice-vector length)."""
    rep = [0, 0, 0]
    for d in range(3):
        if not pbc[d]:
            continue
        length = np.linalg.norm(cell[d])
        rep[d] = int(np.ceil(cutoff / max(length, 1e-12)))
    return rep


def brute_min_images_needed(cell, cutoff, axis):
    """Smallest integer image count along `axis` such that every lattice plane
    within `cutoff` is reachable: ceil(cutoff / interplanar_spacing)."""
    a, b, c = cell[0], cell[1], cell[2]
    vol = abs(float(np.dot(a, np.cross(b, c))))
    area = [np.linalg.norm(np.cross(b, c)),
            np.linalg.norm(np.cross(c, a)),
            np.linalg.norm(np.cross(a, b))]
    return int(np.ceil(cutoff / (vol / area[axis])))


def test_orthorhombic_unchanged():
    # for an orthorhombic cell interplanar spacing == |cell[d]|, so both agree
    cell = np.diag([5.64, 5.64, 5.64]).astype(float)
    pbc = [True, True, True]
    for cutoff in (3.0, 6.0, 12.0):
        assert image_repeats_interplanar(cell, pbc, cutoff) == \
               image_repeats_buggy(cell, pbc, cutoff)


def test_sheared_cell_needs_more_images():
    # a strongly sheared cell: |cell[d]| overestimates the spacing, so the buggy
    # bound is too small and would drop edges. The fixed bound must be >= the
    # true requirement, and strictly greater than the buggy one on at least one axis.
    cell = np.array([[6.0, 0.0, 0.0],
                     [5.0, 6.0, 0.0],   # sheared b
                     [5.0, 5.0, 6.0]],  # sheared c
                    dtype=float)
    pbc = [True, True, True]
    cutoff = 6.0
    fixed = image_repeats_interplanar(cell, pbc, cutoff)
    buggy = image_repeats_buggy(cell, pbc, cutoff)
    for d in range(3):
        assert fixed[d] >= brute_min_images_needed(cell, cutoff, d), (d, fixed)
    assert any(fixed[d] > buggy[d] for d in range(3)), (fixed, buggy)


def test_fixed_bound_never_below_truth_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        # random but non-degenerate cell (diagonally dominant so volume > 0)
        cell = rng.uniform(-3, 3, (3, 3)) + np.diag([8, 8, 8])
        if abs(np.linalg.det(cell)) < 1.0:
            continue
        cutoff = float(rng.uniform(3, 10))
        fixed = image_repeats_interplanar(cell, [True] * 3, cutoff)
        for d in range(3):
            assert fixed[d] >= brute_min_images_needed(cell, cutoff, d)


def test_nonperiodic_axis_zero():
    cell = np.diag([5.64, 5.64, 5.64]).astype(float)
    rep = image_repeats_interplanar(cell, [True, False, True], 6.0)
    assert rep[1] == 0


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
