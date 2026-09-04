#!/usr/bin/env python3
"""H-9 / H15 (audit PART H, rev 31): multi-node DD contract gate — the login-node,
no-torch, no-XPU half of the missing multi-node test coverage.

PART H found the DD path has ZERO automated tests, so its invariants were only
ever exercised by PBS jobs. This pins, in pure Python mirroring the C++, the
arithmetic/algebraic contracts that a regression would silently break:

  1. halo forward/reverse ADJOINTNESS: <S x, y> == <x, Sᵀ y> for the
     owner->ghost scatter S (forward) and the ghost->owner accumulate Sᵀ
     (reverse_exchange). This is THE property forces depend on; if it breaks,
     gradients are silently wrong. (halo_context.cpp forward_exchange/reverse.)
  2. pad_dd_edges arithmetic, INCLUDING the nall==0 case (H5): pad edges are
     pad_nbr -> dummy, never dummy->dummy (r=0), and pad_nbr != dummy always.
  3. pack_forward/unpack_forward and pack_reverse/unpack_reverse are transposes
     (the reverse ACCUMULATES), so a full comm round trip on owned+ghost is the
     identity on owners plus the ghost sum on their owners.
  4. H3 contract: the no_halo control must gate BOTH forward and backward (its
     adjoint is the identity, not Sᵀ) — else the A/B diagnostic is unsound.

Plain python3 (assert-based) and pytest-collectable. No torch, no numpy, no MPI.
"""
import sys


def _fail(m):
    print(f"FAIL {m}"); return False


# ---- model of the halo scatter S and its transpose Sᵀ -------------------------
# A "system" of N nodes split into owners [0,nl) and ghosts [nl,N). Each ghost g
# is a copy of some owner owner_of[g] (owner->ghost scatter). This is exactly what
# LAMMPS forward_comm does to the halo buffer; reverse_comm is its transpose.
def forward_scatter(x, nl, owner_of):
    """S: owners unchanged; each ghost row := its owner row. (forward_exchange)"""
    y = list(x)
    for g in range(nl, len(x)):
        y[g] = x[owner_of[g]]
    return y


def reverse_accumulate(g_in, nl, owner_of):
    """Sᵀ: add each ghost's value onto its owner, then zero ghosts.
    Mirrors reverse_exchange (reverse_comm ADDS ghost->owner; ghosts zeroed)."""
    out = list(g_in)
    for g in range(nl, len(g_in)):
        out[owner_of[g]] += g_in[g]
        out[g] = 0.0
    return out


def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def test_halo_adjointness():
    # N=6: owners 0..2, ghosts 3..5 copying owners 0,1,0.
    nl = 3
    owner_of = {3: 0, 4: 1, 5: 0}
    N = 6
    x = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]      # arbitrary owner features; ghosts set by S
    y = [0.5, -1.0, 2.0, 4.0, -3.0, 1.5]    # arbitrary cotangent
    lhs = _dot(forward_scatter(x, nl, owner_of), y)     # <S x, y>
    rhs = _dot(x, reverse_accumulate(y, nl, owner_of))  # <x, Sᵀ y>
    assert abs(lhs - rhs) < 1e-12, (lhs, rhs)
    print("PASS test_halo_adjointness  <Sx,y>=<x,Sᵀy>")


def test_reverse_zeroes_ghosts():
    nl = 2
    owner_of = {2: 0, 3: 1}
    g = [0.0, 0.0, 5.0, 7.0]
    out = reverse_accumulate(g, nl, owner_of)
    assert out[0] == 5.0 and out[1] == 7.0          # delivered to owners
    assert out[2] == 0.0 and out[3] == 0.0          # ghosts zeroed (no double count)
    print("PASS test_reverse_zeroes_ghosts")


# ---- pad_dd_edges model (C++ PairUMA::pad_dd_edges + setup_dd_pad_nodes) -------
def setup_pad_nodes(nall):
    """Returns (nnodes, dummy, pad_nbr) mirroring setup_dd_pad_nodes (H5)."""
    no_real = (nall == 0)
    extra = 1 if no_real else 0
    nnodes = nall + 1 + extra
    dummy = nall
    pad_nbr = (nall + 1) if no_real else 0
    return nnodes, dummy, pad_nbr


def pad_edges(real_row0, real_row1, edge_cap, dummy, pad_nbr):
    E = len(real_row0)
    assert E <= edge_cap
    row0 = list(real_row0) + [pad_nbr] * (edge_cap - E)
    row1 = list(real_row1) + [dummy] * (edge_cap - E)
    return row0, row1


def test_pad_edges_normal():
    nall = 5
    nnodes, dummy, pad_nbr = setup_pad_nodes(nall)
    assert (nnodes, dummy, pad_nbr) == (6, 5, 0)     # +1 dummy; neighbor = atom 0
    row0, row1 = pad_edges([1, 2], [0, 1], 4, dummy, pad_nbr)
    for k in range(2, 4):
        assert row0[k] == 0 and row1[k] == dummy     # atom0 -> dummy (inert)
        assert not (row0[k] == dummy and row1[k] == dummy)   # never dummy->dummy
    print("PASS test_pad_edges_normal")


def test_pad_edges_zero_atom_rank_H5():
    # The H5 bug: nall==0 -> dummy==0; a naive pad neighbor 0 == dummy -> r=0.
    nall = 0
    nnodes, dummy, pad_nbr = setup_pad_nodes(nall)
    assert (nnodes, dummy, pad_nbr) == (2, 0, 1)     # dummy + a DISTINCT far nbr
    assert pad_nbr != dummy, "H5: pad neighbor must differ from dummy (avoid r=0)"
    row0, row1 = pad_edges([], [], 3, dummy, pad_nbr)
    for k in range(3):
        assert row0[k] == pad_nbr and row1[k] == dummy
        assert row0[k] != row1[k], "H5 regression: dummy->dummy self-loop (r=0)"
    print("PASS test_pad_edges_zero_atom_rank_H5")


# ---- pack/unpack round trip (pack_forward/unpack_forward, pack/unpack_reverse) -
def comm_round_trip(x, nl, owner_of, per_node):
    """Full forward then reverse over a per_node-wide buffer, mirroring the
    LAMMPS pack/unpack pairs used by the halo. Forward fills ghosts from owners;
    reverse accumulates ghosts onto owners and zeroes ghosts."""
    N = len(x)
    buf = [[0.0] * per_node for _ in range(N)]
    for i in range(nl):
        for k in range(per_node):
            buf[i][k] = x[i] * (k + 1)          # arbitrary per-channel content
    # forward: ghost <- owner
    for g in range(nl, N):
        buf[g] = list(buf[owner_of[g]])
    # reverse: owner += ghost; ghost := 0
    for g in range(nl, N):
        for k in range(per_node):
            buf[owner_of[g]][k] += buf[g][k]
        buf[g] = [0.0] * per_node
    return buf


def test_pack_unpack_round_trip():
    nl = 2
    owner_of = {2: 0, 3: 0, 4: 1}            # atom 0 has two ghosts, atom 1 one
    x = [1.0, 10.0, 0.0, 0.0, 0.0]
    per_node = 3
    buf = comm_round_trip(x, nl, owner_of, per_node)
    # owner 0: base + 2 ghost copies = 3x; owner 1: base + 1 = 2x
    for k in range(per_node):
        assert abs(buf[0][k] - 3 * 1.0 * (k + 1)) < 1e-12, buf[0]
        assert abs(buf[1][k] - 2 * 10.0 * (k + 1)) < 1e-12, buf[1]
    for g in range(nl, len(x)):
        assert all(v == 0.0 for v in buf[g]), "ghost rows must be zeroed"
    print("PASS test_pack_unpack_round_trip")


# ---- H3 no_halo control contract ----------------------------------------------
def halo_forward(x, nl, owner_of, no_halo):
    return list(x) if no_halo else forward_scatter(x, nl, owner_of)


def halo_backward(g, nl, owner_of, no_halo):
    # H3: adjoint of identity is identity, NOT Sᵀ.
    return list(g) if no_halo else reverse_accumulate(g, nl, owner_of)


def test_no_halo_control_is_self_adjoint():
    nl = 2
    owner_of = {2: 0, 3: 1}
    x = [1.0, 2.0, 0.0, 0.0]
    y = [0.3, 0.7, 1.1, -0.4]
    # With no_halo, forward is identity; its adjoint MUST also be identity.
    lhs = _dot(halo_forward(x, nl, owner_of, True), y)
    rhs = _dot(x, halo_backward(y, nl, owner_of, True))
    assert abs(lhs - rhs) < 1e-12, (lhs, rhs)
    # And it must be a TRUE identity, not Sᵀ (the H3 bug used Sᵀ in backward):
    assert halo_backward(y, nl, owner_of, True) == y, "H3: no_halo backward must be identity"
    print("PASS test_no_halo_control_is_self_adjoint")


# ---- H4: DD ghost-shell depth requirement (init_style_dd, audit rev 32) --------
def required_shell(num_layers, dd_k, cutoff):
    """Replica of the req_shell rule: a per-layer halo (dd_k>=num_layers) needs
    1*cutoff; a shallower one needs num_layers*cutoff. num_layers<=0 -> 1*cutoff
    (legacy artifact, warn-only)."""
    if dd_k > 0 and dd_k >= num_layers:
        return cutoff
    return num_layers * cutoff if num_layers > 0 else cutoff


def shell_ok(have_shell, num_layers, dd_k, cutoff):
    """True if the configured shell is deep enough (or cannot be checked)."""
    if num_layers <= 0:
        return True                      # legacy artifact: warn, don't block
    if have_shell <= 0.0:
        return True                      # user set nothing: not an error here
    return have_shell + 1e-9 >= required_shell(num_layers, dd_k, cutoff)


def test_shell_depth_per_layer_k4():
    # shipped k=4 per-layer artifact: dd_k==num_layers==4 -> 1*cutoff suffices
    assert required_shell(4, 4, 6.0) == 6.0
    assert shell_ok(6.5, 4, 4, 6.0)          # 6.5 >= 6.0 OK
    assert shell_ok(0.0, 4, 4, 6.0)          # unset -> not blocked here
    print("PASS test_shell_depth_per_layer_k4")


def test_shell_depth_k1_needs_deep_halo():
    # a k=1 artifact (single exchange) needs the full num_layers*cutoff = 24 A
    assert required_shell(4, 1, 6.0) == 24.0
    assert not shell_ok(6.5, 4, 1, 6.0), "6.5 A shell must be rejected for k=1"
    assert shell_ok(24.0, 4, 1, 6.0)
    print("PASS test_shell_depth_k1_needs_deep_halo")


def test_shell_depth_legacy_metadata_not_blocked():
    # num_layers absent (0): cannot verify -> warn, never block
    assert shell_ok(6.5, 0, 0, 6.0)
    print("PASS test_shell_depth_legacy_metadata_not_blocked")


# ---- H6: DD flag agreement (dd_flag_agreement, audit rev 32) -------------------
def flags_agree(per_rank_flags):
    """per_rank_flags: list of [no_halo, halo_test, edge_cap] per rank. Returns
    True iff every column is identical across ranks (else the run would deadlock)."""
    if not per_rank_flags:
        return True
    ncol = len(per_rank_flags[0])
    for c in range(ncol):
        col = [r[c] for r in per_rank_flags]
        if min(col) != max(col):
            return False
    return True


def test_flag_agreement_all_equal():
    assert flags_agree([[0, 0, 917504], [0, 0, 917504], [0, 0, 917504]])
    print("PASS test_flag_agreement_all_equal")


def test_flag_agreement_detects_no_halo_mismatch():
    # rank 1 set UMA_DD_NO_HALO=1, others did not -> mismatched collective counts
    assert not flags_agree([[0, 0, 917504], [1, 0, 917504]])
    print("PASS test_flag_agreement_detects_no_halo_mismatch")


def test_flag_agreement_detects_cap_mismatch():
    assert not flags_agree([[0, 0, 917504], [0, 0, 1376256]])
    print("PASS test_flag_agreement_detects_cap_mismatch")


def main():
    tests = [
        test_halo_adjointness,
        test_reverse_zeroes_ghosts,
        test_pad_edges_normal,
        test_pad_edges_zero_atom_rank_H5,
        test_pack_unpack_round_trip,
        test_no_halo_control_is_self_adjoint,
        test_shell_depth_per_layer_k4,
        test_shell_depth_k1_needs_deep_halo,
        test_shell_depth_legacy_metadata_not_blocked,
        test_flag_agreement_all_equal,
        test_flag_agreement_detects_no_halo_mismatch,
        test_flag_agreement_detects_cap_mismatch,
    ]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
