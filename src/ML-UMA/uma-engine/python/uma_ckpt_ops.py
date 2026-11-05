"""Per-block activation-checkpoint ops for traceable UMA export (option h).

torch.utils.checkpoint does not survive torch.jit.trace, so the traced module
otherwise retains ALL message-passing blocks' activations -> OOM at large N.

Design: the escn block loop is rewritten (at export time) to call
    x = torch.ops.uma_ckpt.block(i, x, edge_distance_vec, edge_distance,
                                 atomic_numbers, edge_index, sys_node_embedding)
per block. The top module records these calls. Each escn block is ALSO exported
as its own TorchScript sub-module (model_block_{i}.pt). At runtime the C++ engine
registers the sub-modules and implements uma_ckpt::block by running block i under
CheckpointModuleFn (forward under no_grad; recompute in backward) -> only ONE
block's activations are live at a time, reproducing eager AC's memory profile in
the pure-traced (no-Python) path.

MEMORY FIX (design-doc option i, 2026-08-23): the block sub-module now takes the
SMALL precursor edge tensors edge_distance_vec [E,3] (~34 MB) and edge_distance
[E], plus atomic_numbers [natoms] and edge_index [2,E] (int, small), and
RECOMPUTES wigner / wigner_inv_envelope / x_edge INTERNALLY. The C++ checkpoint
therefore saves for backward only node-sized x + these small precursors (NOT the
6.5 GiB wigner / wigner_inv_env / ~4 GiB x_edge), recomputing the big edge
tensors in backward and freeing them after — matching the eager-AC memory
profile (~62 GiB OOM avoided).

MEMORY FIX (design-doc option j', 2026-08-24): even option-i recompute built the
FULL-edge wigner [E,25,25]=6.5 GiB + wigner_inv 6.5 GiB + x_edge for ALL edges
and THEN split into chunks — a ~13 GiB transient per block -> OOM at N=18. Now
the wigner / wigner_inv_env / x_edge recompute is moved ENTIRELY into the CHUNK
module and done PER CHUNK from the CHUNK's edge precursors: the block splits ONLY
the small precursors (edge_distance_vec [Ec,3], edge_distance [Ec], edge_index
[2,Ec]) and calls uma_ckpt.chunk per chunk; the chunk module builds only its own
[Ec,25,25] wigner (~0.07 GiB @ Ec=16384). No edge-FULL tensor is ever built at
block level or op level.

Export-time kernel is a single-process stand-in: it just calls the registered
Python block module directly (so tracing records the block's ops into the block
sub-module, and the top graph records the uma_ckpt.block call). C++ replaces the
kernel at runtime.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch.library import Library, impl

_LIB: Library | None = None
_BLOCKS: list = []          # python callables (block forwards) for export tracing
_CHUNKS: list = []          # python callables (per-chunk forwards) for export tracing
_EDGEDEG: Callable | None = None  # single per-chunk edge_degree forward for tracing
_INVARIANTS: dict = {}      # loop-invariant tensors captured at export


def _ensure_lib() -> Library:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = Library("uma_ckpt", "DEF")
    # block(idx, x, edge_distance_vec, edge_distance, atomic_numbers, edge_index,
    #       sys_node_emb) -> x  (option i: small precursors saved, wigner
    #       recomputed inside the block).
    lib.define(
        "block(int idx, Tensor x, Tensor edge_distance_vec, "
        "Tensor edge_distance, Tensor atomic_numbers, Tensor edge_index, "
        "Tensor sys_node_emb) -> Tensor"
    )

    @impl(lib, "block", "CompositeExplicitAutograd")
    def _block_impl(idx: int, x, edge_distance_vec, edge_distance,
                    atomic_numbers, edge_index, sys_node_emb):
        # Export stand-in: call the real block forward directly so its ops trace
        # (into the separately-exported block module). Runtime C++ overrides this.
        fn = _BLOCKS[int(idx)]
        return fn(x, edge_distance_vec, edge_distance, atomic_numbers,
                  edge_index, sys_node_emb)

    # --- option (j') per-CHUNK activation checkpointing (per-chunk wigner) --
    # chunk(block_idx, x_full, edge_distance_vec, edge_distance, atomic_numbers,
    #       edge_index, node_offset, mole_start, natoms) -> partial
    #       ([natoms, ...] node-sized)
    #
    # One call wraps ONE edge chunk. The precursor tensors are the CHUNK's SMALL
    # precursors:
    #   x_full            [natoms, sph, C]  NODE-sized (== x for single tile;
    #                     ~0.05 GiB, fine to save)
    #   edge_distance_vec [Ec, 3]   fp64   CHUNK precursor
    #   edge_distance     [Ec]      fp64   CHUNK precursor
    #   atomic_numbers    [natoms]  int64  full node set (small)
    #   edge_index        [2, Ec]   int64  CHUNK
    # The chunk MODULE recomputes THIS chunk's wigner / wigner_inv_env / x_edge
    # INTERNALLY from these precursors (from _get_rotmat_and_wigner +
    # prepare_wigner + envelope + embeddings), so ONLY a [Ec,25,25] wigner
    # (~0.07 GiB @ Ec=16384) ever exists — the FULL-edge [E,25,25] wigner
    # (6.5 GiB) / wigner_inv_env (6.5 GiB) / x_edge (~4 GiB) are NEVER built at
    # any point (block-level or op-level). At runtime the C++ engine wraps each
    # chunk call in a CheckpointModuleFn so only ONE chunk's SO2 intermediates +
    # its own [Ec,25,25] wigner are live at a time (eager-AC profile in the
    # pure-traced / no-Python path). node_offset / mole_start / natoms are ints
    # (mole_start = running edge count for MoLE ac_start_idx; natoms =
    # x_original_shape for the edge->node scatter).
    lib.define(
        "chunk(int block_idx, Tensor x_full, Tensor edge_distance_vec, "
        "Tensor edge_distance, Tensor atomic_numbers, Tensor edge_index, "
        "int node_offset, int mole_start, int natoms) -> Tensor"
    )

    @impl(lib, "chunk", "CompositeExplicitAutograd")
    def _chunk_impl(block_idx: int, x_full, edge_distance_vec, edge_distance,
                    atomic_numbers, edge_index, node_offset: int,
                    mole_start: int, natoms: int):
        # Export stand-in: call the registered per-block chunk callable, which
        # recomputes this chunk's wigner/x_edge then runs
        # set_mole_ac_start_index + Edgewise.forward_chunk. Its ops trace into
        # the block's chunk sub-module (model_chunk_{i}.pt). Runtime C++
        # overrides this with a per-chunk CheckpointModuleFn.
        fn = _CHUNKS[int(block_idx)]
        return fn(x_full, edge_distance_vec, edge_distance, atomic_numbers,
                  edge_index, int(node_offset), int(mole_start), int(natoms))

    # --- P1-b per-CHUNK checkpointed EDGE_DEGREE PROLOGUE (per-chunk wigner) --
    # edge_degree(x, edge_distance_vec, edge_distance, atomic_numbers,
    #             edge_index, node_offset, mole_start, natoms) -> x
    #
    # Checkpoints the LAST un-checkpointed full-edge transient: the edge-degree
    # embedding prologue (export_blocks_xpu.py old lines 482-505). The monolithic
    # prologue built the FULL-edge wigner [E,25,25]=6.5 GiB + wigner_inv 6.5 GiB
    # + wigner_inv_env + x_edge for ALL edges and ran EdgeDegreeEmbedding
    # (index_add_ scatter, IndexAddBackward0 = 12.82 GiB at N=18) with grad ON,
    # outside every checkpoint -> THE N=18 blocker (dead-weight fix got within
    # 0.26 GiB). This op mirrors uma_ckpt::chunk EXACTLY: the top forward splits
    # ONLY the small per-chunk precursors (edge_distance_vec [Ec,3],
    # edge_distance [Ec], edge_index [2,Ec]) and calls edge_degree per chunk;
    # the ONE edgedeg chunk module recomputes THIS chunk's wigner/wigner_inv_env
    # /x_edge internally (only [Ec,25,25] wigner ~0.07 GiB ever built) then runs
    # EdgeDegreeEmbedding.forward_chunk, which ACCUMULATES its scatter INTO x
    # (running accumulation over the [natoms,...] x). The top forward passes the
    # running x back in and reassigns, so only ONE [natoms,...] accumulator +
    # one chunk's transients are live. At runtime the C++ engine wraps each
    # edge_degree call in a CheckpointModuleFn so no full-edge prologue transient
    # is ever built (eager-AC profile in the pure-traced / no-Python path).
    #   x                 [natoms, sph, C]  NODE-sized accumulator (returned
    #                     updated; forward_chunk accumulates into it)
    #   edge_distance_vec [Ec, 3]   fp64   CHUNK precursor
    #   edge_distance     [Ec]      fp64   CHUNK precursor
    #   atomic_numbers    [natoms]  int64  full node set (atomic_numbers_full)
    #   edge_index        [2, Ec]   int64  CHUNK
    #   node_offset       int   (gp_node_offset; 0 for single-tile)
    #   mole_start        int   (running edge count for MoLE ac_start_idx;
    #                            advances by chunk edge count)
    #   natoms            int   (== x.shape[0], the edge->node scatter size)
    lib.define(
        "edge_degree(Tensor x, Tensor edge_distance_vec, "
        "Tensor edge_distance, Tensor atomic_numbers, Tensor edge_index, "
        "int node_offset, int mole_start, int natoms) -> Tensor"
    )

    @impl(lib, "edge_degree", "CompositeExplicitAutograd")
    def _edge_degree_impl(x, edge_distance_vec, edge_distance, atomic_numbers,
                          edge_index, node_offset: int, mole_start: int,
                          natoms: int):
        # Export stand-in: call the registered edge_degree callable, which
        # recomputes this chunk's wigner/wigner_inv_env/x_edge then runs
        # EdgeDegreeEmbedding.forward_chunk (accumulates into x). Its ops trace
        # into model_edgedeg_chunk.pt (ONE module, not per-block). Runtime C++
        # overrides this with a per-chunk CheckpointModuleFn.
        fn = _EDGEDEG
        return fn(x, edge_distance_vec, edge_distance, atomic_numbers,
                  edge_index, int(node_offset), int(mole_start), int(natoms))

    _LIB = lib
    return lib


def install_ckpt_ops() -> None:
    _ensure_lib()


def register_block_callables(blocks: list) -> None:
    global _BLOCKS
    _BLOCKS = list(blocks)


def block_callables() -> list:
    return _BLOCKS


def register_chunk_callables(chunks: list) -> None:
    global _CHUNKS
    _CHUNKS = list(chunks)


def chunk_callables() -> list:
    return _CHUNKS


def register_edge_degree_callable(fn: Callable) -> None:
    global _EDGEDEG
    _EDGEDEG = fn


def edge_degree_callable() -> Callable | None:
    return _EDGEDEG
