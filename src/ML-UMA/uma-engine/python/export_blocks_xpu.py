#!/usr/bin/env python3
"""Phase-6 (j) per-CHUNK-checkpointed TorchScript export for XPU UMA.

Exports ONE top module ``model_traced.pt`` (energy-only, differentiable w.r.t.
pos, like EnergyExportWrapper) whose escn message-passing block loop has been
rewritten to call ``torch.ops.uma_ckpt.block(i, ...)`` per block, PLUS each escn
block as its own TorchScript sub-module ``model_block_{i}.pt`` (real weights),
PLUS a per-block CHUNK sub-module ``model_chunk_{i}.pt`` that wraps ONE
``Edgewise.forward_chunk`` call.

Option (j') — the real memory unit is the edge-CHUNK, AND the wigner/x_edge
recompute must itself be PER CHUNK. Option (i)/(j) recomputed the FULL-edge
wigner [E,25,25]=6.5 GiB + wigner_inv 6.5 GiB + x_edge for ALL edges and THEN
split into chunks -> a ~13 GiB transient per block -> OOM at N=18. Fix: move the
wigner/wigner_inv_env/x_edge recompute ENTIRELY into the CHUNK module and do it
PER CHUNK from the CHUNK's edge precursors. The block now splits ONLY the small
precursors (edge_distance_vec [Ec,3], edge_distance [Ec], edge_index [2,Ec]) and
calls ``torch.ops.uma_ckpt.chunk(block_idx, x_full, edge_distance_vec_c,
edge_distance_c, atomic_numbers, edge_index_c, node_offset, mole_start, natoms)``
per chunk; each chunk MODULE builds only its OWN [Ec,25,25] wigner (~0.07 GiB @
Ec=16384). No edge-full wigner is ever built at block level OR op level. The top
block graph records a SEQUENCE of uma_ckpt.chunk calls; at runtime the C++ engine
wraps EACH chunk call in a CheckpointModuleFn so only ONE chunk's SO2
intermediates + its [Ec,25,25] wigner are live at a time (eager-AC memory profile
in the pure-traced / no-Python path).

See docs/phase6_graph_parallel_xpu_plan.md section "(j) per-CHUNK C++
checkpoint" (extended to (j') per-chunk wigner recompute).

Build-time only (FairChem Python). Traces on XPU (baked constants live on
xpu:0, matching runtime). Carries the single-tile campaign fixes:
  - shape-generic quaternion-Wigner trace patches (Phase 3)
  - FP64 Wigner-prep edge-chunk fix for correct backward at large edges (Phase 1)
  - activation_checkpointing OFF in escn (Phase 3c): we REPLACE it with the
    per-block uma_ckpt op; escn's own torch.utils.checkpoint is NOT used.
  - merge_mole OFF (Phase 3b: energy-exact)

BLOCK SUB-MODULE INTERFACE (option i, memory fix — settled):
    block(x, edge_distance_vec, edge_distance, atomic_numbers, edge_index,
          sys_node_emb) -> x
where
    x                 [natoms, sph_feature_size, sphere_channels]  node-sized
    edge_distance_vec [E, 3]   fp64   SMALL precursor (~34 MB @ E=1.4M)
    edge_distance     [E]      fp64   SMALL precursor
    atomic_numbers    [natoms] int64  (atomic_numbers_full; == atomic_numbers
                                        for the single-tile / non-GP export)
    edge_index        [2, E]   int64
    sys_node_emb      [natoms, sphere_channels]  node-sized
The block does NOT build wigner/wigner_inv_env/x_edge at all (design-doc option
j'). It SPLITS the small precursors per chunk and calls uma_ckpt.chunk per chunk;
each chunk module recomputes its OWN [Ec,25,25] wigner. So no edge-full tensor
(6.5 GiB wigner / 6.5 GiB wigner_inv_env / ~4 GiB x_edge) is ever built at block
or op level (avoids the ~13 GiB transient -> ~62 GiB OOM at N=18).

The loop-invariant scalars ``total_atoms_across_gp_ranks`` and ``node_offset``
(== gp_node_offset; 0 for the single-tile / non-GP export) and the
balance_channels inputs (charge, spin, natoms, batch) are BOUND as constants at
export time (captured from the N=2 sample). balance_channels is FOLDED INTO the
block sub-module (appended after escn_md_block.forward) so one block sub-module
fully reproduces one loop iteration. charge/spin are scalars (0 for NaCl); this
is correct for fixed charge/spin runs (documented; matches the design doc's
"balance_channels folded into each block sub-module" bullet).

Env:
  UMA_CKPT   (default hen uma-s-1p2.pt)
  UMA_TASK   (default omat)
  OUT        artifact dir (writes model_traced.pt + model_block_{i}.pt + metadata.json)
  N_LIST     comma NxNxN size(s) to build the trace sample from (default "2")
  FXPU_WIGNER_PREP_CHUNK / _MODE  (Wigner-chunk fix knobs)
  RECONSTRUCT (default 1) run the in-process reconstruct==monolithic validation
  EXPORT_WORLD (default 1)  graph-parallel world size W. W==1 = single-tile
                            (BIT-IDENTICAL to the pre-GP path). W>1 = per-rank GP
                            AC artifacts (P4 AC+GP merge).
  EXPORT_RANK  (default 0)  this rank R in [0, W). Export one rank per invocation
                            (fresh process per rank recommended, like
                            export_shards_xpu.py, to avoid XPU memory buildup).

GP artifact naming / directory scheme (documented for the C++ agent):
  W==1: OUT/model_traced.pt, OUT/model_block_{i}.pt, OUT/model_chunk_{i}.pt,
        OUT/model_edgedeg_chunk.pt, OUT/metadata.json  (UNCHANGED single-tile).
  W>1 : OUT/w{W}/r{R}/model_traced.pt + model_block_{i}.pt + model_chunk_{i}.pt +
        model_edgedeg_chunk.pt + metadata.json  (per-rank subdir; same filenames
        inside the subdir so the C++ mpi_peer_predictor can load a rank's AC
        module set by pointing at OUT/w{W}/r{R}). metadata.json records world,
        rank, num_blocks, edge_ac_chunk, gp_node_offset, total_atoms.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HEN = Path("/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen")
for p in (HEN / "shim", HEN / "patches", HEN):
    if p.is_dir():
        sys.path.insert(0, str(p))


def build_nacl(n, rattle=0.05, seed=0):
    from ase import Atoms
    a = 5.64
    na = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], float)
    cl = na + 0.5
    syms, sc = [], []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                o = np.array([ix, iy, iz], float)
                for f in na:
                    syms.append("Na"); sc.append((f + o) / n)
                for f in cl:
                    syms.append("Cl"); sc.append((f + o) / n)
    cell = np.eye(3) * (a * n)
    atoms = Atoms(symbols=syms, scaled_positions=sc, cell=cell, pbc=True)
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(0, rattle, atoms.positions.shape)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0
    return atoms


class BlockSubModule(torch.nn.Module):
    """One escn message-passing block + folded balance_channels, expressed as
    an eSCNMD_Block structure whose Edgewise call is REPLACED by a per-chunk
    uma_ckpt.chunk loop (design-doc option j' per-chunk memory fix).

    Interface (fixed): block(x, edge_distance_vec, edge_distance,
    atomic_numbers, edge_index, sys_node_emb) -> x. Reproduces exactly one
    iteration of escn_md.forward's block loop (escn_md.py:782-803):
    self.blocks[i](...) then balance_channels.

    UNLIKE option i, the block does NOT recompute the FULL-edge wigner /
    wigner_inv_env / x_edge (6.5 + 6.5 + ~4 GiB transient) and then split. It
    now SPLITS ONLY the small precursors (edge_distance_vec [E,3], edge_distance
    [E], edge_index [2,E]) into activation_checkpoint_chunk_size chunks and, for
    EACH chunk, calls torch.ops.uma_ckpt.chunk(block_idx, x_full,
    edge_distance_vec_c, edge_distance_c, atomic_numbers, edge_index_c,
    node_offset, mole_start, natoms). Each chunk MODULE recomputes its OWN
    [Ec,25,25] wigner internally. The partials are summed (collapse >8) exactly
    as eager. So NO edge-full tensor is ever built at block level.

    The block reproduces the eSCNMD_Block.forward structure manually so the
    edge_wise step can be replaced by the chunk loop (escn_md_block.py:427-464):
        x_res = x; x = norm_1(x); x[:,0,:] += sys_node_emb
        x_edgewise = SUM_chunks uma_ckpt.chunk(...); x = x_edgewise + x_res
        x_res = x; x = norm_2(x); x = atom_wise(x) + x_res
    then balance_channels.

    Bound submodules (from the real block, real weights): norm_1, norm_2,
    atom_wise, and edge_wise (used only to read
    activation_checkpoint_chunk_size; its forward_chunk lives in the chunk
    module). balance_channels config/targets bound as before.
    """

    def __init__(self, backbone, block, block_idx, total_atoms, node_offset,
                 charge, spin, natoms, batch, world=1):
        super().__init__()
        self.block_idx = int(block_idx)
        self.total_atoms = int(total_atoms)
        self.node_offset = int(node_offset)
        # GP (P4 AC+GP merge): W>1 -> the block's Edgewise upfront gather is a
        # real uma_peer.all_gather_nodes(x, total_atoms) (== escn_md_block.py
        # allgather_collect). W==1 -> identity (x_full == x), bit-identical to the
        # single-tile path.
        self.world = int(world)

        # --- eSCNMD_Block structural submodules (real weights) ---------------
        self.norm_1 = block.norm_1
        self.norm_2 = block.norm_2
        self.atom_wise = block.atom_wise
        # Read ONLY the chunk size (an int); do NOT register edge_wise as a
        # submodule. edge_wise's heavy forward_chunk + per-chunk wigner recompute
        # live in the chunk module (model_chunk_{i}.pt). Registering it here made
        # torch.jit.trace serialize ~580 MB of UNUSED weights per block (2.33 GiB
        # dead across 4 blocks) that were loaded onto the tile for nothing.
        # (Agent-1 review P0-b; BlockSubModule.forward never calls edge_wise.)
        self.activation_checkpoint_chunk_size = int(
            block.edge_wise.activation_checkpoint_chunk_size
        )

        # balance_channels config (ints) + captured target tensors.
        self.charge_channel_start = int(backbone.charge_channel_start)
        self.charge_channel_end = int(backbone.charge_channel_end)
        self.spin_channel_start = int(backbone.spin_channel_start)
        self.spin_channel_end = int(backbone.spin_channel_end)
        # charge/spin are scalars (fixed composition) -> safe as constants.
        self.register_buffer("charge", charge.clone().detach())
        self.register_buffer("spin", spin.clone().detach())
        # NOTE: natoms/batch are N-DEPENDENT -> must be derived from x at runtime,
        # NOT baked (baking the trace-N shape breaks index_add_ at other N).

    def _edgewise_chunked(self, x_full, edge_distance_vec, edge_distance,
                          atomic_numbers, edge_index, natoms):
        """Replace Edgewise.forward with a per-chunk uma_ckpt.chunk loop.

        Splits ONLY the small precursors (edge_distance_vec dim0, edge_distance
        dim0, edge_index dim1) by activation_checkpoint_chunk_size and calls the
        chunk op per chunk; each chunk module recomputes its own [Ec,25,25]
        wigner + x_edge internally. Partials summed exactly as eager (escn
        Edgewise.forward:178-202: collapse when >8).
        """
        chunk = self.activation_checkpoint_chunk_size
        edge_distance_vec_parts = edge_distance_vec.split(chunk, dim=0)
        edge_distance_parts = edge_distance.split(chunk, dim=0)
        edge_index_parts = edge_index.split(chunk, dim=1)

        # RUNNING SUM accumulation: each chunk returns a full [natoms,sph,chan]
        # node partial (~1.1 GiB at N=18). torch.stack(new_embeddings).sum() would
        # materialize many partials at once (~13 GiB) -> OOM. Accumulate in place
        # so only ONE accumulator + one partial are live. Each uma_ckpt.chunk is
        # independently checkpointed in C++, so its transients free per chunk.
        accum = None
        mole_start = 0
        for idx in range(len(edge_index_parts)):
            partial = torch.ops.uma_ckpt.chunk(
                self.block_idx,
                x_full,
                edge_distance_vec_parts[idx],
                edge_distance_parts[idx],
                atomic_numbers,
                edge_index_parts[idx],
                self.node_offset,
                mole_start,
                natoms,
            )
            accum = partial if accum is None else accum + partial
            mole_start += edge_index_parts[idx].shape[1]
        return accum

    def _balance(self, x):
        from fairchem.core.models.uma.escn_md import balance_channels_batched
        # Single system, one rank's node partition on input (x is [n_local, ...]).
        # batch = zeros(n_local) maps this rank's nodes to the single system.
        #
        # natoms MUST be the FULL-SYSTEM atom count, NOT n_local. Under GP,
        # balance_channels_batched (escn_md.py:181-194) does:
        #   system_sums = index_add_(local channels)          # per-rank partial
        #   system_sums = all_reduce_with_grad(system_sums)   # FULL-system sum
        #   corrections = (system_sums - target) / natoms     # <-- natoms=FULL N
        #   out = channels - corrections[batch]
        # i.e. escn's own block loop passes natoms=data_dict["natoms"] (the FULL
        # system count, which _generate_graph leaves untouched under GP) while
        # batch=data_dict["batch"] is the LOCAL partition. Using n_local here made
        # corrections W-times too large (the all_reduce'd sum divided by N/W
        # instead of N) -> the l=0 charge/spin channels were corrupted in EVERY
        # block on EVERY GP rank -> wrong energy + forces. This is bit-identical
        # to n_local only when world==1 (n_local == full N), so the single-tile
        # path is unaffected. self.total_atoms is the full system N (== nat for
        # W==1), baked at export from the sample.
        n_local = x.shape[0]
        batch = torch.zeros(n_local, dtype=torch.long, device=x.device)
        full_natoms = self.total_atoms if self.world > 1 else n_local
        natoms = torch.full((1,), full_natoms, dtype=torch.long, device=x.device)
        if self.charge_channel_end > self.charge_channel_start:
            x = balance_channels_batched(
                emb=x, target=self.charge, natoms=natoms, batch=batch,
                start_idx=self.charge_channel_start,
                end_idx=self.charge_channel_end, target_offset=0.0)
        if self.spin_channel_end > self.spin_channel_start:
            x = balance_channels_batched(
                emb=x, target=self.spin, natoms=natoms, batch=batch,
                start_idx=self.spin_channel_start,
                end_idx=self.spin_channel_end, target_offset=1.0)
        return x

    def forward(self, x, edge_distance_vec, edge_distance, atomic_numbers,
                edge_index, sys_node_emb):
        # eSCNMD_Block.forward structure (escn_md_block.py:427-464) with the
        # edge_wise call replaced by the per-chunk uma_ckpt.chunk loop. NO
        # edge-full wigner / wigner_inv_env / x_edge is ever built here — only
        # the small precursors are split; each chunk builds its OWN [Ec,25,25].
        natoms = x.shape[0]
        x_res = x
        x = self.norm_1(x)
        # sys_node_embedding added into the l=0 channel before edgewise
        # (escn_md_block.py:442-443).
        x = torch.cat(
            (
                (x[:, 0:1, :] + sys_node_emb.unsqueeze(1)),
                x[:, 1:, :],
            ),
            dim=1,
        )
        # Single-tile / non-GP (W==1): x_full == x (the block's Edgewise upfront
        # gather is identity) -> BIT-IDENTICAL to the pre-GP single-tile path.
        # GP (W>1): gather full-N node features ONCE per block via the uma_peer
        # collective, exactly like escn_md_block.py's allgather_collect
        #   x_full = gp_utils.gather_from_model_parallel_region_sum_grad(
        #                x, total_atoms_across_gp_ranks)
        # total_atoms == full N (8*Ncell^3); this rank holds only its
        # node_partition of x on input, and x_full is the full-N feature set the
        # per-chunk SO2 conv scatters from. edge_index chunks carry GLOBAL center
        # indices; node_offset (== gp_node_offset for this rank) is threaded into
        # each uma_ckpt.chunk so the edge->node scatter targets this rank's
        # partition (edge_index[1] - node_offset). At runtime uma_peer routes to
        # XcclPeer; here the export stand-in scatters this rank's shard into a
        # zero-filled [total_atoms, ...] buffer.
        if self.world > 1:
            x_full = torch.ops.uma_peer.all_gather_nodes(x, self.total_atoms)
        else:
            x_full = x
        x_edgewise = self._edgewise_chunked(
            x_full, edge_distance_vec, edge_distance, atomic_numbers,
            edge_index, natoms,
        )
        x = x_edgewise + x_res

        x_res = x
        x = self.norm_2(x)
        x = self.atom_wise(x) + x_res

        x = self._balance(x)
        return x


class _ChunkCore(torch.nn.Module):
    """Tensor-only core: recompute THIS chunk's wigner/x_edge from the chunk's
    small precursors, then run ONE Edgewise.forward_chunk (design-doc j').

    torch.jit.trace only accepts Tensor example inputs, so the int scalars
    (node_offset / mole_start / natoms) are handled by the scripted outer
    ChunkSubModule; this core takes ONLY tensors and derives natoms from
    x_full.shape[0]. node_offset is 0 for the single-tile / non-GP export and
    mole_start is baked to 0 for single-system MoLE (the traceable MOLE forward,
    mole_sizes.numel()==1, ignores ac_start_idx).

    The wigner/x_edge recompute (escn_md.forward:792-804 prologue + escn edge-
    embedding) is done PER CHUNK from the chunk's edge_distance_vec [Ec,3] /
    edge_distance [Ec] / atomic_numbers[edge_index]. Since it operates only on
    the chunk's Ec edges, ONLY a [Ec,25,25] wigner (~0.07 GiB @ Ec=16384) is
    ever built here — never the full-edge [E,25,25] (6.5 GiB). The backbone
    submodules required for the recompute are bound as attributes/submodules so
    the traced chunk module is self-contained.
    """

    def __init__(self, backbone, block, edge_wise, node_offset: int,
                 n_local: int = 0):
        super().__init__()
        self.edge_wise = edge_wise
        # GP: the edge->node scatter output size (x_original_shape) is this rank's
        # node_partition size n_local, NOT x_full.shape[0] (which is the FULL N
        # after all_gather_nodes). Baked per-rank (artifacts are per-rank + N-
        # specific). n_local<=0 (single-tile / W==1) -> derive from x_full.shape[0]
        # at runtime (== full N == n_local), keeping W==1 BIT-IDENTICAL.
        self.n_local = int(n_local)

        # --- backbone submodules bound for PER-CHUNK wigner/x_edge recompute --
        self.envelope = backbone.envelope
        self.distance_expansion = backbone.distance_expansion
        self.source_embedding = backbone.source_embedding
        self.target_embedding = backbone.target_embedding
        self.wigner_data = getattr(backbone, "wigner_data", None)
        self.mappingReduced = backbone.mappingReduced
        self.backend = backbone.backend
        # bound method: recomputes raw (wigner, wigner_inv) from edge vecs.
        self._get_rotmat_and_wigner = backbone._get_rotmat_and_wigner
        # wigner-prep scalars (escn_md.forward:792-804).
        self.cutoff = float(backbone.cutoff)
        self.lmax = int(backbone.lmax)
        self.mmax = int(backbone.mmax)
        self.use_quaternion_wigner = bool(backbone.use_quaternion_wigner)
        # coefficient_index selection buffer (None when mmax == lmax).
        if self.mmax != self.lmax:
            self.register_buffer(
                "coefficient_index",
                backbone.coefficient_index.clone().detach(),
                persistent=False,
            )
        else:
            self.coefficient_index = None

        self.node_offset = int(node_offset)

    def _recompute_chunk_edge_tensors(self, edge_distance_vec, edge_distance,
                                      atomic_numbers, edge_index):
        """Rebuild THIS chunk's (wigner, wigner_inv_envelope, x_edge) from the
        chunk's small precursors. Mirrors escn_md.forward:792-804 (obtain wigner)
        + the edge-embedding block, but only over the chunk's Ec edges so only a
        [Ec,25,25] wigner exists (never the 6.5 GiB full-edge wigner).
        """
        # (1) raw wigner from the CHUNK's edge_distance_vec, then prepare_wigner.
        wigner, wigner_inv = self._get_rotmat_and_wigner(edge_distance_vec)
        coefficient_index = self.coefficient_index if self.mmax != self.lmax else None
        wigner, wigner_inv = self.backend.prepare_wigner(
            wigner, wigner_inv, self.mappingReduced, coefficient_index,
        )
        # (2) envelope fused into wigner_inv (escn edge embedding).
        edge_envelope = self.envelope(edge_distance / self.cutoff).reshape(-1, 1, 1)
        wigner_inv_envelope = wigner_inv * edge_envelope
        # (3) x_edge = cat(dist_expansion, source_emb, target_emb).
        edge_distance_embedding = self.distance_expansion(edge_distance)
        source_embedding = self.source_embedding(atomic_numbers[edge_index[0]])
        target_embedding = self.target_embedding(atomic_numbers[edge_index[1]])
        x_edge = torch.cat(
            (edge_distance_embedding, source_embedding, target_embedding),
            dim=1,
        )
        return wigner, wigner_inv_envelope, x_edge

    def forward(self, x_full, edge_distance_vec, edge_distance, atomic_numbers,
                edge_index):
        # PER-CHUNK wigner/x_edge recompute (only [Ec,25,25] ever built).
        wigner, wigner_inv_env, x_edge = self._recompute_chunk_edge_tensors(
            edge_distance_vec, edge_distance, atomic_numbers, edge_index,
        )
        # x_original_shape (scatter output size): n_local under GP (this rank's
        # partition), else x_full.shape[0] (== full N for single-tile).
        natoms = self.n_local if self.n_local > 0 else x_full.shape[0]
        return self.edge_wise.forward_chunk(
            x_full,
            natoms,
            x_edge,
            edge_index,
            wigner,
            wigner_inv_env,
            self.node_offset,
            0,
        )


class ChunkSubModule(torch.nn.Module):
    """Wraps ONE edge chunk (per-chunk wigner recompute + Edgewise.forward_chunk)
    for a single block.

    Interface (fixed — the C++ agent recompute Function loads this and wraps it
    in a per-chunk recompute Function):
        chunk(x_full, edge_distance_vec, edge_distance, atomic_numbers,
              edge_index, node_offset: int, mole_start: int) -> partial
    where
        x_full            [natoms, sph_feature_size, sphere_channels] NODE-sized
                          (== x for single-tile / non-GP; ~0.05 GiB, saved)
        edge_distance_vec [Ec, 3]   fp64   CHUNK precursor
        edge_distance     [Ec]      fp64   CHUNK precursor
        atomic_numbers    [natoms]  int64  full node set (small)
        edge_index        [2, Ec]   int64  CHUNK
        node_offset       int   (gp_node_offset; 0 for single-tile)
        mole_start        int   (running edge count for MoLE ac_start_idx;
                                 ignored for single-system MoLE, kept for C++/GP)
    natoms is derived internally as x_full.shape[0] (the edge->node scatter
    size). Returns the [natoms, ...] node-sized partial embedding for this chunk.

    INTERNALLY (escn_md.forward:792-804 + edge embedding + forward_chunk):
        (1) wigner,wigner_inv = _get_rotmat_and_wigner(edge_distance_vec);
            wigner,wigner_inv = prepare_wigner(...);
            wigner_inv_env = wigner_inv * envelope(edge_distance/cutoff)
        (2) x_edge = cat(distance_expansion(edge_distance),
                         source_embedding(atomic_numbers[edge_index[0]]),
                         target_embedding(atomic_numbers[edge_index[1]]))
        (3) set_mole_ac_start_index(edge_wise, mole_start)  (inside forward_chunk)
        (4) edge_wise.forward_chunk(x_full, natoms, x_edge, edge_index, wigner,
                                    wigner_inv_env, node_offset, 0)
    Only the chunk's [Ec,25,25] wigner + SO2 transient + node-sized x_full ever
    exist inside -> under the C++ per-chunk checkpoint only ONE chunk's memory is
    live, and no edge-full wigner is EVER built.

    Scripted (not traced) so the int args survive in the saved signature; the
    heavy tensor ops (incl. the per-chunk wigner recompute) live in the traced
    `core` submodule.
    """

    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x_full, edge_distance_vec, edge_distance, atomic_numbers,
                edge_index, node_offset: int = 0, mole_start: int = 0):
        return self.core(x_full, edge_distance_vec, edge_distance,
                         atomic_numbers, edge_index)


class _EdgeDegChunkCore(torch.nn.Module):
    """Tensor-only core for the EDGE-DEGREE prologue, per chunk (P1-b).

    Mirrors _ChunkCore EXACTLY but for the edge_degree_embedding prologue
    (export_blocks_xpu.py old lines 482-505) instead of Edgewise. Recompute
    THIS chunk's wigner/wigner_inv_env/x_edge from the chunk's small precursors
    (edge_distance_vec [Ec,3], edge_distance [Ec], atomic_numbers[edge_index]),
    then run ONE EdgeDegreeEmbedding.forward_chunk which ACCUMULATES its scatter
    INTO x (running accumulation over the [natoms,...] accumulator). Only a
    [Ec,25,25] wigner (~0.07 GiB @ Ec=16384) ever exists — never the full-edge
    [E,25,25] (6.5 GiB) wigner / wigner_inv_env / x_edge.

    ONE module (not per-block): the edge_degree prologue happens once, before
    the block loop. node_offset is baked (0 for single-tile / non-GP export).

    torch.jit.trace only accepts Tensor example inputs, so the int scalars
    (node_offset / mole_start / natoms) are handled by the scripted outer
    EdgeDegreeChunkSubModule; this core takes ONLY tensors.
    """

    def __init__(self, backbone, node_offset: int):
        super().__init__()
        # EdgeDegreeEmbedding itself (holds rad_func + backend.edge_degree_scatter).
        self.edge_degree_embedding = backbone.edge_degree_embedding

        # --- backbone submodules bound for PER-CHUNK wigner/x_edge recompute --
        self.envelope = backbone.envelope
        self.distance_expansion = backbone.distance_expansion
        self.source_embedding = backbone.source_embedding
        self.target_embedding = backbone.target_embedding
        self.wigner_data = getattr(backbone, "wigner_data", None)
        self.mappingReduced = backbone.mappingReduced
        self.backend = backbone.backend
        # bound method: recomputes raw (wigner, wigner_inv) from edge vecs.
        self._get_rotmat_and_wigner = backbone._get_rotmat_and_wigner
        # wigner-prep scalars (escn_md.forward:792-804).
        self.cutoff = float(backbone.cutoff)
        self.lmax = int(backbone.lmax)
        self.mmax = int(backbone.mmax)
        self.use_quaternion_wigner = bool(backbone.use_quaternion_wigner)
        # coefficient_index selection buffer (None when mmax == lmax).
        if self.mmax != self.lmax:
            self.register_buffer(
                "coefficient_index",
                backbone.coefficient_index.clone().detach(),
                persistent=False,
            )
        else:
            self.coefficient_index = None

        self.node_offset = int(node_offset)

    def _recompute_chunk_edge_tensors(self, edge_distance_vec, edge_distance,
                                      atomic_numbers, edge_index):
        """Rebuild THIS chunk's (wigner_inv_envelope, x_edge) from the chunk's
        small precursors. Identical recompute to _ChunkCore but returns only the
        tensors edge_degree_embedding.forward_chunk needs (x_edge,
        wigner_inv_envelope). Only a [Ec,25,25] wigner exists.
        """
        wigner, wigner_inv = self._get_rotmat_and_wigner(edge_distance_vec)
        coefficient_index = self.coefficient_index if self.mmax != self.lmax else None
        wigner, wigner_inv = self.backend.prepare_wigner(
            wigner, wigner_inv, self.mappingReduced, coefficient_index,
        )
        edge_envelope = self.envelope(edge_distance / self.cutoff).reshape(-1, 1, 1)
        wigner_inv_envelope = wigner_inv * edge_envelope
        edge_distance_embedding = self.distance_expansion(edge_distance)
        source_embedding = self.source_embedding(atomic_numbers[edge_index[0]])
        target_embedding = self.target_embedding(atomic_numbers[edge_index[1]])
        x_edge = torch.cat(
            (edge_distance_embedding, source_embedding, target_embedding),
            dim=1,
        )
        return wigner_inv_envelope, x_edge

    def forward(self, x, edge_distance_vec, edge_distance, atomic_numbers,
                edge_index):
        # PER-CHUNK wigner/x_edge recompute (only [Ec,25,25] ever built).
        wigner_inv_env, x_edge = self._recompute_chunk_edge_tensors(
            edge_distance_vec, edge_distance, atomic_numbers, edge_index,
        )
        # EdgeDegreeEmbedding.forward_chunk ACCUMULATES its scatter INTO x
        # (running accumulation over the [natoms,...] accumulator).
        return self.edge_degree_embedding.forward_chunk(
            x,
            x_edge,
            edge_index,
            wigner_inv_env,
            self.node_offset,
        )


class EdgeDegreeChunkSubModule(torch.nn.Module):
    """Wraps ONE edge chunk of the EDGE-DEGREE prologue (per-chunk wigner
    recompute + EdgeDegreeEmbedding.forward_chunk accumulate-into-x). ONE module
    (model_edgedeg_chunk.pt), not per-block — the prologue runs once.

    Interface (fixed — for the C++ agent; the C++ recompute Function loads this
    and wraps each uma_ckpt.edge_degree call in a per-chunk recompute Function):
        edge_degree(x, edge_distance_vec, edge_distance, atomic_numbers,
                    edge_index, node_offset: int, mole_start: int) -> x
    where
        x                 [natoms, sph_feature_size, sphere_channels] NODE-sized
                          accumulator (forward_chunk accumulates INTO it;
                          returns the updated x)
        edge_distance_vec [Ec, 3]   fp64   CHUNK precursor
        edge_distance     [Ec]      fp64   CHUNK precursor
        atomic_numbers    [natoms]  int64  full node set (atomic_numbers_full)
        edge_index        [2, Ec]   int64  CHUNK
        node_offset       int   (gp_node_offset; 0 for single-tile)
        mole_start        int   (running edge count for MoLE ac_start_idx;
                                 ignored for single-system MoLE, kept for C++/GP)

    Scripted (not traced) so the int args survive in the saved signature; the
    heavy tensor ops (incl. the per-chunk wigner recompute + forward_chunk) live
    in the traced `core` submodule.
    """

    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x, edge_distance_vec, edge_distance, atomic_numbers,
                edge_index, node_offset: int = 0, mole_start: int = 0):
        return self.core(x, edge_distance_vec, edge_distance,
                         atomic_numbers, edge_index)


def _pad_edges_to_chunk_multiple(graph_dict, edge_ac_chunk, cutoff):
    """P2.1: pad edge_index/edge_distance/edge_distance_vec up to a fixed multiple
    of edge_ac_chunk so the traced per-chunk loop count is constant across edge
    drift. Padded edges are self-loops (0->0) placed BEYOND the cutoff so the
    radial envelope zeroes their contribution (and gradient) exactly.

    edge_index         [2, E]  int64   -> [2, E_pad]
    edge_distance      [E]     float    -> [E_pad]     (pad value = 2*cutoff)
    edge_distance_vec  [E, 3]  float    -> [E_pad, 3]  (pad = [2*cutoff,0,0])
    where E_pad = ceil(E / edge_ac_chunk) * edge_ac_chunk. If E is already a
    multiple, one extra full chunk is added so runtime drift upward stays within
    the same chunk count band; the extra chunk is all-padded (zero contribution).
    """
    ei = graph_dict["edge_index"]
    ed = graph_dict["edge_distance"]
    edv = graph_dict["edge_distance_vec"]
    E = int(ei.shape[1])
    # Bake a FIXED padded edge count at trace time (a multiple of edge_ac_chunk,
    # plus one guard chunk). torch.jit.trace unrolls the split loop to exactly
    # ceil(E_pad/chunk) iterations; the RUNTIME C++ engine pads its (drifting)
    # edge count to this SAME E_pad before calling the module, so the runtime edge
    # count always yields the identical chunk count -> no list-length mismatch.
    # The pad target is recorded in metadata (edge_pad_multiple) for the C++ side.
    n_chunks = (E // edge_ac_chunk) + 1
    E_pad = n_chunks * edge_ac_chunk
    pad = E_pad - E
    if pad <= 0:
        return graph_dict
    dev = ed.device
    fdt = ed.dtype
    far = 2.0 * float(cutoff)
    # self-loop on node 0, beyond cutoff -> envelope(d/cutoff)=0 -> zero message
    # (and zero gradient); numerically identical to omitting the edge.
    ei_pad = torch.zeros((2, pad), dtype=ei.dtype, device=ei.device)
    ed_pad = torch.full((pad,), far, dtype=fdt, device=dev)
    edv_pad = torch.zeros((pad, 3), dtype=edv.dtype, device=edv.device)
    edv_pad[:, 0] = far
    graph_dict = dict(graph_dict)
    graph_dict["edge_index"] = torch.cat([ei, ei_pad], dim=1)
    graph_dict["edge_distance"] = torch.cat([ed, ed_pad], dim=0)
    graph_dict["edge_distance_vec"] = torch.cat([edv, edv_pad], dim=0)
    return graph_dict


def make_ckpt_forward(backbone, submodules, edge_ac_chunk=None):
    """Return a bound forward() that rewrites the block loop to uma_ckpt.block
    AND the edge-degree prologue to a uma_ckpt.edge_degree chunk loop (P1-b).

    Everything else in escn_md.forward (epilogue norm+out) is preserved. The
    block loop body becomes torch.ops.uma_ckpt.block(i, ...); the edge-degree
    prologue becomes a per-chunk torch.ops.uma_ckpt.edge_degree(...) loop (each
    chunk module recomputes its OWN [Ec,25,25] wigner). ``edge_ac_chunk`` is the
    prologue's edge chunk size (EDGE_AC_CHUNK; default env or 16384).
    """
    from torch.profiler import record_function

    if edge_ac_chunk is None:
        edge_ac_chunk = int(os.environ.get("EDGE_AC_CHUNK", "16384"))
    edge_ac_chunk = int(edge_ac_chunk)

    def forward(self, data_dict):
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()
        data_dict["atomic_numbers_full"] = data_dict["atomic_numbers"]
        data_dict["batch_full"] = data_dict["batch"]

        csd_mixed_emb = self.csd_embedding(
            charge=data_dict["charge"],
            spin=data_dict["spin"],
            dataset=data_dict.get("dataset", default=None),
        )
        self.set_MOLE_coefficients(
            atomic_numbers_full=data_dict["atomic_numbers_full"],
            batch_full=data_dict["batch_full"],
            csd_mixed_emb=csd_mixed_emb,
        )
        if not self.regress_config.direct_forces:
            if self.regress_config.forces or self.regress_config.stress:
                data_dict["pos"].requires_grad_(True)
            if self.regress_config.stress:
                data_dict["cell"].requires_grad_(True)

        with record_function("generate_graph"):
            graph_dict = self._generate_graph(data_dict)

        # --- P2.1 EDGE PADDING (fixed-multiple chunk count) ------------------
        # ROOT CAUSE: the prologue loop and every block's internal Edgewise loop
        # split the edge tensors by edge_ac_chunk in a Python for-loop, which
        # torch.jit.trace UNROLLS to a fixed number of uma_ckpt.chunk /
        # uma_ckpt.edge_degree calls = ceil(E_trace / edge_ac_chunk). At runtime a
        # different edge count E' gives a different chunk count -> the traced
        # graph's baked list length mismatches ("Expected K elements in a list but
        # found K+1"): the N=24 / N=16 / N=36 NVT step-1 crashes.
        #
        # FIX (external-graph path): the edge_index is supplied by the caller
        # (external_graph_gen=True; the C++ engine passes eidx/coff into the
        # module). The C++ engine pads the runtime edge_index up to a fixed
        # multiple of edge_ac_chunk (self-loops beyond cutoff -> zero envelope ->
        # zero contribution/gradient) BEFORE calling the module, so the edge count
        # the split loop sees is ALWAYS a chunk multiple == the traced count. The
        # trace example is likewise pre-padded (see main()) so the baked chunk
        # count matches. No in-graph padding here (it cannot fix a baked loop
        # length); padding lives at the single controllable boundary: the caller.
        # Runtime pad rule + zero-contribution are validated by the N=24 NVT gate.

        # NOTE (P1-b): the FULL-edge "obtain wigner" prologue
        # (_get_rotmat_and_wigner + prepare_wigner over ALL edges -> wigner /
        # wigner_inv [E,25,25] = 6.5 + 6.5 GiB) is REMOVED. It fed ONLY the
        # edge_degree_embedding prologue (wigner_inv * edge_envelope), which is
        # now chunked+checkpointed via uma_ckpt.edge_degree (each chunk module
        # recomputes its OWN [Ec,25,25] wigner). The block loop already
        # recomputes its own per-chunk wigner from precursors, so NOTHING else
        # uses the full-edge wigner. No full-edge wigner is ever built.

        with record_function("atom embedding"):
            x_message = torch.zeros(
                data_dict["atomic_numbers"].shape[0],
                self.sph_feature_size,
                self.sphere_channels,
                device=data_dict["pos"].device,
                dtype=data_dict["pos"].dtype,
            )
            x_message[:, 0, :] = self.sphere_embedding(data_dict["atomic_numbers"])

        sys_node_embedding = csd_mixed_emb[data_dict["batch"]]
        x_message[:, 0, :] = x_message[:, 0, :] + sys_node_embedding

        self.set_MOLE_sizes(
            nsystems=csd_mixed_emb.shape[0],
            batch_full=data_dict["batch_full"],
            edge_index=graph_dict["edge_index"],
        )
        self.log_MOLE_stats()

        # --- REWRITTEN EDGE-DEGREE PROLOGUE (P1-b per-chunk checkpoint) ------
        # The monolithic prologue built the FULL-edge wigner_inv_env [E,25,25]
        # (6.5 GiB) + x_edge for ALL edges and ran edge_degree_embedding
        # (index_add_ scatter, IndexAddBackward0 = 12.82 GiB at N=18) with grad
        # ON, OUTSIDE every checkpoint — the last un-checkpointed full-edge
        # transient and THE N=18 blocker. It is REPLACED by a chunk loop over
        # uma_ckpt.edge_degree, mirroring the block loop's uma_ckpt.chunk: split
        # ONLY the small per-chunk precursors (edge_distance_vec [Ec,3],
        # edge_distance [Ec], edge_index [2,Ec]) and call edge_degree per chunk.
        # The ONE edgedeg chunk module recomputes THIS chunk's wigner/
        # wigner_inv_env/x_edge internally (only [Ec,25,25] wigner ever built)
        # then runs EdgeDegreeEmbedding.forward_chunk, which ACCUMULATES its
        # scatter INTO x. So pass the RUNNING x_message and reassign — only ONE
        # [natoms,...] accumulator + one chunk's transients are ever live. NO
        # full-edge wigner / wigner_inv_env / x_edge is built at any point.
        with record_function("edge embedding"):
            edge_index_full = graph_dict["edge_index"]
            edge_distance_vec_full = graph_dict["edge_distance_vec"]
            edge_distance_full = graph_dict["edge_distance"]
            atomic_numbers_full = data_dict["atomic_numbers_full"]
            gp_node_offset = int(data_dict["gp_node_offset"])
            natoms = x_message.shape[0]

            edv_parts = edge_distance_vec_full.split(edge_ac_chunk, dim=0)
            ed_parts = edge_distance_full.split(edge_ac_chunk, dim=0)
            ei_parts = edge_index_full.split(edge_ac_chunk, dim=1)

            mole_start = 0
            for idx in range(len(ei_parts)):
                # forward_chunk accumulates into x -> pass the running x_message
                # and reassign. mole_start advances by this chunk's edge count.
                x_message = torch.ops.uma_ckpt.edge_degree(
                    x_message,
                    edv_parts[idx],
                    ed_parts[idx],
                    atomic_numbers_full,
                    ei_parts[idx],
                    gp_node_offset,
                    mole_start,
                    natoms,
                )
                mole_start += ei_parts[idx].shape[1]
        # ---------------------------------------------------------------------

        # NOTE: x_edge_per_layer (backend.get_layer_radial_emb) removed — it is
        # DEAD in the rewritten loop (blocks recompute x_edge per chunk from
        # precursors). (Agent-1 review P1-c.)

        # --- REWRITTEN BLOCK LOOP (option i memory fix) ----------------------
        # Pass the SMALL precursors (edge_distance_vec [E,3], edge_distance [E],
        # atomic_numbers_full [natoms] int, edge_index [2,E] int) instead of the
        # 6.5 GiB wigner / 6.5 GiB wigner_inv_env / ~4 GiB x_edge. Each block
        # RECOMPUTES those internally (BlockSubModule._recompute_edge_tensors),
        # so the C++ per-block checkpoint saves only x + these small tensors and
        # recomputes wigner in backward (freed after) = eager-AC memory.
        # atomic_numbers here is atomic_numbers_full (full node set); for the
        # single-tile / non-GP export it equals data_dict["atomic_numbers"].
        # sys_node_embedding is node-sized (small); balance_channels folded in.
        edge_index = graph_dict["edge_index"]
        edge_distance_vec = graph_dict["edge_distance_vec"]
        edge_distance = graph_dict["edge_distance"]
        atomic_numbers_full = data_dict["atomic_numbers_full"]
        for i in range(self.num_layers):
            with record_function(f"message passing {i}"):
                # DD k=4 (UMA_DD_HALO=1 at export): refresh the 6 A ghost node
                # features before each block via the spatial halo exchange, so a
                # thin (one-layer) halo suffices instead of a deep 24 A halo.
                # No-op at runtime when HaloContext is inactive (single-rank / GP).
                if self.dd_halo:
                    x_message = torch.ops.uma_halo.exchange(x_message)
                x_message = torch.ops.uma_ckpt.block(
                    i,
                    x_message,
                    edge_distance_vec,
                    edge_distance,
                    atomic_numbers_full,
                    edge_index,
                    sys_node_embedding,
                )
        # ---------------------------------------------------------------------

        x_message = self.norm(x_message)
        out = {
            "node_embedding": x_message,
            "batch": data_dict["batch"],
        }
        return out

    return forward


def main() -> int:
    ckpt = Path(os.environ.get("UMA_CKPT", str(HEN / "uma-cache" / "uma-s-1p2.pt")))
    task = os.environ.get("UMA_TASK", "omat")
    out_root = Path(os.environ["OUT"]); out_root.mkdir(parents=True, exist_ok=True)
    n_list = [int(x) for x in os.environ.get("N_LIST", "2").split(",") if x.strip()]
    n_trace = n_list[0]
    do_reconstruct = os.environ.get("RECONSTRUCT", "1").strip() in ("1", "true", "yes")

    # --- P4 AC+GP merge: graph-parallel world/rank (default W=1 = single-tile) --
    world = int(os.environ.get("EXPORT_WORLD", "1"))
    rank = int(os.environ.get("EXPORT_RANK", "0"))
    if world < 1 or rank < 0 or rank >= world:
        raise SystemExit(f"invalid EXPORT_WORLD/EXPORT_RANK: {rank}/{world}")
    gp = world > 1
    # W==1: write directly into OUT (BIT-IDENTICAL single-tile layout). W>1: write
    # per-rank into OUT/w{W}/r{R}/ so each rank has its own AC module set.
    out = out_root if not gp else (out_root / f"w{world}" / f"r{rank}")
    out.mkdir(parents=True, exist_ok=True)
    # RECONSTRUCT (loads a monolithic reference on the SAME input) is only valid
    # for the single-tile path; under GP the reference forward is not the same
    # (edges are sharded + collectives injected), so skip it for W>1.
    if gp:
        do_reconstruct = False
    print(f"EXPORT_WORLD={world} EXPORT_RANK={rank} gp={gp} -> {out}", flush=True)

    # Wigner-chunk fix (correct FP64 backward at large edge counts).
    os.environ.setdefault("FXPU_WIGNER_PREP_CHUNK", "65536")
    os.environ.setdefault("FXPU_WIGNER_PREP_CHUNK_MODE", "both")
    try:
        from xpu_prepare_wigner import apply_xpu_prepare_wigner_chunking
        note = apply_xpu_prepare_wigner_chunking()
        print(f"wigner-chunk fix applied: {note}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN wigner-chunk not applied: {exc}", flush=True)

    from common import atoms_to_atomic_data, inference_settings_with_dtype
    from export_wrapper import (
        make_traced_export_wrapper,
        make_node_energy_export_wrapper,
    )
    from metadata import build_export_metadata
    from model_loader import get_atom_refs, load_prepared_hydra_model
    from trace_patch import apply_trace_patches, restore_trace_patches
    from uma_ckpt_ops import (
        install_ckpt_ops,
        register_block_callables,
        register_chunk_callables,
        register_edge_degree_callable,
    )
    # P4 AC+GP merge: uma_peer collective ops (all_gather_nodes / all_reduce_sum)
    # + FairChem gp_utils patch, reused verbatim from export_shards_xpu.py.
    from uma_peer_ops import (
        install_export_ops,
        patch_fairchem_gp_utils,
        set_export_rank,
    )

    dtype = "float64"; torch_dtype = torch.float64

    trace_dev = os.environ.get("TRACE_DEV", "xpu").strip()
    if trace_dev == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit(
            "block export (TRACE_DEV=xpu) needs 1 visible XPU tile; run on a "
            "compute node. (login node has no XPU: py_compile / dry-run only.)")
    try:
        from fairchem_xpu_parallel import patch_fairchem_xpu_device
        patch_fairchem_xpu_device()
    except Exception as exc:  # noqa: BLE001
        print(f"WARN patch_fairchem_xpu_device: {exc}", flush=True)

    def settings_for():
        s = inference_settings_with_dtype(dtype)
        s.external_graph_gen = True
        s.activation_checkpointing = False   # replaced by per-block uma_ckpt op
        s.execution_mode = "general"
        s.merge_mole = False
        return s

    install_ckpt_ops()
    install_export_ops()
    # DD k=4: register uma_halo::exchange so torch.jit.trace records the op node
    # (identity at trace; real owned<->ghost movement is the C++ engine at runtime).
    if os.environ.get("UMA_DD_HALO", "0").strip() in ("1", "true", "yes"):
        from uma_halo_ops import install_halo_ops
        install_halo_ops()

    # P4 AC+GP merge: BEFORE building the wrapper, patch FairChem gp_utils so
    # escn runs the GP path (node_partition, gp_node_offset, per-block
    # gather_from_model_parallel_region_sum_grad routed to uma_peer). Reused
    # verbatim from export_shards_xpu.py. W==1 leaves gp_utils.initialized()
    # False -> single-tile path unchanged (we do NOT patch at all when W==1, so
    # the single-tile export is BIT-IDENTICAL to the pre-GP exporter).
    restore_gp = None
    if gp:
        set_export_rank(rank, world)
        restore_gp = patch_fairchem_gp_utils(world, rank)

    atoms = build_nacl(n_trace)
    nat = len(atoms)
    s = settings_for()
    sample = atoms_to_atomic_data(atoms, task_name=task, settings=s)
    model, _, _ = load_prepared_hydra_model(
        str(ckpt), sample, settings=s, device=trace_dev)
    model = model.to(device=trace_dev, dtype=torch_dtype)
    data = atoms_to_atomic_data(atoms, task_name=task, settings=s)

    # Build the traced-export wrapper (energy-only, differentiable). It holds a
    # DEEP COPY of the prepared model (clone_prepared_model), so patching its
    # backbone forward + block callables is isolated from `model` (used later as
    # the untouched monolithic reference in the RECONSTRUCT check).
    # DD k=4 (UMA_DD_HALO=1): trace the NodeEnergyExportWrapper, which returns
    # (node_energy[N], total_energy) with per-atom energy INSIDE the autograd
    # graph. Spatial DD needs per-atom energy so each rank can backprop from its
    # OWNED-only energy sum (forces) and sum owned energies for the global energy.
    # Non-DD keeps the scalar-only wrapper (unchanged product path).
    dd_export = os.environ.get("UMA_DD_HALO", "0").strip() in ("1", "true", "yes")
    if dd_export:
        wrapper = make_node_energy_export_wrapper(model, task).eval().to(trace_dev)
    else:
        wrapper = make_traced_export_wrapper(model, task).eval().to(trace_dev)
    backbone = wrapper.inner.backbone

    # INTRA-block edge-chunk AC: per-block checkpointing (uma_ckpt) bounds
    # CROSS-block memory, but a single block's SO2 conv over ALL edges still
    # peaks too high at large N. Set each block's edge-AC chunk size so the
    # BlockSubModule splits the SMALL precursors into that many-edge chunks and
    # emits a uma_ckpt.chunk op per chunk. Under the C++ per-chunk
    # CheckpointModuleFn (no_grad fwd + recompute), each chunk (incl. its own
    # [Ec,25,25] wigner recompute) frees after use.
    edge_ac_chunk = int(os.environ.get("EDGE_AC_CHUNK", "16384"))
    # DD k=4 halo: inject uma_halo::exchange before each block so a 6 A halo
    # (one message-passing layer) suffices instead of a deep 24 A halo. The op is
    # a runtime no-op unless the pair style installs a HaloContext (multi-node DD).
    backbone.dd_halo = os.environ.get("UMA_DD_HALO", "0").strip() in ("1", "true", "yes")
    import types
    for i, blk in enumerate(backbone.blocks):
        # escn_md_block: edge_wise holds the chunk size (read by BlockSubModule).
        if hasattr(blk, "edge_wise") and hasattr(blk.edge_wise,
                                                 "activation_checkpoint_chunk_size"):
            blk.edge_wise.activation_checkpoint_chunk_size = edge_ac_chunk
        if hasattr(blk, "activation_checkpoint_chunk_size"):
            blk.activation_checkpoint_chunk_size = edge_ac_chunk
        # option (j'): NO Edgewise.forward monkeypatch. The BlockSubModule does
        # the per-chunk uma_ckpt.chunk loop itself (splitting the SMALL
        # precursors), and each chunk module recomputes its own wigner. The
        # block never calls edge_wise.forward, and never builds full-edge wigner.
    print(f"intra-block edge AC chunk size -> {edge_ac_chunk} "
          f"(per-CHUNK uma_ckpt.chunk op; per-chunk wigner recompute)",
          flush=True)

    # Capture loop-invariant scalars + balance_channels tensors for the sample.
    # Single-tile / non-GP export: gp_node_offset == 0, total_atoms == nat.
    charge = data.charge.long().reshape(-1).to(trace_dev)
    spin = data.spin.long().reshape(-1).to(trace_dev)
    natoms_t = torch.tensor([nat], dtype=torch.long, device=trace_dev)
    batch_t = torch.zeros(nat, dtype=torch.long, device=trace_dev)
    # total_atoms is ALWAYS the full system N (8*Ncell^3): it is the size of the
    # all_gather_nodes / gather_from_model_parallel_region buffer and of the
    # per-chunk edge->node scatter target. node_offset is this rank's partition
    # start = node_partition.min() with node_partition =
    # tensor_split(arange(nat), W)[rank] (== graph_shard.h / escn GP contract).
    # W==1: node_offset == 0, total_atoms == nat (BIT-IDENTICAL single-tile).
    total_atoms = nat
    if gp:
        _node_partition = torch.tensor_split(torch.arange(nat), world)[rank]
        node_offset = int(_node_partition.min().item())
        n_local = int(_node_partition.numel())
        print(f"GP rank {rank}/{world}: node_partition n_local="
              f"{n_local} node_offset={node_offset} "
              f"total_atoms={total_atoms}", flush=True)
    else:
        node_offset = 0
        n_local = 0   # 0 -> chunk cores derive n_local from x_full (== full N)

    submodules = torch.nn.ModuleList([
        BlockSubModule(
            backbone, backbone.blocks[i], i, total_atoms, node_offset,
            charge, spin, natoms_t, batch_t, world,
        ).eval().to(device=trace_dev, dtype=torch_dtype)
        for i in range(backbone.num_layers)
    ])
    register_block_callables([submodules[i] for i in range(backbone.num_layers)])

    # option (j'): one _ChunkCore per block. It recomputes THIS chunk's wigner /
    # wigner_inv_env / x_edge from the chunk precursors, then runs that block's
    # edge_wise.forward_chunk (tensor-only, for tracing). The uma_ckpt.chunk
    # stand-in dispatches to a small adapter over the core.
    chunk_cores = torch.nn.ModuleList([
        _ChunkCore(backbone, backbone.blocks[i],
                   backbone.blocks[i].edge_wise, node_offset, n_local)
        .eval().to(device=trace_dev, dtype=torch_dtype)
        for i in range(backbone.num_layers)
    ])

    def _chunk_adapter(core):
        def call(x_full, edge_distance_vec, edge_distance, atomic_numbers,
                 edge_index, node_offset_arg, mole_start_arg, natoms_arg):
            # node_offset baked into core (single-tile 0); mole_start/natoms
            # unused for single-system (natoms derived from x_full). The core
            # recomputes this chunk's wigner/x_edge from the precursors.
            return core(x_full, edge_distance_vec, edge_distance,
                        atomic_numbers, edge_index)
        return call

    register_chunk_callables(
        [_chunk_adapter(chunk_cores[i]) for i in range(backbone.num_layers)])

    # P1-b: ONE edgedeg chunk core (NOT per-block). It recomputes THIS chunk's
    # wigner/wigner_inv_env/x_edge from the chunk precursors then runs
    # EdgeDegreeEmbedding.forward_chunk (accumulates into x). The
    # uma_ckpt.edge_degree stand-in dispatches to a small adapter over the core.
    edgedeg_core = (
        _EdgeDegChunkCore(backbone, node_offset)
        .eval().to(device=trace_dev, dtype=torch_dtype)
    )

    def _edgedeg_adapter(core):
        def call(x, edge_distance_vec, edge_distance, atomic_numbers,
                 edge_index, node_offset_arg, mole_start_arg, natoms_arg):
            # node_offset baked into core (single-tile 0); mole_start/natoms
            # unused for single-system. The core recomputes this chunk's
            # wigner/x_edge from precursors and accumulates into x.
            return core(x, edge_distance_vec, edge_distance,
                        atomic_numbers, edge_index)
        return call

    register_edge_degree_callable(_edgedeg_adapter(edgedeg_core))

    # Monkeypatch the backbone instance forward (bound method) to the rewritten
    # block-loop forward. Only THIS wrapper's backbone is patched; restored in
    # finally. `model` (reference) is a separate object, untouched.
    orig_forward = backbone.forward
    backbone.forward = types.MethodType(
        make_ckpt_forward(backbone, submodules, edge_ac_chunk), backbone)

    example = list(wrapper.example_inputs_from_data(data))
    example = [t.to(trace_dev) if torch.is_tensor(t) else t for t in example]
    example[0] = example[0].to(torch_dtype)   # pos
    example[2] = example[2].to(torch_dtype)   # cell
    example[5] = example[5].to(torch_dtype)   # cell_offsets

    # P4 AC+GP merge: shard this RANK's edges by NODE PARTITION (must match
    # graph_shard.h + the escn GP contract), reused verbatim from
    # export_shards_xpu.py. example layout: [pos, atomic_numbers, cell, pbc,
    # edge_index(4), cell_offsets(5), charge, spin]. Keep edges whose CENTER
    # (edge_index[1]) is in node_partition = tensor_split(arange(nat), W)[rank].
    # A naive contiguous edge slice includes out-of-partition centers, which
    # escn's `edge_index[1] - node_offset` drives to index -1 (the historic
    # "index -1 size 3888" GP bug). At runtime the C++ engine re-shards edges the
    # same way; the baked example edges only need to be a VALID rank shard so the
    # trace records the right per-rank graph shape. W==1: no sharding (identity),
    # single-tile trace unchanged.
    if gp:
        eidx = example[4]; coff = example[5]
        nodes = torch.tensor_split(
            torch.arange(nat, device=eidx.device), world)[rank]
        centers = eidx[1]
        keep = torch.isin(centers, nodes)
        idx = keep.nonzero().squeeze(-1)
        example[4] = eidx.index_select(1, idx).contiguous()
        example[5] = coff.index_select(0, idx).contiguous()
        print(f"GP rank {rank}/{world}: sharded edges "
              f"{int(eidx.shape[1])} -> {int(example[4].shape[1])}", flush=True)

    # DD k=4: pad the trace example to a fixed edge cap so the unrolled chunk
    # loop (ceil(E/EDGE_AC_CHUNK)) is baked at the SAME count every rank uses at
    # runtime (which also pads to UMA_DD_EDGE_CAP). Append a dummy node and fill
    # with dummy->dummy self-loops. example layout:
    #   [0]=pos[n,3] [1]=z[n] [2]=cell [3]=pbc [4]=edge_index[2,E] [5]=coff[E,3]
    #   [6]=charge [7]=spin
    # --- P2.1 / W10 edge-cap padding of the trace example ---------------------
    # Pad the trace example's edge_index up to a FIXED capacity that is a multiple
    # of edge_ac_chunk, so torch.jit.trace bakes a constant chunk-loop count. The
    # C++ runtime pads its (drifting) per-step edge count to the SAME cap via
    # graph_shard::pad_edges_to_capacity, so the chunk count never mismatches ->
    # fixes the N=24/N=16/N=36 NVT step-1 crash. Padded edges are self-loops on an
    # appended dummy node placed far beyond the cutoff -> envelope=0 -> exactly
    # zero energy/force contribution. Enabled by default (UMA_EDGE_PAD=1); the DD
    # path (UMA_DD_HALO=1) uses its explicit UMA_DD_EDGE_CAP.
    edge_pad_on = os.environ.get("UMA_EDGE_PAD", "1").strip() not in (
        "0", "false", "no")
    edge_pad_cap = 0
    edge_pad_atom = 0
    if dd_export or edge_pad_on:
        pos_e = example[0]; z_e = example[1]
        eidx = example[4]; coff = example[5]
        n_real = int(pos_e.shape[0])
        E_real = int(eidx.shape[1])
        if dd_export:
            cap = int(os.environ.get("UMA_DD_EDGE_CAP", "0"))
            if cap <= 0:
                raise SystemExit("UMA_DD_HALO=1 requires UMA_DD_EDGE_CAP > 0 "
                                 "(the fixed traced edge/chunk capacity)")
            if E_real > cap:
                raise SystemExit(f"trace edge count {E_real} exceeds "
                                 f"UMA_DD_EDGE_CAP {cap}; raise the cap")
            # DD path: appended dummy node (legacy behaviour).
            dummy = n_real
            far = 1.0e6
            pos_pad = torch.full((1, 3), far, dtype=pos_e.dtype, device=pos_e.device)
            example[0] = torch.cat([pos_e, pos_pad], dim=0).contiguous()
            example[1] = torch.cat(
                [z_e, z_e.new_full((1,), int(z_e[0]) if z_e.numel() else 1)],
                dim=0).contiguous()
            n_pad = cap - E_real
            pad_e = torch.full((2, n_pad), dummy, dtype=eidx.dtype,
                               device=eidx.device)
            pad_c = torch.zeros((n_pad, 3), dtype=coff.dtype, device=coff.device)
            example[4] = torch.cat([eidx, pad_e], dim=1).contiguous()
            example[5] = torch.cat([coff, pad_c], dim=0).contiguous()
            edge_pad_cap = int(cap)
            edge_pad_atom = int(dummy)
        else:
            # NORMAL path (P2.1): NO dummy node — pad with self-loops with a large
            # cell_offset (coff[:,0]=2.0 -> shift 2 lattice vectors -> |r| >> cutoff
            # -> envelope 0 -> zero contribution). CRITICAL for GP: the pad edge's
            # CENTER (row 1) is the scatter target into this rank's LOCAL node
            # accumulator, so it MUST be a node this rank OWNS. Use node_offset
            # (the rank's first owned global node); for W==1 that is 0. A global
            # index the rank does not own (e.g. 0 on rank 1) scatters out of the
            # local partition -> GPU segfault. Edge indices in the shard are GLOBAL
            # (all_gather_nodes reconstructs the full set), so pad_atom is global
            # node_offset. C++ must pad with the SAME per-rank node_offset.
            # cap = (E // chunk + 1) * chunk: chunk multiple + one guard chunk.
            cap = ((E_real // edge_ac_chunk) + 1) * edge_ac_chunk
            n_pad = cap - E_real
            pad_atom = int(node_offset)
            pad_e = torch.full((2, n_pad), pad_atom, dtype=eidx.dtype,
                               device=eidx.device)
            pad_c = torch.zeros((n_pad, 3), dtype=coff.dtype, device=coff.device)
            pad_c[:, 0] = 2.0
            # C++ prepends pad edges (cat({pad, real})); match that ordering so the
            # per-chunk split sees the same layout at trace and runtime.
            example[4] = torch.cat([pad_e, eidx], dim=1).contiguous()
            example[5] = torch.cat([pad_c, coff], dim=0).contiguous()
            edge_pad_cap = int(cap)
            edge_pad_atom = int(pad_atom)
        print(f"P2.1 edge-cap: trace edges padded {E_real} -> {edge_pad_cap} "
              f"(pad_atom={edge_pad_atom}); chunk count baked at "
              f"{-(-edge_pad_cap // edge_ac_chunk)} (edge_ac_chunk={edge_ac_chunk})",
              flush=True)

    report = {"n_trace": n_trace, "natoms": nat,
              "num_layers": int(backbone.num_layers),
              "world": world, "rank": rank, "gp": gp,
              "gp_node_offset": node_offset, "total_atoms": total_atoms}
    ok, err = True, None
    traced = None
    try:
        apply_trace_patches(shape_generic=True, checkpoint_passthrough=True)
        try:
            # (c) no_grad trace: traced module is energy-only; forces come from
            # C++ autograd::grad at runtime. no_grad (NOT inference_mode) so
            # constants stay differentiable.
            with torch.no_grad():
                _ = wrapper(*example)
                traced = torch.jit.trace(wrapper, tuple(example), strict=False)
                # Trace each block sub-module separately (real weights) so C++
                # can load + checkpoint them. Use the SAME example activations
                # the top graph feeds the op, captured live.
                block_examples, chunk_examples, edgedeg_example = \
                    _capture_block_examples(
                        backbone, submodules, chunk_cores, edgedeg_core,
                        wrapper, example, trace_dev, torch_dtype)
                traced_blocks = []
                for i in range(backbone.num_layers):
                    tb = torch.jit.trace(submodules[i], block_examples[i],
                                         strict=False)
                    traced_blocks.append(tb)
                # option (j'): trace each block's CHUNK core (per-chunk wigner
                # recompute + one forward_chunk, tensor-only) then wrap in a
                # scripted ChunkSubModule so the int (node_offset, mole_start)
                # signature survives. C++ loads model_chunk_{i}.pt and wraps each
                # uma_ckpt.chunk call in a per-chunk recompute Function.
                traced_chunks = []
                for i in range(backbone.num_layers):
                    # chunk example tuple is
                    # (x_full, edge_distance_vec, edge_distance, atomic_numbers,
                    #  edge_index, node_offset, mole_start, natoms); trace with
                    # tensors only (first 5).
                    ce = chunk_examples[i]
                    core_example = (ce[0], ce[1], ce[2], ce[3], ce[4])
                    tcore = torch.jit.trace(chunk_cores[i], core_example,
                                            strict=False)
                    csub = ChunkSubModule(tcore).eval()
                    tc = torch.jit.script(csub)
                    traced_chunks.append(tc)
                # P1-b: trace the ONE edgedeg chunk core (per-chunk wigner
                # recompute + one EdgeDegreeEmbedding.forward_chunk accumulate,
                # tensor-only) then wrap in a scripted EdgeDegreeChunkSubModule
                # so the int (node_offset, mole_start) signature survives. C++
                # loads model_edgedeg_chunk.pt (ONE module) and wraps each
                # uma_ckpt.edge_degree call in a per-chunk recompute Function.
                ee = edgedeg_example
                edgedeg_core_example = (ee[0], ee[1], ee[2], ee[3], ee[4])
                tedcore = torch.jit.trace(edgedeg_core, edgedeg_core_example,
                                          strict=False)
                edsub = EdgeDegreeChunkSubModule(tedcore).eval()
                traced_edgedeg = torch.jit.script(edsub)
        finally:
            restore_trace_patches()

        top_path = out / "model_traced.pt"
        # opt2: the top module only DISPATCHES uma_ckpt.block / uma_ckpt.edge_degree
        # ops (the heavy block/MOLE/edge_degree weights execute inside the separate
        # model_block_*/model_chunk_*/model_edgedeg modules). Those weights are
        # baked but DEAD in the top graph (~2 GiB/rank). freeze() folds constants
        # and drops attributes/params unreachable from forward -> strips the dead
        # weights. Bit-exact (removes only unused tensors). Env UMA_NO_FREEZE=1 to
        # disable.
        if os.environ.get("UMA_NO_FREEZE", "0").strip() not in ("1", "true", "yes"):
            try:
                _frozen = torch.jit.freeze(traced.eval())
                traced = _frozen
                print("opt2: froze top module (dead weights stripped)", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"opt2: freeze skipped ({type(exc).__name__}: {exc})", flush=True)
        traced.save(str(top_path))
        torch.jit.load(str(top_path), map_location="cpu")
        try:
            _mb = os.path.getsize(str(top_path)) / 1048576.0
            print(f"opt2: model_traced.pt = {_mb:.1f} MB", flush=True)
        except OSError:
            pass
        block_paths = []
        for i, tb in enumerate(traced_blocks):
            bp = out / f"model_block_{i}.pt"
            tb.save(str(bp))
            torch.jit.load(str(bp), map_location="cpu")
            block_paths.append(str(bp))
        chunk_paths = []
        for i, tc in enumerate(traced_chunks):
            cp = out / f"model_chunk_{i}.pt"
            tc.save(str(cp))
            torch.jit.load(str(cp), map_location="cpu")
            chunk_paths.append(str(cp))
        # P1-b: ONE edgedeg chunk module (not per-block).
        edgedeg_path = out / "model_edgedeg_chunk.pt"
        traced_edgedeg.save(str(edgedeg_path))
        torch.jit.load(str(edgedeg_path), map_location="cpu")
        report["top"] = str(top_path)
        report["blocks"] = block_paths
        report["chunks"] = chunk_paths
        report["edgedeg_chunk"] = str(edgedeg_path)
    except Exception as exc:  # noqa: BLE001
        ok, err = False, f"{type(exc).__name__}: {exc}"
        import traceback; traceback.print_exc()
    finally:
        backbone.forward = orig_forward

    report["ok"] = ok
    report["error"] = err
    print(f"export -> {'OK' if ok else 'FAIL ' + str(err)}", flush=True)

    if ok:
        meta = build_export_metadata(
            model=model, model_name="uma-s-1p2", task_name=task,
            settings=s, export_format="per_block_ckpt",
            checkpoint_path=str(ckpt),
            atom_refs=get_atom_refs("uma-s-1p2"),
            export_notes=[
                "phase6 (j') per-CHUNK-checkpointed export, per-chunk wigner (XPU)",
                "top: model_traced.pt (energy-only, uma_ckpt.block loop)",
                f"blocks: model_block_0..{backbone.num_layers - 1}.pt "
                "(each block splits the SMALL precursors per chunk and emits a "
                "sequence of uma_ckpt.chunk calls; NO edge-full wigner built)",
                f"chunks: model_chunk_0..{backbone.num_layers - 1}.pt "
                "(each recomputes its OWN [Ec,25,25] wigner/x_edge from the chunk "
                "precursors then runs one Edgewise.forward_chunk; C++ wraps each "
                "uma_ckpt.chunk call in a per-chunk recompute Function)",
                "edgedeg: model_edgedeg_chunk.pt (P1-b ONE module; the "
                "edge-degree PROLOGUE is chunked+checkpointed via "
                "uma_ckpt.edge_degree — each chunk recomputes its OWN [Ec,25,25] "
                "wigner/wigner_inv_env/x_edge then runs "
                "EdgeDegreeEmbedding.forward_chunk accumulating INTO x; NO "
                "full-edge wigner/x_edge built in the prologue; C++ wraps each "
                "uma_ckpt.edge_degree call in a per-chunk recompute Function)",
                "balance_channels folded into block sub-module; charge/spin "
                "bound as constants (fixed charge/spin runs)",
            ])
        meta_d = meta.to_dict()
        # C++ BlockContext prefers metadata num_blocks (else counts files).
        meta_d["num_blocks"] = int(backbone.num_layers)
        meta_d["num_chunk_modules"] = int(backbone.num_layers)
        # P1-b: ONE edgedeg chunk module (single, not per-block).
        meta_d["num_edgedeg"] = 1
        meta_d["edgedeg_chunk_module"] = "model_edgedeg_chunk.pt"
        meta_d["edge_ac_chunk"] = edge_ac_chunk
        # P2.1 edge-cap padding: the fixed traced edge capacity (a multiple of
        # edge_ac_chunk) and the dummy pad atom index. The C++ engine pads its
        # per-step edge_index up to edge_pad_cap on atom edge_pad_atom (self-loops
        # beyond cutoff -> zero contribution) so the traced chunk count is
        # invariant to per-step edge drift. 0 => padding off (legacy).
        meta_d["edge_pad_cap"] = int(edge_pad_cap)
        meta_d["edge_pad_atom"] = int(edge_pad_atom)
        # P4 AC+GP merge metadata (read by the C++ mpi_peer_predictor): world,
        # rank, gp_node_offset (this rank's node_partition start), total_atoms
        # (full system N, the all_gather_nodes buffer size). W==1 => world=1,
        # rank=0, gp_node_offset=0, total_atoms=nat (single-tile, unchanged).
        meta_d["world"] = int(world)
        meta_d["rank"] = int(rank)
        meta_d["gp"] = bool(gp)
        meta_d["gp_node_offset"] = int(node_offset)
        meta_d["total_atoms"] = int(total_atoms)
        # DD k=4: artifact traced with the per-layer halo op + per-atom energy.
        meta_d["dd_halo"] = bool(dd_export)
        if dd_export:
            meta_d["dd_k"] = int(backbone.num_layers)      # exchanges/fwd (k=4)
            meta_d["dd_halo_op"] = "uma_halo::exchange"
            meta_d["returns_node_energy"] = True           # top returns (node_e, total)
            meta_d["edge_ac_chunk"] = edge_ac_chunk
            # Node-feature width for the halo comm buffer (LAMMPS must size its
            # forward/reverse comm buffer to this many doubles/atom BEFORE the run).
            meta_d["sph_feature_size"] = int(backbone.sph_feature_size)
            meta_d["sphere_channels"] = int(backbone.sphere_channels)
            meta_d["dd_halo_width"] = int(backbone.sph_feature_size) * \
                int(backbone.sphere_channels)
            meta_d["export_notes"].append(
                "DD k=4 spatial domain decomposition: top module returns "
                "(node_energy[N], total_energy); uma_halo::exchange called before "
                "each of num_layers blocks to refresh the 6 A ghost features. "
                "C++ backprops from sum(node_energy[owned]); halo backward routes "
                "ghost grads to owners. Edge list padded to a fixed cap "
                "(UMA_DD_EDGE_CAP) so the traced chunk count is rank-invariant.")
        if gp:
            meta_d["export_format"] = "per_block_ckpt_gp"
            meta_d["gp_gather_op"] = "uma_peer::all_gather_nodes"
            meta_d["export_notes"].append(
                f"P4 AC+GP merge: world={world} rank={rank} "
                f"gp_node_offset={node_offset} total_atoms={total_atoms}; each "
                "block gathers full-N node features once via "
                "uma_peer.all_gather_nodes(x, total_atoms) (== "
                "escn_md_block allgather_collect) then runs the per-chunk "
                "uma_ckpt.chunk loop; edges sharded by node partition "
                "(tensor_split(arange(N), W)[R] centers); C++ handles the force "
                "all_reduce. Artifacts in OUT/w{W}/r{R}/.")
        (out / "metadata.json").write_text(
            json.dumps(meta_d, indent=2, default=str))

    if ok and do_reconstruct:
        try:
            rec_ok = run_reconstruct_check(
                model=model, wrapper=wrapper, data=data, task=task,
                out=out, backbone=backbone, submodules=submodules,
                chunk_cores=chunk_cores, edgedeg_core=edgedeg_core,
                example=example, trace_dev=trace_dev, torch_dtype=torch_dtype,
                make_ckpt_forward=make_ckpt_forward,
                edge_ac_chunk=edge_ac_chunk,
            )
            report["reconstruct_ok"] = bool(rec_ok)
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            report["reconstruct_ok"] = False
            print(f"RECONSTRUCT FAIL (exception): {exc}", flush=True)

    # P4 AC+GP merge: graph-structure validation. Full GP correctness needs 2+
    # tiles (C++ XcclPeer side), so here we confirm the traced top graph contains
    # the expected collective + checkpoint op structure:
    #   - uma_peer.all_gather_nodes ONCE per block (== num_layers) under GP
    #   - uma_ckpt.block ONCE per block
    #   - uma_ckpt.chunk / uma_ckpt.edge_degree present (per-chunk AC)
    if ok:
        # Reload the saved artifacts and scan every method graph. uma_ckpt::block
        # and uma_ckpt::edge_degree live in the TOP graph; uma_peer::all_gather_
        # nodes and uma_ckpt::chunk live in the per-block graphs. Reloading the
        # on-disk .pt is the authoritative check of what will run in C++.
        def _all_method_graphs(path):
            try:
                m = torch.jit.load(str(path), map_location="cpu")
            except Exception:
                return ""
            parts = []
            for sm in m.modules():
                # opt2: torch.jit.freeze inlines/renames forward, so the `graph`
                # property getter can raise RuntimeError ("Method 'forward' is
                # not defined") rather than being absent. hasattr() only swallows
                # AttributeError, so guard the whole access; frozen graphs are
                # then simply skipped (this scan is a best-effort op-count check,
                # NOT a correctness gate).
                try:
                    parts.append(sm.graph.str())
                except Exception:
                    pass
            return "\n".join(parts)

        top_graph_text = _all_method_graphs(out / "model_traced.pt")
        # opt2: a frozen top module exposes no readable graph string, so the
        # top-level ops (uma_ckpt::block / uma_ckpt::edge_degree) are not
        # countable. Detect this so the structure check does NOT falsely FAIL on
        # the block/edge_degree counts (the per-block graphs, which carry
        # all_gather_nodes + chunk, are NOT frozen and are still checked).
        top_opaque = (top_graph_text.strip() == "")
        graph_texts = [top_graph_text]
        for i in range(int(backbone.num_layers)):
            graph_texts.append(_all_method_graphs(out / f"model_block_{i}.pt"))
            graph_texts.append(_all_method_graphs(out / f"model_chunk_{i}.pt"))
        graph_texts.append(_all_method_graphs(out / "model_edgedeg_chunk.pt"))
        allgraphs = "\n".join(graph_texts)
        n_gather = allgraphs.count("uma_peer::all_gather_nodes")
        n_block = allgraphs.count("uma_ckpt::block")
        n_chunk = allgraphs.count("uma_ckpt::chunk")
        n_edeg = allgraphs.count("uma_ckpt::edge_degree")
        report["graph_check"] = {
            "all_gather_nodes": n_gather, "block": n_block,
            "chunk": n_chunk, "edge_degree": n_edeg,
            "expected_gather_per_block": world > 1,
            "top_frozen_opaque": bool(top_opaque),
        }
        print(f"[graph-check] all_gather_nodes={n_gather} block={n_block} "
              f"chunk={n_chunk} edge_degree={n_edeg} (num_layers="
              f"{int(backbone.num_layers)})"
              f"{' [top frozen: block/edge_degree counts N/A]' if top_opaque else ''}",
              flush=True)
        if gp:
            # block/edge_degree live ONLY in the (opaque-when-frozen) top graph;
            # skip those two counts if the top module is frozen (opt2).
            gp_ok = (n_gather == int(backbone.num_layers)
                     and n_chunk > 0
                     and (top_opaque or (n_block == int(backbone.num_layers)
                                         and n_edeg > 0)))
            report["graph_check"]["gp_structure_ok"] = bool(gp_ok)
            print(f"GP GRAPH-STRUCTURE {'PASS' if gp_ok else 'FAIL'} "
                  f"(expect all_gather_nodes==num_layers, chunk>0"
                  f"{'' if top_opaque else ', block==num_layers, edge_degree>0'})",
                  flush=True)
        else:
            report["graph_check"]["single_tile_no_gather_ok"] = (n_gather == 0)

    if restore_gp is not None:
        restore_gp()

    (out / "block_export_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nDONE -> {out}", flush=True)
    return 0 if ok else 2


def _capture_block_examples(backbone, submodules, chunk_cores, edgedeg_core,
                            wrapper, example, trace_dev, torch_dtype):
    """Run the patched wrapper once, capturing each block's AND each block's
    first-chunk input tuples, AND the edge-degree prologue's first-chunk tuple.

    The uma_ckpt.block stand-in calls the registered block callable; inside it
    the (option-j') BlockSubModule chunk loop emits uma_ckpt.chunk ops (with the
    chunk PRECURSORS) that call the registered chunk callable. The prologue
    emits uma_ckpt.edge_degree ops (P1-b) that call the registered edge_degree
    callable. We temporarily wrap ALL THREE so each block, each chunk, and the
    edgedeg chunk sub-module get a faithful example for their own
    torch.jit.trace.
    """
    from uma_ckpt_ops import (
        register_block_callables,
        register_chunk_callables,
        register_edge_degree_callable,
    )
    captured: dict[int, tuple] = {}
    captured_chunk: dict[int, tuple] = {}
    captured_edgedeg: dict[int, tuple] = {}
    real = list(submodules)
    real_cores = list(chunk_cores)

    def make_recorder(i, mod):
        def rec(x, edge_distance_vec, edge_distance, atomic_numbers,
                edge_index, sys_node_emb):
            captured[i] = (x.detach(), edge_distance_vec.detach(),
                           edge_distance.detach(), atomic_numbers,
                           edge_index, sys_node_emb.detach())
            return mod(x, edge_distance_vec, edge_distance, atomic_numbers,
                       edge_index, sys_node_emb)
        return rec

    def make_chunk_recorder(i, core):
        def rec(x_full, edge_distance_vec, edge_distance, atomic_numbers,
                edge_index, node_offset, mole_start, natoms):
            # Capture only the FIRST chunk per block as the trace example
            # (all chunks share the same op signature; chunk core is
            # shape-generic over the chunk edge count).
            if i not in captured_chunk:
                captured_chunk[i] = (
                    x_full.detach(), edge_distance_vec.detach(),
                    edge_distance.detach(), atomic_numbers, edge_index,
                    int(node_offset), int(mole_start), int(natoms),
                )
            return core(x_full, edge_distance_vec, edge_distance,
                        atomic_numbers, edge_index)
        return rec

    def restore_chunk_adapter(i, core):
        def call(x_full, edge_distance_vec, edge_distance, atomic_numbers,
                 edge_index, node_offset_arg, mole_start_arg, natoms_arg):
            return core(x_full, edge_distance_vec, edge_distance,
                        atomic_numbers, edge_index)
        return call

    def make_edgedeg_recorder(core):
        def rec(x, edge_distance_vec, edge_distance, atomic_numbers,
                edge_index, node_offset, mole_start, natoms):
            # Capture only the FIRST prologue chunk as the trace example
            # (all chunks share the op signature; core is edge-count-generic).
            if 0 not in captured_edgedeg:
                captured_edgedeg[0] = (
                    x.detach(), edge_distance_vec.detach(),
                    edge_distance.detach(), atomic_numbers, edge_index,
                    int(node_offset), int(mole_start), int(natoms),
                )
            return core(x, edge_distance_vec, edge_distance,
                        atomic_numbers, edge_index)
        return rec

    def restore_edgedeg_adapter(core):
        def call(x, edge_distance_vec, edge_distance, atomic_numbers,
                 edge_index, node_offset_arg, mole_start_arg, natoms_arg):
            return core(x, edge_distance_vec, edge_distance,
                        atomic_numbers, edge_index)
        return call

    register_block_callables(
        [make_recorder(i, real[i]) for i in range(len(real))])
    register_chunk_callables(
        [make_chunk_recorder(i, real_cores[i]) for i in range(len(real_cores))])
    register_edge_degree_callable(make_edgedeg_recorder(edgedeg_core))
    try:
        with torch.no_grad():
            _ = wrapper(*example)
    finally:
        register_block_callables(real)
        register_chunk_callables(
            [restore_chunk_adapter(i, real_cores[i])
             for i in range(len(real_cores))])
        register_edge_degree_callable(restore_edgedeg_adapter(edgedeg_core))
    block_examples = [captured[i] for i in range(len(real))]
    chunk_examples = [captured_chunk[i] for i in range(len(real_cores))]
    edgedeg_example = captured_edgedeg[0]
    return block_examples, chunk_examples, edgedeg_example


def run_reconstruct_check(*, model, wrapper, data, task, out, backbone,
                          submodules, chunk_cores, edgedeg_core, example,
                          trace_dev, torch_dtype, make_ckpt_forward,
                          edge_ac_chunk):
    """CRITICAL validation: reconstruct forward from saved block, chunk AND
    edgedeg modules and compare energy + autograd force to the ORIGINAL
    monolithic model on the SAME N input. Prints RECONSTRUCT PASS/FAIL.

    Reconstruction = LOADED model_edgedeg_chunk.pt prologue (via
    uma_ckpt.edge_degree per chunk) + for-each-block(load model_block_i.pt; call
    it, whose Edgewise.forward emits uma_ckpt.chunk calls -> LOADED
    model_chunk_i.pt) + epilogue, wired via the SAME rewritten backbone forward.
    This proves the on-disk top+block+chunk+edgedeg split reproduces the
    monolithic model, not just the in-memory modules.
    """
    from uma_ckpt_ops import (
        register_block_callables,
        register_chunk_callables,
        register_edge_degree_callable,
    )

    num_layers = int(backbone.num_layers)
    loaded = [torch.jit.load(str(out / f"model_block_{i}.pt"),
                             map_location=trace_dev)
              for i in range(num_layers)]
    loaded_chunks = [torch.jit.load(str(out / f"model_chunk_{i}.pt"),
                                    map_location=trace_dev)
                     for i in range(num_layers)]
    loaded_edgedeg = torch.jit.load(str(out / "model_edgedeg_chunk.pt"),
                                    map_location=trace_dev)

    # --- Reference: ORIGINAL unpatched monolithic model on the same input. ---
    from export_wrapper import EnergyExportWrapper
    ref_wrapper = EnergyExportWrapper(model, task, traceable=True).eval().to(trace_dev)
    pos_ref = example[0].clone().detach().to(torch_dtype).requires_grad_(True)
    ref_args = [pos_ref] + [example[k] for k in range(1, len(example))]
    e_ref = ref_wrapper(*ref_args)
    f_ref = torch.autograd.grad(e_ref.sum(), pos_ref, create_graph=False)[0]

    # --- Reconstruction: patched backbone forward -> LOADED block modules ->
    #     LOADED chunk modules (via uma_ckpt.chunk). ---
    register_block_callables(
        [(lambda m: (lambda x, edv, ed, an, ei, sne:
                     m(x, edv, ed, an, ei, sne)))(loaded[i])
         for i in range(num_layers)])
    register_chunk_callables(
        [(lambda m: (lambda x_full, edv, ed, an, ei, no, ms, na:
                     m(x_full, edv, ed, an, ei, int(no), int(ms))))(
                         loaded_chunks[i])
         for i in range(num_layers)])
    register_edge_degree_callable(
        (lambda m: (lambda x, edv, ed, an, ei, no, ms, na:
                    m(x, edv, ed, an, ei, int(no), int(ms))))(loaded_edgedeg))
    import types
    orig_forward = backbone.forward
    backbone.forward = types.MethodType(
        make_ckpt_forward(backbone, submodules, edge_ac_chunk), backbone)
    try:
        pos_rec = example[0].clone().detach().to(torch_dtype).requires_grad_(True)
        rec_args = [pos_rec] + [example[k] for k in range(1, len(example))]
        e_rec = wrapper(*rec_args)
        f_rec = torch.autograd.grad(e_rec.sum(), pos_rec, create_graph=False)[0]
    finally:
        backbone.forward = orig_forward
        register_block_callables([submodules[i] for i in range(num_layers)])
        register_chunk_callables(
            [(lambda c: (lambda x_full, edv, ed, an, ei, no, ms, na:
                         c(x_full, edv, ed, an, ei)))(chunk_cores[i])
             for i in range(num_layers)])
        register_edge_degree_callable(
            (lambda c: (lambda x, edv, ed, an, ei, no, ms, na:
                        c(x, edv, ed, an, ei)))(edgedeg_core))

    de = float((e_rec - e_ref).abs().max().item())
    df = float((f_rec - f_ref).abs().max().item())
    e_tol, f_tol = 1e-8, 1e-6
    passed = (de < e_tol) and (df < f_tol)
    print(f"[reconstruct] |dE|={de:.3e} eV (tol {e_tol:.0e})  "
          f"max|dF|={df:.3e} (tol {f_tol:.0e})", flush=True)
    print(f"RECONSTRUCT {'PASS' if passed else 'FAIL'}", flush=True)
    return passed


if __name__ == "__main__":
    raise SystemExit(main())
