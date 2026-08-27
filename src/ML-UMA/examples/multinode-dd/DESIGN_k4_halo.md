# Phase B: k=4 per-layer halo exchange — design

Goal: N=32 single point on 2 nodes, energy + per-atom force parity vs the 12-tile
ASE-GP oracle. Halo = 6 A (one message-passing layer), k=4 exchanges/forward.

## Node ordering (the contract everything depends on)

Under DD the traced model's node axis IS the LAMMPS owned+ghost order for this
rank:

```
node index:  [ 0 .. nlocal )      owned atoms (this rank)
             [ nlocal .. nall )   ghost atoms (copies of atoms owned elsewhere)
```

`build_dd_graph` already emits `edge_index` in this indexing (row0=neighbor j,
row1=center i, both in [0,nall)). `x_message` in the traced graph is
`[nall, sph_feature_size, sphere_channels]`.

## The op

```
uma_halo::exchange(Tensor x) -> Tensor        # x: [nall, F, C]
```

- **forward:** for every ghost node g (image of a remote owned atom a on rank r),
  overwrite x[g] with the current x[a] fetched from rank r. Owned rows unchanged.
  Result: all 6 A ghosts now carry their owners' up-to-date layer features.
- **backward:** each rank produced grad wrt x_out over ALL nall rows. A ghost
  row's grad belongs to the atom's OWNER. So send ghost-row grads back and
  ACCUMULATE (+=) onto the owner's owned row; zero the local ghost-row grad after
  sending (its contribution now lives on the owner). Owned-row grads pass
  through unchanged plus the accumulated remote contributions.

This is the exact adjoint of a gather (forward scatter owned->ghost, backward
reduce ghost->owner), analogous to AllGatherNodesFn in peer_context.cpp but with
a SPATIAL (LAMMPS ghost) map instead of the contiguous GP partition.

Injected in the top block loop (export_blocks_xpu.py:735), k=4:

```python
for i in range(self.num_layers):
    x_message = torch.ops.uma_halo.exchange(x_message)   # refresh ghosts
    x_message = torch.ops.uma_ckpt.block(i, x_message, ...)
```

Exchange BEFORE each block so block i reads correct neighbor features. 4 layers
-> 4 exchanges. (An exchange before layer 0 is required too: after the atom
embedding, ghost rows hold this rank's embedding of the ghost's Z, which is
actually already correct since embedding is per-atom and local; but a uniform
"exchange before every block" is simplest and correct. The pre-layer-0 exchange
is a no-op in value for the embedding but keeps the pattern uniform.)

## Transport: HaloContext singleton (mirrors PeerContext)

LAMMPS builds the owned<->ghost communication plan every neighbor rebuild
(comm->sendlist/sendnum/recvnum per swap, and the ghost's source owned index).
`pair_uma.cpp` captures a flat exchange plan once per build and installs it into a
process-wide `HaloContext`:

```
struct HaloPlan {
  // For forward (owned -> ghost): per neighbor rank, list of (local owned idx)
  //   to send, and the contiguous ghost-row range to receive into.
  // Implemented via LAMMPS comm forward/reverse, OR raw MPI on captured lists.
  ...
};
```

Two implementation options for the actual data movement:

- **Option T1 (reuse LAMMPS comm):** register pack/unpack_forward_comm and
  pack/unpack_reverse_comm on PairUMA; the op calls back into a bound PairUMA*
  that runs comm->forward_comm(this)/reverse_comm(this) on a staged buffer.
  Idiomatic; host-staged (D2H/H2D per exchange).
- **Option T2 (raw MPI on captured index lists):** HaloContext holds the
  send/recv index lists LAMMPS computed and does MPI_Isend/Irecv on the tensor
  rows directly. More control, still host-staged initially, GPU-aware later.

Start with **T1** (least new transport code, correctness first). The op forwards
to the bound PairUMA which stages x rows into a [nall, F*C] double buffer, runs
LAMMPS forward_comm to fill ghost rows, and returns. Backward runs reverse_comm.

## MoLE under k=4

Same as documented: set_MOLE_coefficients computes a per-system MEAN of the
composition embedding over the atomic_numbers it is handed. Under DD hand it
owned+ghost; for uniform NaCl the owned+ghost mean == global mean to parity
tolerance. Exact fix (owned-only allreduce of per-Z counts fed into the mean) is
mole_composition_allreduce() -> engine hook (Phase B+ if needed). Energy parity
will tell us if the homogeneous approximation holds.

## CRITICAL FINDING: per-atom energy (E1) is required for FORCES, not just energy

Careful analysis of the DD backward (see chat 2026-08-27): if each rank
backprops from its FULL subsystem energy E_r = sum over (owned+ghost) e_a, the
force on an atom picks up spurious gradients from the ghost-energy terms e_g,
which are also (correctly) counted on the ghost's owner. So naive
grad(E_subsystem, pos) is WRONG for forces under DD.

Correct DD forces: each rank must backprop from its OWNED-atom energy sum
E_owned(r) = sum_{a in owned(r)} e_a. Then:
  - the owner's own grad(E_owned, x_i) is the local part;
  - cross-rank parts (atom i as a ghost on neighbor ranks) arrive via the
    uma_halo::exchange BACKWARD (reverse comm ghost-grad -> owner).
Sum = exact global F_i = -dE_global/dx_i.

=> E1 (per-atom energy) is a HARD PREREQUISITE for the force gate too, because
the backprop root must be the owned-only energy sum. The infrastructure already
exists: `NodeEnergyExportWrapper` (export_wrapper.py:152) returns
(node_energy[N], total) with per-atom energy INSIDE the autograd graph, built
specifically for spatial DD. It must be:
  1. traced through the checkpointed block/chunk forward (currently
     export_blocks_xpu.py traces EnergyExportWrapper, which returns only the
     scalar). Integrate NodeEnergyExportWrapper into the k=4 DD export.
  2. consumed by a predict_body variant that receives node_energy[nnodes], forms
     E_owned = node_energy[0:nlocal].sum(), and calls grad(E_owned, pos). Forces
     for owned rows are then exact after the halo backward; energy for owned rows
     sums across ranks (eng_vdwl) for the global energy gate.

This unifies the force and energy paths: both come from the per-atom energy.

## Energy under DD (now testable)

k=4 does NOT by itself give a global energy scalar. The model returns one energy
for this rank's (owned+ghost) subsystem, which double-counts. For GLOBAL energy
we need per-atom energy of OWNED atoms summed across ranks (LAMMPS eng_vdwl).
Options:
- **E1:** export per-atom energy (eflag_atom) from the model; sum owned rows;
  LAMMPS reduces. Cleanest, enables the energy gate.
- **E2 (interim):** compute global energy as sum over ranks of (owned energy),
  where owned energy = full subsystem energy attributed to owned atoms. Without
  per-atom energy this is not separable, so E1 is required for the energy gate.

Decision: implement E1 (per-atom energy head is already in UMA; the traced top
returns node_embedding -> energy via the energy head; we need the per-atom
energy BEFORE the final sum). Then:
  eng_vdwl += sum_{i in owned} e_atom[i]
and forces as before (owned rows). This gives BOTH energy and force parity.

## Edge padding (P2.1) — required

Per-rank nall (and edge count) varies across ranks. The traced chunk count is
baked at export. Pad each rank's edge list up to a fixed multiple of
EDGE_AC_CHUNK, and pad nodes up to a capacity, so one artifact serves all ranks.
Padded edges/nodes contribute zero (masked). This is the P2.1 fix, mandatory for
DD. Export one k=4 artifact sized to worst-case per-rank capacity.

## Test

2 nodes x 12 tiles, N=32, run 0. Compare to hen/pbs/out/ase12_n32:
  energy: |dE| per-atom <= 1e-3 meV/atom
  force:  per-atom max|dF| <= 1e-5 eV/A over ALL atoms
via parity_vs_asegp.py (energy gate ENABLED once E1 lands).
```
