# Dev plan — multi-node parallelism for UMA in LAMMPS

**Date:** 2026-08-26
**Status:** design proposal, pre-implementation
**Supersedes for multi-node:** the graph-parallel (GP) path in
`src/ML-UMA/uma-engine/src/mpi_peer_predictor.cpp` (retained for intra-node use).

---

## 0. Recommendation in one paragraph

**Use spatial domain decomposition (DD) with a *tunable-depth halo* and
per-layer halo exchange — not graph parallelism, and not a single 24 Å halo.**
Graph parallelism is fundamentally unscalable across nodes because its
communication volume is `O(N_total)` per rank per layer and does not shrink as
nodes are added; at N=38 on 16 nodes it would move ~388 GB/node/step. A single
deep 24 Å halo is *correct* and needs almost no communication, but wastes
2.5–6.8× in redundant compute. Exchanging node features at every message-passing
layer boundary reduces the required halo to **6 Å** and the redundancy to
**1.3–1.8×**, at a communication cost of ~40 ms/step against a current step time
of ~18 s — i.e. **0.2%**. Critically, **all of these variants are numerically
exact**; the halo depth `n_layers × cutoff` fully contains the receptive field.
That means your 0.01 meV/atom error budget does **not** need to be spent on the
parallel decomposition at all, and should instead be spent on precision
reduction and, as a stretch, sub-exact halo truncation.

---

## 1. Problem statement

UMA-s-1p2 as deployed here (`metadata.json`: `num_blocks: 4`, `cutoff: 6.0`,
`max_neighbors: 300`) is a 4-layer message-passing network with a 6 Å per-layer
cutoff. The receptive field of an atom is therefore **24 Å**. Any decomposition
must reconcile that 24 Å non-locality with a desire to put different atoms on
different nodes.

Current state (validated, `SESSION_RECOVERY_2026-08-26.md`):

- Single tile: N=18, 46,656 atoms.
- 12 tiles (1 Aurora node), graph-parallel over XCCL: N=38, 438,976 atoms,
  10-step NVT in 800 s; N=32 optimized to 235 s wall / 180 s loop.
- N=40 OOMs. **Everything above is single-node.**

The goal is to scale past one node.

---

## 2. Why graph parallelism cannot cross nodes

The GP path shards *edges* across ranks while replicating *nodes*. Every rank
must therefore see every atom's feature vector at every layer, via
`uma_peer::all_gather_nodes` (`src/peer_context.cpp:153`,
`python/uma_peer_ops.py:47`), called once per block
(`export_blocks_xpu.py:1191`: "all_gather_nodes ONCE per block == num_layers").

Node feature tensor is `x [natoms, sph_feature_size, sphere_channels]`
(`export_blocks_xpu.py:42`). Taking `sph_feature_size = 9` (lmax=2) and
`sphere_channels = 128` → **1152 values/atom = 9216 B/atom in FP64**.
*(These two dimensions are not recorded in `metadata.json` — task M1 below
measures them; every number in this document scales linearly with them.)*

**Received volume per rank per all-gather = `N_total × 9216 B`, independent of
world size.** At N=38 (438,976 atoms) that is **4.0 GB per gather**. With 4
layers, forward + backward:

| Scale | GP received volume / rank / step | / node / step (12 tiles) |
|---|---|---|
| N=38, any world | 4.0 GB × 4 × 2 = **32 GB** | **388 GB** |

Against an Aurora node injection bandwidth of ~200 GB/s aggregate, that is
**~1.9 s/step of pure inter-node all-to-all**, and it *does not improve* as
nodes are added — it grows with the system. Add the fact that
`all_gather_nodes` is an all-to-all collective appearing 8× per step in the
middle of the autograd graph, and inter-node GP is not viable.

This is not a tuning problem. It is the wrong asymptotic.

**Keep GP for intra-node use.** On one node the same traffic crosses a shared
Xe-Link fabric where it is already demonstrated to beat the FairChem reference
(235 s vs 258 s). Nothing in this plan discards that work.

---

## 3. The halo-depth trade

Under DD, each rank owns a spatial subdomain of edge `L` and must additionally
hold *ghost* atoms out to a halo depth `h`. Redundant compute is the volume
ratio:

```
R = ((L + 2h) / L)^3
```

Let `k` = number of halo exchanges performed *during* the forward pass. With
`k` exchanges the network only needs `n_layers / k` layers of locality between
exchanges, so:

```
h = (n_layers / k) × cutoff = (4 / k) × 6 Å
```

- `k = 1`: h = 24 Å — one halo, **zero mid-network communication**
- `k = 2`: h = 12 Å — exchange after layers 2 and 4
- `k = 4`: h = 6 Å — exchange after every layer (standard practice)

**All three are numerically exact.** The halo fully contains the receptive
field in each case; this is a compute-vs-communication trade, not an
accuracy trade.

For NaCl (ρ = 0.0446 atoms/Å³, N=38 → 214.3 Å box, 438,976 atoms), cubic node
split:

| Nodes | L (Å) | R, k=1 (h=24) | R, k=2 (h=12) | R, k=4 (h=6) |
|---:|---:|---:|---:|---:|
| 1 | 214.3 | 1.00 | 1.00 | 1.00 |
| 4 | 135.0 | 2.49 | 1.63 | **1.29** |
| 8 | 107.2 | 3.03 | 1.84 | **1.38** |
| 16 | 85.1 | 3.83 | 2.11 | **1.49** |
| 64 | 53.6 | 6.81 | 3.03 | **1.84** |

Parallel efficiency (compute only) at 64 nodes: 15% for `k=1`, 33% for `k=2`,
**54% for `k=4`**.

Communication cost of `k=4` at 16 nodes: halo atoms = 27,436 × 0.49 = 13,444,
× 9216 B = **124 MB per exchange**; 4 forward + 4 reverse (gradient
accumulation) = **~1.0 GB/node/step in FP64, ~0.5 GB in FP32**. At 200 GB/s
that is **~5 ms**, and even at a pessimistic single-NIC 25 GB/s it is **40 ms**
— against a measured ~18 s/step. Communication is **not** the bottleneck for
DD; redundant compute is.

**Conclusion: `k=4`, h=6 Å, per-layer halo exchange.** `k=1` is a valuable
zero-model-change stepping stone (§7 Phase A) and remains the right choice at
≤4 nodes.

---

## 4. Where the accuracy budget should go

Since the decomposition is exact, spend the 0.01 meV/atom (1e-5 eV/atom)
budget elsewhere. Ranked by payoff/risk:

1. **FP32 halo payload (exact-ish, free).** Halve halo bytes. Relative error
   ~1e-7 on boundary features only. Essentially free; do it unconditionally
   once measured.
2. **Overlap halo exchange with interior compute (zero error).** Partition each
   subdomain into interior (no ghost dependence) and boundary shells; launch
   the exchange, compute interior, then boundary. Hides essentially all of the
   40 ms. Do this before any lossy option.
3. **Mixed precision message passing.** `PRECISION_MIXED` already exists
   (`pair_uma.cpp:327`) but the extgraph path currently rejects it
   (`pair_uma.cpp:213-215`). Potential 2–4× on XPU. **Caution:** FP32
   accumulation error over 4 layers may well *exceed* 1e-5 eV/atom. This must
   be measured against the budget, not assumed. Keep force accumulation and the
   energy reduction in FP64 regardless.
4. **Sub-exact halo (stretch).** Because the radial envelope is attenuated
   approaching 6 Å, a halo of 5.0–5.5 Å per layer drops only weakly-weighted
   contributions. At `k=4`, h=5 Å at 16 nodes gives R=1.40 instead of 1.49.
   This is a genuine but small win for a real (measurable) error; only pursue
   if §3 redundancy proves limiting. **Gate it on a measured error curve
   E(h) and force conservation, not on intuition.**
5. **Do NOT** reuse stale halo features across MD steps. It breaks energy
   conservation and the failure mode (slow drift) is exactly the kind that
   passes a single-point parity gate and ruins a production trajectory.

---

## 5. Architecture

### 5.1 The key realization: stop fighting LAMMPS

`pair_uma.cpp:143-297` currently **all-gathers every atom onto every rank**
(`MPI_Allgatherv` at `:258,:264,:267`), sorts by tag (`:275`), evaluates the
full system on every rank, and then discards all but the owned slice
(`:302-311`). This is the *opposite* of LAMMPS's native spatial decomposition,
and it exists only to serve GP.

Under DD that entire block is **deleted**. LAMMPS already provides everything:

| Need | LAMMPS mechanism |
|---|---|
| Spatial decomposition | native, free |
| Ghost atoms to depth h | `comm_modify cutoff 24.0` (or 6.0 for k=4) |
| Feature halo exchange | `comm->forward_comm(this)` + `pack/unpack_forward_comm` |
| Gradient reverse accumulation | `comm->reverse_comm(this)` + `pack/unpack_reverse_comm` |
| Energy reduction | already sums `eng_vdwl` across ranks |
| Load balance | `fix balance` / RCB |

`Pair` exposes `pack_forward_comm`/`unpack_forward_comm`/`pack_reverse_comm`/
`unpack_reverse_comm` as virtuals precisely for this. Using them makes the pair
style idiomatic, composable with `fix balance`, and removes the rank-0-only
energy hack at `pair_uma.cpp:315`.

**Caveat:** LAMMPS comm buffers are host-side. Per-exchange that is a
device→host→device round trip of 124 MB (~4 ms at 50 GB/s) — acceptable, and
the simplest correct starting point. Phase C replaces the payload transport
with GPU-aware MPI / oneCCL point-to-point on device pointers, using LAMMPS
only for the ghost *index* bookkeeping.

### 5.2 Graph construction change

`build_ext_graph` (`pair_uma.cpp:617-725`) currently maps every ghost back to
its owned image via `owned_of_tag` (`:693`) so that only `nlocal` graph nodes
exist. **Under DD this is exactly wrong** — ghosts must be first-class graph
nodes carrying their own features.

New variant `build_ext_graph_dd`:
- Graph nodes = `nlocal + nghost` (owned first, ghosts after).
- Centers = owned atoms only, for the *final* layer; for intermediate layers
  centers include ghosts within the remaining receptive field.
- No `owned_of_tag` remap, no integer image recovery, **no cell offsets** —
  ghost coordinates are already unwrapped and absolute, so
  `edge_vec = x[j] - x[i]` directly. This *removes* the orthorhombic-only
  restriction at `:621-625` and the `std::lround` image recovery at `:706-708`.
  **Triclinic support falls out for free.**

This is a significant simplification, not just a port.

### 5.3 Differentiable halo exchange op

Add `uma_halo::exchange(Tensor x, int layer) -> Tensor`, registered on the
Autograd key, following the pattern already proven in `peer_context.cpp:139-160`:

- **forward:** owned features → ghost slots (LAMMPS forward comm).
- **backward:** ghost gradients → accumulate onto owners (reverse comm).

This is the exact adjoint pair `all_gather`/`all_reduce` already implemented at
`peer_context.cpp:105-116`, so the autograd structure is understood and tested
(`python/test_gather_bwd_semantics.py`). Reuse that test harness.

The traced model gains one `uma_halo::exchange` call per layer boundary,
injected by the exporter exactly as `uma_peer::all_gather_nodes` is today
(`export_blocks_xpu.py:1191`). **The export machinery needs no new
capability** — only a new op and a `k`-dependent injection point.

### 5.4 The one genuinely global dependency: MoLE — **RESOLVED**

`csd_embedding` + `set_MOLE_coefficients` (`export_blocks_xpu.py:626-672`)
produce a **system-level** embedding `sys_node_emb` from global composition.
Under DD no rank sees the global system.

**This is no longer an open risk.** nvalchemi hit exactly this problem and
their fix is readable in `nvalchemi/models/uma.py:786-868`
(`_distributed_set_mole_coefficients`). Their analysis, verbatim:

> The MoLE expert-mixing coefficients depend on a per-system **mean of the
> composition embedding**. Under domain decomposition the input carries this
> rank's `owned + ghost` atoms plus inert dead atoms (`Z=0`) from the caps
> padder; both pollute the stock mean (dead rows add a `Z=0` embedding, and a
> per-rank mean differs from the global one), **shifting every MoLE-linear
> weight by a small per-system amount**.

Their fix (`uma.py:845-860`):

```python
comp_by_atom = backbone_self.composition_embedding(atomic_numbers_full)
comp_sum = system_sum(comp_by_atom, batch_full, nsys, scope=Scope.OWNED)
ones     = comp_by_atom.new_ones(comp_by_atom.shape[0], 1)
count    = system_sum(ones, batch_full, nsys, scope=Scope.OWNED)
include_self = 1.0 if np.isclose(backbone_self.model_version, 1.0) else 0.0
composition  = comp_sum / (count + include_self).clamp_min(1.0)
# routing_mlp + coefficient norm unchanged
```

`system_sum(..., scope=Scope.OWNED)` sums **owned rows only** (dropping ghost
and dead rows) then all-reduces across the mesh
(`nvalchemi/distributed/_core/enums.py:36-39`).

**Four confirmed facts, and what each means for us:**

1. **Position-independent — M2 is answered.** The reduction is over
   `composition_embedding(atomic_numbers)`, a function of `Z` only. No
   positions enter. Our cache-once plan is sound.
2. **It is a *mean*, not a count — this changes our implementation.** Our
   original sketch allreduced a per-element atom-count histogram. That is
   sufficient (the mean is reconstructible from counts, since
   `composition_embedding` is a per-`Z` lookup: `mean = Σ_Z count_Z · emb(Z) / Σ_Z count_Z`),
   but note the denominator subtlety at `uma.py:853-856`: fairchem's
   `index_reduce(mean, include_self)` seeds an **extra zero row** on
   `model_version == 1.0`, so the denominator is `count + 1`, not `count`.
   **Get this wrong and every MoLE weight is off by a factor of
   `N/(N+1)`.** For a 46k-atom system that is a ~2e-5 relative shift — small,
   but systematic, and it will show up as a constant energy offset that a
   force-only gate would miss.
3. **Ghosts must be excluded.** The reduction is `Scope.OWNED`. Counting ghost
   atoms double-counts the overlap region and biases the mean. Our
   composition allreduce must therefore sum **owned atoms only**
   (`i < nlocal`), which is natural in LAMMPS.
4. **It is not differentiable in their implementation, and need not be.** The
   coefficients are treated as a forward-only quantity. Since composition is
   constant during NVT, no gradient flows through the reduction.

**Revised plan for us:** allreduce the per-element **owned** atom-count vector
once at setup and on composition change; reconstruct the mean with the
`include_self` correction; cache. Still off the per-step path. Downgraded from
**High risk** to **implementation detail with one specific trap**
(the `+1` denominator).

**Validation:** compare `sys_node_emb` computed under DD against the
single-rank value bitwise before comparing any energy. A wrong mean produces a
*plausible but wrong* energy that parity gates on a single rank count cannot
detect.

**Caveat on merged MoLE.** nvalchemi bypasses this fix entirely when
`merge_mole` is on (`uma.py:836-840`: falls back to stock when not on CUDA,
"the merged path is already exact"). Our exporter sets `merge_mole = False`
(`export_blocks_xpu.py:831`, and `metadata.json` confirms `"merge_mole": false`),
so **we are on the path that needs the fix.**

### 5.5 Memory

DD gives `O(N/P × R)` per rank, the same asymptotic as GP's edge sharding, so
the validated per-chunk activation checkpointing (`block_context.h`) applies
unchanged and the N=18/tile single-tile ceiling still governs. With 12 tiles ×
P nodes, capacity scales linearly in P for the first time.

---

## 6. Hybrid: DD across nodes, GP within a node

The lowest-risk composition, and the recommended target:

```
inter-node : spatial DD, k halo exchanges     (new)
intra-node : existing XCCL graph parallelism  (validated, unchanged)
```

Each node evaluates its (owned + halo) subsystem using the **already-validated**
12-tile GP machinery. Only the halo exchange and force reverse-comm are new
code. GP's `all_gather_nodes` then operates on `N_node × R` atoms over the fast
on-node fabric, where it is known to perform well, and never crosses a node
boundary.

This preserves every validated result and confines new code to one op plus one
graph-construction variant.

---

## 7. Phased plan

### Phase 0 — measurements that decide the design (do first, ~2 days)

No code changes. These numbers determine whether `k=1` suffices.

- **M1.** Instrument the exporter to record `sph_feature_size`,
  `sphere_channels`, and per-layer node-tensor bytes into `metadata.json`.
  Every estimate in §3 scales linearly with these. *(They are absent from
  today's metadata.)*
- ~~**M2.** Verify MoLE/`csd_embedding` depends only on composition.~~
  **DONE** — answered by nvalchemi `uma.py:786-868`; see §5.4. Remaining
  sub-task: confirm our `model_version` to fix the `include_self` denominator.
- **M3.** Measure achieved inter-node bandwidth on Aurora with an
  oneCCL/MPI point-to-point benchmark at 100 MB messages, 2–16 nodes.
- **M4.** Profile the current N=32 12-tile step: compute vs XCCL vs
  neighbor-build. Establishes the baseline that redundant halo compute is
  charged against.
- **M5.** Confirm ghost-atom scaling: run LAMMPS with
  `comm_modify cutoff 24.0` on N=32 across 2/4/8 nodes and record actual
  ghost counts vs the §3 model. Density models lie; measure.

**Decision gate:** if M5 shows `R ≤ 1.5` at the target node count with `k=1`,
implement Phase A only and stop. That is plausible at ≤4 nodes.

### Phase A — deep halo, `k=1`, no model changes (~1–2 weeks)

Exact, no export changes, no new ops. Each rank runs the **existing unmodified
single-system pipeline** on owned+halo atoms and keeps forces for owned atoms
only.

1. `comm_modify cutoff 24.0`; drop the `MPI_Allgatherv` block
   (`pair_uma.cpp:229-316`).
2. Implement `build_ext_graph_dd` (§5.2) — ghosts as real nodes.
3. Forces: keep rows `[0, nlocal)`; contributions to ghosts are owned by the
   neighbor rank, which computes them itself. **No reverse comm needed at
   `k=1`** — this is why Phase A is cheap.
4. Energy: each rank adds only its owned atoms' energy to `eng_vdwl`; LAMMPS
   sums. Remove the rank-0-only hack (`pair_uma.cpp:315`).
5. MoLE composition allreduce (§5.4).

**Gate A:** 2-node vs 1-node on N=18, `|dE| ≤ 1e-6 eV`, per-atom
`max|dF| ≤ 1e-5 eV/Å` on ≥100 atoms, AG=FD ≤ 1e-5. Must be *exact*, since
`k=1` is exact. Any discrepancy means the halo is incomplete.

**Expected:** works to ~4–8 nodes; enables N > 40 for the first time.

### Phase B — per-layer halo, `k=4` (~3–4 weeks)

1. `uma_halo::exchange` op + autograd (§5.3), reusing the
   `peer_context.cpp` pattern and the `test_gather_bwd_semantics.py` harness.
2. Exporter injects the op at layer boundaries; `k` recorded in
   `metadata.json` and validated against the runtime halo depth at load.
3. `pack/unpack_forward_comm` + `pack/unpack_reverse_comm` on `PairUMA`.
4. `comm_modify cutoff 6.0`.

**Gate B:** `k=4` vs `k=1` vs single-node, all three FP64-equivalent. Plus a
100-step NVT energy-conservation check (drift per atom per ps) — a
single-point gate cannot catch a wrong reverse comm.

### Phase C — performance (~2–3 weeks)

1. Interior/boundary overlap of exchange with compute (§4.2).
2. FP32 halo payload (§4.1).
3. GPU-aware point-to-point transport bypassing host staging (§5.1 caveat).
4. `fix balance` / RCB for non-uniform systems.
5. Revisit mixed precision (§4.3) against the measured error budget.

### Phase D — optional

- Sub-exact halo with a measured `E(h)` curve (§4.4).
- Pure DD intra-node (12 sub-domains) vs the GP hybrid — decide by measurement.

---

## 8. Reference: nvalchemi

NVIDIA's ALCHEMI Toolkit implements exactly this design and independently
corroborates the choice:

```python
domain_cfg = DomainConfig(cutoff=float(wrapper.cutoff), skin=0.5, mesh=mesh)
with DomainParallel(dynamics=integrator, config=domain_cfg, ...) as dynamics:
    owned = dynamics.partition(batch if dm.rank == 0 else None)
```

Note it passes the **single-layer** `wrapper.cutoff`, *not* `cutoff × n_layers`
— confirming per-layer halo exchange (`k = n_layers`) rather than a deep halo.
Its README documents "halo exchange and cross-rank reductions are automatic"
and lists "domain decomposition optimization" on the roadmap, i.e. it is beta.

### 8.1 They implement *both* strategies, and default UMA to halo

`docs/userguide/distributed.md` documents two `StrategyKind`s:

| Strategy | Layout | Their stated use |
|---|---|---|
| `halo` (**default**) | owned + spatial ghost rows | "scatter-heavy MPNN (MACE, NequIP, Allegro, ORB, **UMA**)" |
| `GRAPH_PARTITION` | node partition, **positions replicated**, per-layer feature all-gather | models that build their own NL internally |

`SPEC_UMA_HALO` is the shipped preset for UMA (`uma.py:1137`), and
`distribution_spec` defaults to halo with graph-partition only as an opt-in
(`uma.py:1090-1112`). **This is independent corroboration of the §2/§3
recommendation**: an organization that implemented both, for this exact model,
ships halo/DD as the default.

Their graph-partition description also confirms our §2 arithmetic — "the full
geometry **replicated**", "a per-message-passing-layer feature `all_gather`" —
i.e. precisely our current GP path, and they scope it to *memory* relief rather
than scaling.

### 8.2 Three transferable lessons

1. **Fixed-shape caps (`_UMAGraphPadder`).** They pad each rank's graph to a
   per-rank capacity cap so shapes stay static across MD steps, reaching a
   "recompile-free steady state". This is **the same fix as our P2.1 edge
   padding**, arrived at independently — and it confirms that padding is
   mandatory under DD, not optional. Their caps "grow a few times during
   warm-up and then settle": a good growth policy to copy.
2. **Their UMA backbone is halo-*unaware*.** `SPEC_UMA_HALO` uses "local
   scatter (eSCN backbone is halo-unaware)" (`distributed.md` preset table) —
   the model runs a normal forward over `owned + ghost` and DD is handled
   *only at the boundaries*: per-block ghost-row refresh, owned-only energy
   reduction, and ghost force contributions routed to owners in consolidation
   (`uma.py:1091-1098`). **This validates our Phase A design exactly**: run the
   unmodified pipeline on owned+halo and fix up the boundaries.
3. **Precision pinning under shape changes.** They warn that distributed vs
   single-process results diverge well beyond FP32 rounding because different
   tensor shapes select different (TF32 vs full-precision) kernels, and ship
   `pin_fp32()` to force agreement. Our artifacts are FP64 with `"tf32": false`
   (`metadata.json`) so we are insulated today — but this is a **direct warning
   about §4.3 mixed precision**: the moment we go FP32, the DD-vs-single-node
   gate will show shape-dependent divergence unrelated to the decomposition.
   Their conclusion is worth adopting: bitwise equality is impossible under DD
   regardless, because cross-rank reduction order differs, so the gate must
   always be a tolerance.

Actionable follow-ups:
- Read `nvalchemi/distributed/` for their interior/boundary overlap strategy
  and their ghost-gradient reduction; both are directly transferable.
- `benchmark/distributed/` gives a scaling baseline to compare against.
- The existing `src/ML-UMA/examples/nvalchemi_path/` harness already runs UMA
  through nvalchemi in env `nvalchemi312`. **Extend it to multi-GPU
  `DomainParallel` as an independent cross-check of Gate A/B** — a second
  implementation of the same physics is the strongest validation available.
- Note their `uma` extra is mutually exclusive with the CUDA extras
  (`nvalchemi_path/README.md`), so their DD path may not be exercisable with
  UMA without resolving that pin. Check before depending on it.

---

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| ~~MoLE has hidden position-dependent global state~~ | ~~High~~ → **Low** | **Resolved** (§5.4): nvalchemi `uma.py:786-868` confirms composition-only. One trap: the `include_self` `+1` denominator |
| `sphere_channels`/`lmax` larger than assumed → comm ×N | Medium | Task M1; all estimates scale linearly |
| Reverse comm gradient accumulation subtly wrong | **High** | Exact-vs-`k=1` gate + 100-step energy drift; single-point parity will *not* catch it |
| Host-staged halo dominates at scale | Medium | Phase C.3 GPU-aware transport |
| Redundant compute worse than modeled (non-cubic domains, poor balance) | Medium | Task M5 measures real ghost counts; `fix balance` |
| Traced-artifact chunk count is N-specific (`SESSION_RECOVERY:337`) — per-rank atom counts now vary and *drift* | **High** | **Blocked on `DEV_PLAN_code_quality.md` P2.1 (fixed-multiple edge padding).** Under DD every rank has a different, time-varying atom/edge count; the current N-specific artifact scheme cannot work. **Fix P2.1 first.** |
| Aurora inter-node collective performance unknown at these sizes | Medium | Task M3 |

**Hard dependency:** P2.1 (edge padding) from the code-quality plan is a
**prerequisite**, not a parallel track. DD makes per-rank edge counts vary
across ranks *and* across steps, which is precisely the failure mode that
already breaks N=24 today.

---

## 10. Summary

| Option | Comm scaling | Redundancy | Exact? | Verdict |
|---|---|---|---|---|
| Graph parallel (current) | `O(N_total)`/rank/layer | 1.0 | yes | **on-node only** |
| DD, k=1 (24 Å halo) | `O(surface)`, positions only | 2.5–6.8× | yes | **Phase A**, ≤8 nodes |
| DD, k=2 (12 Å) | `O(surface)` × 2 | 1.6–3.0× | yes | fallback |
| **DD, k=4 (6 Å)** | `O(surface)` × 4 | **1.3–1.8×** | **yes** | **target** |
| DD + sub-exact halo | slightly less | 1.2–1.7× | no | stretch only |

Because DD is exact, the 0.01 meV/atom budget is left entirely available for
precision reduction (§4) — a much better use of it than paying for the
decomposition.

**Immediate next actions:** M1, M2, M5 (Phase 0), and P2.1 from
`DEV_PLAN_code_quality.md`. Do not start Phase A before M2 and P2.1.

---
---

# PART II — Implementation session log (2026-08-27)

**Branch:** `uma-multinode-dd` (off `1a8f91f608`, base `cfd1657b89`).
**Author:** implementation agent. **Purpose of this section:** hand a reviewing
agent the complete design evolution, every fix, all empirical results, and the
one open bug — with exact file/line/commit pointers — so the result can be
independently reviewed and the remaining bug resolved.

**Target (unchanged):** N=32 NaCl (262,144 atoms) single point on **2 nodes**
(24 XPU tiles), matching the **12-tile 1-node ASE-GP oracle**
(`hen/pbs/out/ase12_n32`: E = −885377.0600366206 eV, per-atom FP64 forces in
`forces_w12.npy`) via `scripts/parity_vs_asegp.py`:
`|dE| ≤ 1e-3 meV/atom` and per-atom `max|dF| ≤ 1e-5 eV/Å` over all atoms.

## II.1 Design decisions taken (and why)

1. **k=4 per-layer halo, 6 Å, chosen over k=1 (24 Å).** PART I §3 math plus the
   measured single-tile ceiling (~46,656 atoms) show k=1's 24 Å halo makes each
   2-node rank hold ~62k atoms → OOM (see §II.7 "N=32 fits 1 node not 2"
   derivation, verified on-node: 2-node k=1 nall≈62,086 > ceiling). k=4's 6 Å
   halo gives nall≈18–20k/rank (R≈1.7×) → fits. This is why the user directed
   "halo 6 Å and k=4."

2. **DD is a NEW code path, `UMA_DD=1`, NOT the existing GP-over-MPI (`mn_active`)
   path.** The GP path all-gathers every atom to every rank
   (`pair_uma.cpp` mn_active block); DD spatially decomposes. `load_predictor`
   was changed so DD ranks build the **single-tile** `Predictor`, not
   `MpiPeerPredictor` (`pair_uma.cpp` load_predictor: `if (dd_active_) {} else
   if (nprocs>1)`).

3. **Ghosts are first-class graph nodes with absolute coords.** `build_dd_graph`
   uses a `REQ_FULL|REQ_GHOST` neighbor list so every owned+ghost node is a
   center; `edge_vec = x[j]-x[i]` directly (zero cell offsets), which also
   removes the orthorhombic restriction (triclinic works for free).

4. **Per-atom energy (E1) is REQUIRED for forces, not just energy** (key finding,
   §II.4). Each rank backprops from `sum(node_energy[0:nlocal])`, NOT the whole
   owned+ghost subsystem energy (which injects spurious ghost-energy force
   gradients). Uses the pre-existing `NodeEnergyExportWrapper`
   (`export_wrapper.py:152`).

5. **Edge padding (P2.1) to a fixed cap** so the traced per-chunk loop count is
   rank-invariant (per-rank edge counts differ across the 2×3×4 rank grid).
   Cap = 917504 (14×65536) covers the 2-node worst rank (~790k edges).

## II.2 What was built (files + commits)

| Component | File(s) | Commit |
|---|---|---|
| DD compute path, graph build, MoLE allreduce | `pair_uma.cpp` run_compute_dd/build_dd_graph/mole_composition_allreduce; `pair_uma.h` | 4e18a8bdcb |
| `uma_halo::exchange` autograd op + HaloContext | `uma-engine/include/uma/halo_context.h`, `src/halo_context.cpp`; CMake | 830fe11679 |
| LAMMPS comm transport (pack/unpack fwd+rev) | `pair_uma.cpp` install_halo_callbacks + pack/unpack_*_comm | 830fe11679 |
| Exporter halo-op injection (UMA_DD_HALO=1) | `export_blocks_xpu.py` block loop | 830fe11679 |
| Edge padding runtime + export | `pair_uma.cpp`, `export_blocks_xpu.py` | 2277b1bf12, f37a251ce2 |
| Per-atom energy (E1): predict_body_dd | `predictor.{h,cpp}` predict_host_extgraph_dd/predict_body_dd; `pair_uma.cpp` | 835ec7fbab |
| Halo-op Python registration for trace | `uma-engine/python/uma_halo_ops.py` | db33f2f9c8 |
| dd_halo_width in metadata (comm sizing) | `metadata.{h,cpp}`, `export_blocks_xpu.py` | e2356a4eb5 |
| Diagnostics (self-tests + norms) | `pair_uma.cpp`, `halo_context.cpp` | abaaab1f1b, 3c9445a9b5 |
| Example: input, data gen, run/export scripts | `examples/multinode-dd/*` | multiple |

**Design docs:** `examples/multinode-dd/DESIGN_k4_halo.md` (op semantics,
transport, E1), `examples/multinode-dd/README.md` (run recipe).

## II.3 Bugs found and fixed on Aurora (in order)

Each was a real failure caught by running; all fixed and committed.

1. **Python trace: `uma_halo.exchange` not registered.** torch.jit.trace runs
   without libuma_engine → `AttributeError`. Fix: `uma_halo_ops.py` (torch.library
   identity op for tracing; C++ does real movement at runtime). Commit db33f2f9c8.
2. **Multi-node MPI init fails** (`OFI fi_getinfo: No data available`). Conda
   ships `libfabric.so.1` WITHOUT the Aurora cxi (Slingshot) provider; it shadowed
   the system one. Fix: prepend `/opt/cray/libfabric/*/lib64`, `FI_PROVIDER=cxi`
   in `run_dd_parity.pbs`. Commit 21194455bb. (Single-node GP worked only because
   it used `FI_PROVIDER=tcp`.)
3. **Segfault in `build_dd_graph`.** REQ_GHOST full lists can list neighbors
   outside [0,nall); reading `x[j]` before the bounds check segfaulted. Fix:
   bounds-check `j` before `x[j]`; skip out-of-range (rim ghosts, acceptable for
   k=4). Commit d4686450b2.
4. **Segfault in predict (comm buffer overrun).** `comm_forward` was set to
   1152 doubles/atom INSIDE the op callback, AFTER `Comm::init()` had sized
   `buf_send` for the default small width → `pack_forward_comm` overran → SIGSEGV.
   Fix: export `dd_halo_width` (=sph_feature_size×sphere_channels=9×128=1152) in
   metadata; set `comm_forward/comm_reverse` in `init_style` so `Comm::init`
   sizes the buffer. Commit e2356a4eb5.
5. **Garbage E/F (E=+4.7M eV, force cos=−0.08).** Padded edges were dummy→dummy
   self-loops with edge_distance=0 (NOT >cutoff): r=0 poisons the radial/SO2 edge
   basis and corrupts the whole batch. Fix: inert padding = neighbor atom 0 (real),
   center = dummy at (1e6,1e6,1e6) → edge_distance≫cutoff → envelope 0 → zero
   message; dummy center E/F discarded. Runtime + export matched. Commit f37a251ce2.

## II.4 KEY FINDING — per-atom energy required for FORCES

Under DD each rank's model returns an energy over its owned+ghost subsystem.
Backpropagating that whole-subsystem energy is WRONG for forces: the ghost-energy
terms inject spurious gradients (a ghost's energy is also, correctly, counted on
its owner). Correct DD forces require backprop from the OWNED-only energy sum
`E_owned(r)=Σ_{a∈owned} e_a`; the halo backward then routes cross-rank
contributions (an atom that is a ghost elsewhere) to its owner. Therefore per-atom
energy is a hard prerequisite for the FORCE gate, not merely the energy gate.
Implemented via `NodeEnergyExportWrapper` (returns `(node_energy[N], total)`) and
`predict_body_dd` (backprops `sum(node_energy[0:nlocal])`; per-atom denorm; per-atom
element refs — NOT the scalar `undo_element_references` scatter_add, which would
pile all refs onto atom 0). See `predictor.cpp:predict_body_dd`, commit 835ec7fbab.

## II.5 Current empirical results (N=32, 2 nodes × 12 tiles = 24 ranks)

Pipeline runs end-to-end: build OK, export OK, 24-rank launch OK, single point
completes in ~31–41 s, forces dumped for all 262,144 atoms.

Parity vs 12-tile ASE-GP oracle:

| Config | E_lmp (eV) | dE (meV/atom) | force cos | rms\|dF\| | max\|dF\| |
|---|---|---|---|---|---|
| **halo ON (k=4)** | −882,333.37 | 11.61 | **0.644** | 0.140 | 1.28 |
| **halo OFF** (`UMA_DD_NO_HALO=1`) | −882,195.43 | 12.14 | **0.803** | 0.089 | 1.16 |
| oracle | −885,377.06 | 0 | 1.0 | 0 | 0 |

**Both gates FAIL.** Energy is 0.34% off; forces are qualitatively right but far
from parity. **Turning the halo ON makes forces WORSE (0.80 → 0.64)** — the
central anomaly.

## II.6 Isolation tests — every primitive proven CORRECT

Built-in self-tests (`UMA_DD_HALO_TEST=1`, `pair_uma.cpp`), run on the REAL
N=32 2-node comm plan:

| Test | Result | Meaning |
|---|---|---|
| Forward comm owner→ghost | `ghost != owner_tag: 0` | transport + pack/unpack + row ordering exact |
| Reverse comm ghost→owner | `sum(owned)=217213 == total ghosts=217213` | adjoint accumulation exact |
| Ghost Z vs owner Z | `mismatch: 0` | ghost embeddings equal owners' |
| Per-layer ghost delta (`UMA_DD_DEBUG`) | pre-block-0 = 13.6% | EXPECTED: edge-degree prologue makes ghosts differ before the first exchange |

Conclusion: the halo transport, buffer layout, node/Z alignment, and the
forward/backward VALUE adjoint are all correct. The exchange refreshes ghosts as
designed. The pre-block-0 13.6% change is legitimate (the edge-degree prologue
runs before the block loop and gives ghosts incomplete-neighbor values that the
exchange correctly refreshes).

## II.7 OPEN BUG — hypothesis for the reviewer

**Symptom:** every DD primitive is provably correct, yet composing the halo
exchange into the model degrades forces (cos 0.80→0.64) while barely changing
energy (a forward-only quantity). Energy-good / forces-bad is a gradient bug
signature.

**Leading hypothesis:** interaction between TWO custom autograd mechanisms —
`HaloExchangeFn` (host round-trip XPU→CPU→XPU inside
`AutoDispatchBelowADInplaceOrView`) and per-chunk activation checkpointing
(`uma_ckpt::chunk`/`block`/`edge_degree` recompute forward in backward,
`predictor.cpp:predict_body`). If the halo op's ghost refresh is not reproduced
consistently between the original forward and the checkpoint RECOMPUTE, gradients
through ghost features are wrong → force error that leaves energy intact.

**Confirming test (not yet run):** set `UMA_NO_RECOMPUTE=1` (and `_BLOCK/_CHUNK/
_EDEG`) to disable per-chunk AC (retain activations). If forces then improve
toward cos→1.0, the halo×checkpoint composition is confirmed as the bug. NOTE:
N=32 with AC OFF may OOM a tile — may need to confirm on the largest N that fits
AC-off, per the user's "N=32 only" preference this must be weighed.

**Secondary hypotheses to rule out:**
- halo-OFF is only cos=0.80 (not ~1.0): an isolated 6.5 Å subdomain with 4 layers
  is inherently wrong at boundaries (k=1 with a 6 Å halo is under-resolved), so
  0.80 is the expected "no-refresh" baseline, not a second bug. A CORRECT k=4
  halo should lift this to ~1.0.
- MoLE per-rank composition mean: currently the traced model computes composition
  from the owned+ghost atomic_numbers it is handed (homogeneous-NaCl
  approximation). `mole_composition_allreduce()` computes the exact global per-Z
  counts but is DIAGNOSTIC ONLY — not yet fed into the traced MoLE. For uniform
  NaCl the owned+ghost vs global mean differ only slightly; unlikely to explain
  cos=0.64 but should be quantified. (See §5.4 for the nvalchemi-exact fix.)
- Backward zeroing of ghost rows in `reverse_exchange` (halo_context.cpp): proven
  correct by the reverse self-test, but the interaction with the block residual
  (`x_out = edgewise(x_full) + x_res`, where the exchange replaced x_res ghost
  rows) across the recompute is the untested surface.

**Recommended path for the reviewer:**
1. Run the `UMA_NO_RECOMPUTE=1` A/B to confirm/deny the halo×AC hypothesis.
2. If confirmed: make the halo op checkpoint-aware, OR restructure so ghost
   refresh happens OUTSIDE the checkpointed region (exchange results fed as
   block inputs that the recompute treats as constants), OR verify the recompute
   re-invokes `uma_halo::exchange` identically (it should, since it is in the
   traced graph — check whether the recompute path actually re-runs the op or
   uses saved tensors).
3. Independently: gradient-check a TINY DD case (2 ranks, ~10 atoms per rank, a
   system that fits AC-off) with finite differences on owned-atom forces — the
   cheapest exact test of the full DD autograd chain.

## II.8 How to reproduce (all on Aurora)

```bash
# build (login node OK; ~15 min):
export PATH=/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/cmake-3.31.11-pss2phi/bin:$PATH
bash scripts/phase6_build_lammps_xccl.sh            # -> build-lmp-xccl/lmp

# export k=4 DD artifact (debug queue, ~10 min):
qsub src/ML-UMA/examples/multinode-dd/export_dd.pbs # -> scripts/out/dd/n32_k4_cap917504

# 2-node N=32 parity (debug queue supports 2 nodes):
qsub -v N=32,NODES=2,TPN=12 src/ML-UMA/examples/multinode-dd/run_dd_debug_q.pbs

# diagnostics:
#   UMA_DD_HALO_TEST=1  -> forward/reverse/Z self-tests (run_dd_halotest.pbs)
#   UMA_DD_NO_HALO=1    -> A/B with exchange disabled  (run_dd_nohalo.pbs)
#   UMA_DD_DEBUG=1      -> per-stage + per-layer ghost-delta prints
```

Artifact metadata to verify: `dd_halo:true, dd_k:4, returns_node_energy:true,
dd_halo_width:1152, sph_feature_size:9, sphere_channels:128`.

## II.9 Status summary for review

- **Working & verified:** build; 2-node MPI; DD graph build; ghost neighbor list;
  edge padding (rank-invariant chunk count); comm buffer sizing; forward/reverse
  halo transport (exact); node/Z ordering (exact); per-atom energy path; additive
  global energy (eng_vdwl sum); end-to-end single point at N=32 on 2 nodes.
- **Correct-by-test but composite fails:** the k=4 halo exchange (value adjoint
  proven, yet degrades forces when composed with the checkpointed model).
- **Not passing:** parity gates (energy 0.34% off, forces cos=0.64).
- **Open bug:** halo × activation-checkpoint autograd composition (hypothesis in
  §II.7); confirming test identified and not yet run.
- **Not yet addressed:** exact MoLE global-mean wiring (currently homogeneous
  approximation); optimization/timing (deferred until parity passes).
