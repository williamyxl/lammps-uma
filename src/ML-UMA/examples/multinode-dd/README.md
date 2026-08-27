# Multi-node spatial domain decomposition (DD) — Phase A

Branch: `uma-multinode-dd`. Goal: run the N=32 NaCl single point across multiple
nodes and match the **12-tile (1-node) ASE-GP oracle** (`hen/pbs/out/ase12_n32`,
E = −885377.0600366206 eV, per-atom FP64 forces in `forces_w12.npy`) on the
force parity gate first, then optimize timing while holding parity.

## What is implemented

A domain-decomposition path in `pair_style uma`, selected by env `UMA_DD=1`,
independent of the existing graph-parallel-over-MPI (`mn_active`) path:

- `pair_uma.cpp:run_compute_dd` — each MPI rank owns a LAMMPS subdomain,
  evaluates the **single-tile** `Predictor` over its **owned + ghost** atoms, and
  keeps forces for owned atoms only (exact at k=1, no reverse comm).
- `pair_uma.cpp:build_dd_graph` — consumes a `REQ_FULL|REQ_GHOST` neighbor list
  so every owned+ghost node is a center; edges are absolute (`edge_vec =
  x[j]-x[i]`, zero cell offsets), which also removes the orthorhombic-only
  restriction of `build_ext_graph`.
- `pair_uma.cpp:mole_composition_allreduce` — global per-Z owned-atom counts via
  `MPI_Allreduce` (diagnostic + hook for the exact MoLE fix).
- `in.dd_sp` sets `comm_modify cutoff 24.0` = num_layers(4) × cutoff(6) so LAMMPS
  supplies ghosts to the full receptive field.
- `make_data.py` is bit-identical (geometry + atom ordering) to the oracle
  builder, so `parity_vs_asegp.py` lines up atom-for-atom.
- `run_dd_parity.pbs` runs DD and calls `parity_vs_asegp.py` in force-only mode
  (`UMA_DD_SKIP_ENERGY=1`).

## Two BLOCKERS found before the first run (report)

### B1 — deep 24 Å halo does not fit a tile at 2 nodes for N=32

Per-rank owned+ghost atom count (NaCl ρ=0.0446, box 180.5 Å, 24 Å halo):

| decomposition | Lsub | owned/rank | **owned+ghost/rank** | R | fits tile (~46,656)? |
|---|---|---|---|---|---|
| 2 nodes × 12 = 24 tiles | 62.6 Å | 10,923 | **≈ 60,277** | 5.5× | **NO (OOM likely)** |
| 4 nodes × 12 = 48 tiles | 49.7 Å | 5,461 | ≈ 41,535 | 7.6× | marginal |
| 8 nodes × 12 = 96 tiles | 39.4 Å | 2,731 | ≈ 29,787 | 10.9× | yes |

The single-tile traced ceiling is N=18 ≈ 46,656 atoms. At **2 nodes the halo
alone pushes each rank to ~60k atoms**, over the ceiling. This is the deep-halo
(k=1) redundancy the DD plan predicted (§3). Options:

- **(a) Run k=1 at ≥4 nodes** for N=32 (physically the same system, more nodes).
  The user asked for 2-node first; k=1 cannot hold N=32 on 2 nodes.
- **(b) Implement k=4 per-layer halo exchange (Phase B)** — 6 Å halo instead of
  24 Å → per-rank nall ≈ owned + thin shell, fits 2 nodes easily. Requires the
  differentiable `uma_halo::exchange` op. This is the real target of the plan;
  k=1 was only ever the ≤4-node stepping stone.

### B2 — traced artifact chunk count is N-specific (P2.1)

Each rank runs the single-tile model on ~60k atoms with a per-rank **variable**
edge count. The traced artifact bakes its edge-chunk count at the trace-N
(`extract_uma_artifact.sh`: "chunk count is baked at N"). A per-rank atom count
that differs from the trace-N trips `Expected K elements in a list but found K±1`
— the same failure as the N=24 NVT bug. Under DD per-rank counts vary across
ranks and steps, so **P2.1 fixed-multiple edge padding is a hard prerequisite**,
exactly as the plan warned. A single-tile N=32 artifact also does not currently
exist (only the GP w*/r* tree), so one must be exported sized for the worst-case
per-rank `nall` (or, with padding, for a fixed multiple).

## Recommended path (pending user decision)

1. **Short term to get a 2-node parity number:** implement k=4 halo exchange
   (Phase B). This is the only way N=32 fits 2 nodes and is the plan's real
   target. Larger change but correct-by-construction and exact.
2. **Or, to validate the k=1 code now:** run at **4–8 nodes** (fits the tile),
   after (i) exporting a single-tile artifact sized to worst-case per-rank nall
   and (ii) landing P2.1 edge padding so per-rank count drift does not crash.

Both need P2.1 (B2). B1 decides node count (k=1) vs implementing k=4.

## Files

```
in.dd_sp              single-point DD input (comm_modify cutoff 24.0, run 0)
make_data.py          oracle-identical NaCl builder
run_dd_parity.pbs     2-node run + force-parity vs ase12_n32 oracle
```
