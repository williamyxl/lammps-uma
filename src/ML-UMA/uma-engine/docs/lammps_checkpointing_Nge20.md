# LAMMPS + activation checkpointing at N>=20 (4-GPU eager GP) — results

Goal: run NaCl NxNxN (N=20..25) in LAMMPS with activation checkpointing on 4
GPUs, report energy / per-atom force / timing, and compare vs ASE-FairChem FP64.

## Implementation

`UMA_EAGER_CKPT=1` + `pair_style uma precision double devices 4` now routes
through the Ray-free **native GP worker** (`uma_native_gp_worker.py`): one process
forks `workers` GPU rank processes with host-staged (`SharedGatherSlot`)
collectives, eager FairChem model, `activation_checkpointing=True`. (Single-GPU
`devices 1` uses `uma_gp_worker.py`.) Wired via `graph_parallel.cpp::create`
(allows num_devices>1 with checkpointing) + `predictor.cpp` (UMA_EAGER_CKPT).

## Multi-node (MpiPeerPredictor) checkpointing (rev 3)

The multi-node LAMMPS path (`pair_uma` with nprocs>1 -> `MpiPeerPredictor`, C++
NCCL edge-parallel) loads a **traced** shard and so had no checkpointing ->
OOM at large N. Implemented **C++ gradient checkpointing** around the traced
module forward: a custom `torch::autograd::Function` (`CheckpointModuleFn`) runs
the module under NoGrad in forward (no retained activations) and RECOMPUTES it
with grad in backward -> ~3x less activation memory, bit-exact, works with the
opaque traced module. Default ON for the MN path (`UMA_MN_CKPT=0` to disable).
(mpi_peer_predictor.cpp)

**Remaining blocker at N=21 = the SHARD EXPORT, not the runtime.** The runtime
checkpointing is correct, but the traced w8_n{N} shards must first be EXPORTED,
and `export_mp_artifact.py` traces the model on the full N-atom geometry with
`activation_checkpointing=False` (tracing cannot capture torch.utils.checkpoint)
-> the EXPORT itself OOMs at N=74,088 (needs ~40 GiB, tried 5.09 GiB alloc with
1.39 GiB free). So:
  - MN runtime checkpointing: DONE (helps run-time memory for any N with shards).
  - N=21 end-to-end on the MN path: BLOCKED by the export OOM (traced export
    can't checkpoint). Tracing-vs-checkpointing wall, now at export time.

Ways past the export blocker (future work):
  1. Export w8 shards with dynamic atom-count axis so a small-N (fits) export
     runs at N=21 (requires tracing with dynamic shapes; UMA graph currently
     bakes shapes).
  2. Checkpoint the export's forward (needs a scriptable/checkpointable export,
     i.e. not torch.jit.trace).
  3. Use the eager multi-node dist-GP path (no traced shards) once the cross-node
     dist-GP hang (see two_node_investigation.md) is resolved.

The single-node checkpointing path (below) is unaffected and works to N=14 on 1
GPU / N=21 on 4 GPU (1 node).

## RESOLVED (rev 2): forces now bit-exact via real torch.distributed GP

The force bug below was root-caused to the **host-staged kgp SharedGatherSlot
emulation** of the GP collectives (not the GP math): a diagnostic showed real
torch.distributed GP forces are bit-exact vs serial (dF ~6e-16), while kgp gave
~13% partial forces. Fix: a new worker `uma_dist_gp_worker.py` runs the eager
checkpointed model over **real torch.distributed + FairChem gp_utils** (rank 0
speaks the C++ pipe protocol and forks ranks 1..W-1 into the same NCCL group).
`graph_parallel.cpp` selects it for multi-GPU checkpointing.

Corrected results (LAMMPS + checkpointing, 4 GPU, vs ASE-FC FP64):

| N | atoms | box | E_lammps (eV) | \|dE\| | max\|dF\| | mean\|dF\| | SP ms/step |
|--:|------:|----:|--------------:|------:|----------:|-----------:|-----------:|
| 8  | 4,096  | 45 A  | -13841.956101 | 3.6e-12 | **5.7e-16** | - | - |
| 20 | 64,000 | 113 A | -216283.022184 | 1.8e-10 | **6.1e-14** | 3.9e-15 | 4,858 |
| 21 | 74,088 | 118 A | -250375.822189 | 8.7e-11 | **7.1e-14** | 2.0e-14 | 5,536 |
| 22 | 85,184 | 124 A | — | — | — | — | **OOM in LAMMPS** |

### N=21 full comparison vs ASE (energy, per-atom force, timing)

NaCl 21^3 = 74,088 atoms, ~118 A. Both: FP64, task=omat, 4x A100 graph-parallel
+ activation checkpointing, identical geometry.

| metric | LAMMPS-UMA (libtorch) | ASE-FairChem | difference |
|---|---:|---:|---:|
| Energy (eV) | -250375.82218906094 | -250375.82218906110 | \|dE\| = 8.7e-11 (1.2e-15/atom) |
| Per-atom force | \|F\|max = 0.4576 eV/A | \|F\|max = 0.4576 eV/A | max\|dF\|=7.1e-14, mean\|dF\|=2.0e-14 |
| Timing / step | 5,536 ms (NVT loop) | 5,759 ms/SP | LAMMPS ~1.04x faster (noise) |

At 74,088 atoms (9x the 4-GPU baseline ceiling of 8,000; impossible without
checkpointing), LAMMPS-UMA matches ASE-FairChem FP64 to the machine floor in
BOTH energy and per-atom force, at the same per-step cost. Both run the same
eager checkpointed model over the same 4-GPU torch.distributed GP path, so the
~0.2 s timing gap is run-to-run noise (ASE's number also includes the ASE
calculator wrapper + per-iter position perturbation).

- **Energy + per-atom forces now match ASE to the FP64 floor.**
- **LAMMPS 4-GPU checkpointing ceiling = N=21 (74,088)**; N=22 (85,184) fits the
  pure model (ASE) but OOMs in LAMMPS (~34 GiB by torch + LAMMPS/C++ engine
  overhead on GPU 0 leaves too little headroom). So the LAMMPS ceiling is one N
  below the model ceiling.
- SP ~5 s/step at 64-74k on 4-GPU dist GP + checkpointing.

---

## (Historical) Original bug: energy matched ASE, forces WRONG

| N | atoms | box | E_lammps (eV) | E_ase (eV) | \|dE\| | max\|dF\| | NVT ms/step |
|--:|------:|----:|--------------:|-----------:|------:|----------:|------------:|
| 20 | 64,000 | 112.8 A | -216283.022184451 | -216283.022184451 | **2.3e-10** | **6.1e-2** | 68,103 (68 s/step) |
| 21 | 74,088 | 118.4 A | -250375.822189 | (walltime) | - | - | - |

- **Energy is essentially exact** vs ASE (dE 2.3e-10 eV at 64k atoms) — the
  forward GP collectives (energy) are correct, and checkpointing is bit-exact.
- **Per-atom forces are systematically wrong**: max|dF| 6.1e-2, mean 2.6e-2
  eV/A, ALL 64,000 atoms differ (none < 1e-6). This is NOT sorting/indexing.
- **Timing**: 68 s/step at 64k on 4-GPU host-staged GP — very slow (host-staged
  collectives + eager + checkpointing recompute). Debug walltime (1 h) only fit
  N=20 fully + N=21 SP.

## Root cause (force bug)

Correct energy + systematically wrong forces = the **backward (force-gradient)
reduction in the native GP worker is incomplete**. Forces come from rank 0 of
`unit.predict` (uma_native_gp_worker.py:136); their correctness depends on
`kokkos_gp_runtime.py`'s autograd Function patches (`_ReduceFromMP`,
`peer_all_reduce_sum`, gather backward). One of those backward paths does not
fully reduce/scale the per-atom force gradient across the 4 GPUs.

This is a pre-existing bug in the **host-staged native Python GP worker**, a path
that was not exercised by the earlier exact 4-GPU parity (which used the C++
traced MP or the C++ MpiPeerPredictor edge-parallel path — both gave dF ~1e-15).

## Status vs the question

- **"Test N>=20 in LAMMPS+checkpointing"**: N=20 (64,000) and N=21 (74,088) RUN
  in LAMMPS with checkpointing (they fit — baseline OOMs at N=11). Energy is
  correct.
- **Forces**: NOT correct through the native GP worker (backward reduction bug).
  So N>=20 in LAMMPS+checkpointing is **not yet force-accurate**.
- **N=22..25**: not completed (68 s/step exceeded debug walltime; and blocked on
  the force bug anyway).

## Next

Fix the force-gradient reduction in `kokkos_gp_runtime.py` (native GP backward):
ensure the per-atom force gradient is all-reduced/scaled consistently with the
energy path, matching the C++ MpiPeerPredictor recipe (energy scaled 1/world for
the backward, forces summed across ranks). Then re-run N=20..25 for force parity
and timing. Alternatively, drive the eager checkpointed model through the proven
C++ MpiPeerPredictor collective (NCCL) instead of the host-staged kgp path.

## Reproduce
`polaris/pbs/lammps_ckpt_sweep_4gpu.pbs` (M3_SIZES, M3_NVT_STEPS) ->
`ase_ckpt_ref.py`, `lmp_ckpt_report.py`.
