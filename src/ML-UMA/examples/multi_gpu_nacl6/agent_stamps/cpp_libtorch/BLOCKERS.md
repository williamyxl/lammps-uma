# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~01:49 CDT  
**Track:** `cpp_libtorch`

## B1 — `model_state.pt` is weights-only (architecture gap)

Product path: export-time FairChem → per-rank TorchScript with `uma_peer` → C++ process-per-rank runtime. **Not Ray / not Python GP as product.**

## B2 — Serial `model_traced.pt` cannot host MP collectives

Use `model_mp_w{N}_r*` / `model_mp_w{N}_n{NATOMS}_r*`.

## B4 / B5 / B5b / B6 — **resolved**

Process-per-rank + Autograd `uma_peer` + force regime: **all_reduce bwd + force SUM + escale=1/world**.

## B7 — MP TorchScript is **n_atoms-specific** (baked `gp_node_offset`)

**Mitigation:** `model_mp_w{W}_n{N}_r{R}.pt` + `UMA_MP_NATOMS=N`. Legacy `model_mp_w{W}_r*.pt` = n=64.

## Phase 2b — engine/CLI E+F **GREEN** (devices=2 and 4)

| Structure | devices | Job | dE_d1 | max\|ΔF\| | dE_ase |
|-----------|---------|-----|-------|----------|--------|
| nacl64 | 2 | `20925398` | 0 | 5.3e-16 | — |
| NaCl6 1728 | 2 | `20925457` | 1.8e-12 | 5.3e-16 | ≈1.2e-10 |
| nacl64 | 4 | `20925504` | 0 | 6.7e-16 | — |
| NaCl6 1728 | 4 | `20925506` | 1.8e-12 | 5.8e-16 | 1.2e-10 |

## Phase 3 — LAMMPS end-to-end **GREEN**

`lmp -k on g N -sf kk` + `pair_style uma/kk precision double devices N` (1 MPI rank), FP64, `gp=kokkos_libtorch_vesin`.

| devices | Job | Structure | dE_d1 | max\|ΔF\| | dE_ase | pair ms/eval* |
|---------|-----|-----------|-------|----------|--------|---------------|
| 2 | `20925747` | NaCl6 1728 | 9.1e-13 | **0** | 1.2e-10 | ≈361 |
| 4 | `20925801` | NaCl6 1728 | 2.7e-12 | **0** | 1.2e-10 | ≈473 |

\*Honest pair-path timer from `run_multigpu` (not SLURM wall/N_TIMING, which includes setup).

Gates: `lammps_gate_w2_20925747/gate.json`, `lammps_gate_w4_20925801/gate.json`.

### Optional next

- Unbake `gp_node_offset` for size-agnostic MP artifacts.
- Stronger multi-GPU timing study (warm SP only; do not invent Ray/MPI-GP).
