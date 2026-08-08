# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~01:50 CDT  
**Track:** `cpp_libtorch`  
**Status:** Phase 3 **DONE** · Phase 4 report **landed** · Phase 5 multi-node **later / out of scope**

## B1 — `model_state.pt` is weights-only (architecture gap)

Product path: export-time FairChem → per-rank TorchScript with `uma_peer` → C++ process-per-rank runtime. **Not Ray / not Python GP as product.**

## B2 — Serial `model_traced.pt` cannot host MP collectives

Use `model_mp_w{N}_r*` / `model_mp_w{N}_n{NATOMS}_r*`.

## B4 / B5 / B5b / B6 — **resolved**

Process-per-rank + Autograd `uma_peer` + force regime: **all_reduce bwd + force SUM + escale=1/world**.

## B7 — MP TorchScript is **n_atoms-specific** (baked `gp_node_offset`)

**Mitigation:** `model_mp_w{W}_n{N}_r{R}.pt` + `UMA_MP_NATOMS=N`. Legacy `model_mp_w{W}_r*.pt` = n=64. Unbake deferred (not required for green gates).

## Phase 2b — engine/CLI E+F **GREEN**

| Structure | devices | Job | dE_d1 | max\|ΔF\| | dE_ase |
|-----------|---------|-----|-------|----------|--------|
| nacl64 | 2 | `20925398` | 0 | 5.3e-16 | — |
| NaCl6 1728 | 2 | `20925457` | 1.8e-12 | 5.3e-16 | ≈1.2e-10 |
| nacl64 | 4 | `20925504` | 0 | 6.7e-16 | — |
| NaCl6 1728 | 4 | `20925506` | 1.8e-12 | 5.8e-16 | 1.2e-10 |

## Phase 3 — LAMMPS end-to-end **GREEN** (DONE)

`lmp -k on g N -sf kk` + `pair_style uma/kk … devices N` (1 MPI), FP64, `gp=kokkos_libtorch_vesin`. Commit `5513482e9b`.

| devices | Job | dE_d1 | max\|ΔF\| | dE_ase | pair ms |
|---------|-----|-------|----------|--------|---------|
| 2 | `20925747` | 9.1e-13 | 0 | 1.2e-10 | ≈361 |
| 4 | `20925801` | 2.7e-12 | 0 | 1.2e-10 | ≈473 |

## Phase 4 — report **landed**

Canonical: `examples/multi_gpu_nacl6/results/RESULTS.md` (+ `SUMMARY.md` / `SUMMARY.json`).  
Product backend line: **Kokkos+LibTorch** / `kokkos_libtorch_vesin` (not Ray / FairChem GP).

## Phase 5 — multi-node

**Out of scope** for this campaign close-out. Same-node GP is scientifically green; multi-node MPI-GP is a later track.
