# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~01:17 CDT  
**Track:** `cpp_libtorch`

## B1 — `model_state.pt` is weights-only (architecture gap)

Product path: export-time FairChem → per-rank TorchScript with `uma_peer` → C++ process-per-rank runtime. **Not Ray / not Python GP as product.**

## B2 — Serial `model_traced.pt` cannot host MP collectives

Use `model_mp_w{N}_r*` / `model_mp_w{N}_n{NATOMS}_r*`.

## B4 / B5 / B5b / B6 — **resolved**

Process-per-rank + Autograd `uma_peer` + force regime: **all_reduce bwd + force SUM + escale=1/world**.

## B7 — MP TorchScript is **n_atoms-specific** (baked `gp_node_offset`)

**Mitigation:** `model_mp_w{W}_n{N}_r{R}.pt` + `UMA_MP_NATOMS=N`. Legacy `model_mp_w{W}_r*.pt` = n=64.

Optional later: unbake offset for size-agnostic artifacts.

## Gates — devices=2 and devices=4 **GREEN**

### devices=2 (prior)

| Structure | Job | dE_d1 | max\|ΔF\| | dE_ase |
|-----------|-----|-------|----------|--------|
| nacl64 | `20925398` | 0 | 5.3e-16 | — |
| NaCl6 1728 | `20925457` | 1.8e-12 | 5.3e-16 | ≈1.2e-10 |

### devices=4 (this burst)

| Structure | Export | Smoke | dE_d1 | max\|ΔF\| | dE_ase |
|-----------|--------|-------|-------|----------|--------|
| nacl64 | `20925503` → `model_mp_w4_r{0..3}.pt` | `20925504` | **0** | **6.7e-16** | — |
| NaCl6 1728 | `20925505` → `model_mp_w4_n1728_r{0..3}.pt` | `20925506` | **1.8e-12** | **5.8e-16** | **1.2e-10** |

`UMA_MP_NATOMS=1728` for NaCl6. Force defaults unchanged. No unbake needed for w=4.

### Next (optional)

- Unbake `gp_node_offset` for one artifact across sizes.
- Wire LAMMPS `pair_style uma/kk … devices 4` end-to-end smoke if not already covered.
