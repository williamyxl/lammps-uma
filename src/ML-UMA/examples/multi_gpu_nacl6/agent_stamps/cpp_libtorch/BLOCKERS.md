# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~00:50 CDT  
**Track:** `cpp_libtorch`

## B1 — `model_state.pt` is weights-only (architecture gap)

Documented previously. Product path: export-time FairChem → per-rank TorchScript with `uma_peer` → C++ process-per-rank runtime. **Not Ray / not Python GP as product.**

## B2 — Serial `model_traced.pt` cannot host MP collectives

Still true. Use `model_mp_w{N}_r*` / `model_mp_w{N}_n{NATOMS}_r*`.

## B4 / B5 / B5b — deadlock, CVD bake, missing Autograd — **resolved**

See prior stamps. Process-per-rank + Autograd `uma_peer` + process-global rank + single-thread autograd.

## B6 — Force mismatch vs devices=1 — **resolved**

| Sweep `20925383` best | `sumF` + `all_reduce` bwd + `escale=1/world` |
|----------------------|-----------------------------------------------|
| Root cause | Concurrent per-rank `grad(E_full)` needs (1) all_reduce grads on mid-graph `all_reduce_sum`, (2) force all_reduce across edge shards, (3) loss scale `1/world` |
| Defaults | `UMA_ALLREDUCE_WITH_GRAD_BWD` default ON; worker `escale=1/world`; force `SUM` (skip via `UMA_SKIP_FORCE_GP_REDUCE=1`) |

### nacl64 E+F (job `20925398`)

| | devices=1 | devices=2 |
|--|-----------|-----------|
| E (eV) | −216.267998868581 | −216.267998868581 |
| max\|ΔF\| | — | **5.3e-16** |

### NaCl6 1728-atom E+F (job `20925457`)

| | devices=1 | devices=2 | ASE FP64@1 |
|--|-----------|-----------|------------|
| E (eV) | −5830.923720166719 | −5830.923720166721 | −5830.9237201666 |
| dE_d1 | — | **1.8e-12** | dE_ase≈1.2e-10 |
| max\|ΔF\| | — | **5.3e-16** | — |

Requires n-specific export `model_mp_w2_n1728_r{0,1}.pt` (`UMA_MP_NATOMS=1728`).

## B7 — MP TorchScript is **n_atoms-specific** (baked `gp_node_offset`)

FairChem traces `gp_node_offset = node_partition.min()` as a constant. A 64-atom export OOBs on 1728 (`index_add` CUDA assert).

**Mitigation:** export `model_mp_w{W}_n{N}_r{R}.pt` via `--atoms`; runtime `UMA_MP_NATOMS=N`. Legacy `model_mp_w2_r*.pt` = n=64.

**Next (optional):** unbake offset (tensorized from `natoms`/rank) for one artifact across sizes; devices=4.
