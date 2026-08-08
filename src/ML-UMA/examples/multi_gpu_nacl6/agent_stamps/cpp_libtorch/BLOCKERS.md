# C++ LibTorch track — blockers

**Stamp:** 2026-08-08 ~00:22 CDT  
**Track:** `cpp_libtorch`

## B1 — `model_state.pt` is weights-only (architecture gap)

| Fact | Detail |
|------|--------|
| File | `artifacts/uma-s-1p2-omat-f64/model_state.pt` (~2.2 G) |
| Contents | `{"state_dict": …, "dtype": "float64", "task": "omat", "checkpoint_path": …}` |
| Keys | 154 pure `torch.float64` tensors (`backbone.*`, heads, …) |
| Module type (sidecar) | `fairchem.core.models.base.HydraModel` (`eager_arch.json`) |
| `torch.jit.load(model_state.pt)` | **Fails** (not a TorchScript archive) |
| C++ `torch::nn::Module` from state_dict alone | **Impossible** — UMA eSCN/Hydra graph, SO2/SO3 ops, and GP control flow live only in FairChem Python |

**Implication:** LibTorch cannot reconstruct a runnable eager module from `model_state.pt` without either (a) a scripted/traced artifact that embeds the forward, or (b) generated C++ module code mirroring FairChem (out of scope / multi-month).

**Not a pivot to Ray/Python GP.** Product path (landed):

1. **Export-time (Python FairChem as build tool only):** load checkpoint → `uma_peer` custom ops → `torch.jit.trace` per rank → `model_mp_w{N}_r{R}.pt`.
2. **Runtime (C++ only):** register `uma_peer` Autograd ops → process-per-rank exec + `SharedPeerGatherSlot` → vesin → `graph_shard` → N× LibTorch modules.
3. Default `devices>1` = this C++ path. Python workers only if `UMA_PYTHON_GP_WORKER=1`.

## B2 — Serial `model_traced.pt` cannot host MP collectives

Opaque TorchScript energy wrapper has no gather/reduce insertion points. Do **not** claim multi-GPU by N× independent serial forwards.

## B4 — TorchScript multi-thread deadlock — **resolved**

Cause: mid-forward peer collectives need concurrent rank forwards; `jit::Module::forward` is not multi-thread safe.  
**Fix:** process-per-rank `fork+exec` + `/dev/shm` `SharedPeerGatherSlot`.

## B5 — Rank-1 cuda:0 vs cuda:1 device bake — **resolved**

**Fix:** `CUDA_VISIBLE_DEVICES=<rank>` at export and worker exec; worker uses `cuda:0` only.

## B5b — Worker abort mid-predict — **resolved** (energy green)

| Job | Result |
|-----|--------|
| `20925077` | Captured abort: forward OK; `autograd::grad` → *element 0 … does not require grad* |
| Root cause | `uma_peer` registered only under `CompositeExplicitAutograd` → no `grad_fn` through collectives |
| Fix | `TORCH_LIBRARY_IMPL(uma_peer, Autograd, …)` with `AllGatherNodesFn` / `AllReduceSumFn` |
| Follow-ons | Gather bwd arity (return grad for `n_atoms`); no double energy all_reduce; **process-global rank** (not `thread_local` — autograd engine threads); `AutogradState::set_multithreading_enabled(false)` so mid-bwd collectives stay ordered |

### Energy gate (job `20925309`, `nacl64.txt`)

| Path | E (eV) |
|------|--------|
| devices=1 (`uma_parity_cli`) | −216.267998868581 |
| devices=2 C++ LibTorch MP | −216.267998868581 |
| **dE_d1** | **0.0** |

Note: ASE FP64@1 oracle −5830.92 eV is the **1728-atom** NaCl6 geometry, not `nacl64.txt` (64 atoms). Same-structure gate is devices=1 / this artifact.

`SMOKE_OK` on `gpuA100x4` / `bbpl-delta-gpu`. Worker logs: `worker_logs_20925309/`.

### Remaining (not blocking E)

- **Forces:** devices=2 `fmax≈0.286` vs devices=1 `fmax≈0.327` — force parity not yet gated; next: compare `force_max_d1` vs ASE/d1 and tune gather-bwd / force-reduce regime.
- **Scale-up:** devices=4; full NaCl6 (1728) geometry when ready.
