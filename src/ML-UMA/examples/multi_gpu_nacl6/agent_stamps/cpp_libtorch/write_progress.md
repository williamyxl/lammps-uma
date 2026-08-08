# C++ LibTorch track — WRITE progress

## Burst 1 — 2026-08-07 ~23:00 CDT

### Inventory (`model_state.pt`)
- Pure `state_dict` (154 FP64 tensors) + sidecar `eager_arch.json` (`HydraModel`).
- **Cannot** `torch.jit.load` or reconstruct `torch::nn::Module` in C++ from weights alone.
- Documented in `BLOCKERS.md` (B1). Product path = export-time FairChem → per-rank TorchScript with `uma_peer` ops → C++ runtime.

### Landed
| Area | Files |
|------|-------|
| Peer collectives | `include/uma/kokkos_peer.h` — `peer_copy` / `fence_*` / `enable_peer_access` / `PeerGatherSlot` |
| Peer context + ops | `include/uma/peer_context.h`, `src/peer_context.cpp` — `TORCH_LIBRARY(uma_peer)` |
| C++ MP runtime | `include/uma/libtorch_mp.h`, `src/libtorch_mp.cpp` — vesin full graph → N modules |
| GP default | `graph_parallel.*` — **C++ LibTorch default**; Python only if `UMA_PYTHON_GP_WORKER=1` |
| Export tool | `python/uma_peer_ops.py`, `python/export_mp_artifact.py` |
| Tests | `tests/kokkos_peer_device_smoke.cpp`; CPU `kokkos_peer_smoke` OK |
| Build | `build-cpp-mp/` links clean |

### Jobs
- SLURM `20924647` (`export_mp_w2.slurm`): export `model_mp_w2_r{0,1}.pt` + device smoke + `uma_parity_cli --devices 2`.

### Next
1. Wait export job; fix trace failures if any (stay on C++ path — no Ray).
2. Gate E @ devices=2 vs ASE FP64@1 / devices=1 (`ONLY_PATHS=uma_double`, `RECOMPILE=1`).
3. Then forces; then devices=4.

## Burst 2 — 2026-08-07 ~23:05 CDT

- MP export **succeeded**: `model_mp_w2_r{0,1}.pt` + `model_mp_export.json` ok=true.
- `kokkos_peer_device_smoke OK` on A100×2.
- First `uma_parity_cli --devices 2` failed: full edges into rank modules → `index_add` OOB in `edge_degree_scatter`.
- Fix: `LibtorchMpRuntime` now shards edges via `graph_shard.h` per rank (global centers; offset baked in TS).
- Job: smoke_mp_w2 rebuild + parity.


## Burst 3 — 2026-08-07 ~23:25 CDT

- Confirmed `uma_peer::{all_gather_nodes,all_reduce_sum}` in traced graph.
- Threaded `jit::forward` deadlocks on mid-forward collectives (B4).
- Rewrote `LibtorchMpRuntime` to **process-per-rank** + `SharedPeerGatherSlot` (mmap/PTHREAD_PROCESS_SHARED, host-staged).
- Rebuild clean; resubmitting smoke.


## Burst 4 — 2026-08-07 ~23:45 CDT

- Exec workers live; fail was **device bake** (`cuda:0` vs `cuda:1` in TS).
- Fix: per-rank `CUDA_VISIBLE_DEVICES` at export + worker exec; worker uses `cuda:0`.
- Re-export job `20925011`: both ranks `cuda:0` only in graph (verified). Workers start forward (`index_reduce` on both). Then parent `read_all: EOF` — a worker died mid-predict without protocol error (likely CUDA assert / abort). **devices=2 E not green yet.**

### Landed this track
| Item | Status |
|------|--------|
| `model_state.pt` inventory + B1 | done |
| `kokkos_peer` fence/peer_copy + device smoke | OK |
| `model_mp_w2_r{0,1}.pt` + `uma_peer` ops in graph | OK |
| `GraphParallelRuntime` default = C++ LibTorch | OK |
| Process-per-rank exec + `/dev/shm` collectives | OK |
| CVD bake fix | OK |
| devices=2 E vs ASE | **blocked on worker mid-forward crash (B5)** |

### Next WRITE
1. Per-rank worker stderr logs under stamp dir; `CUDA_LAUNCH_BLOCKING=1`.
2. Fix remaining forward crash; then E gate vs ASE FP64@1 / d1.

## Burst 5 — 2026-08-08 ~00:22 CDT (B5b → E green)

### Abort capture (`20925077`, `CUDA_LAUNCH_BLOCKING=1`, `worker_logs_*`)
- Forward completed on both ranks (`forward done, denorm`).
- Real abort: `torch::autograd::grad` → *element 0 of tensors does not require grad and does not have a grad_fn*.
- Not a CUDA assert / shard OOB.

### Fixes landed
1. `TORCH_LIBRARY_IMPL(uma_peer, Autograd)` — `AllGatherNodesFn` (sum_grad bwd) + `AllReduceSumFn` (identity bwd).
2. No post-forward energy all_reduce (TS already reduces; was 2× E).
3. Process-global rank (autograd engine threads broke `thread_local`).
4. `c10::AutogradState::…set_multithreading_enabled(false)` for ordered mid-bwd collectives.
5. Per-rank stderr under `agent_stamps/cpp_libtorch/worker_logs_<jobid>/`.

### Gate (`20925309`)
| | |
|--|--|
| Structure | `artifacts/nacl64.txt` (64 atoms) |
| devices=1 | E = −216.267998868581 eV |
| devices=2 | E = −216.267998868581 eV |
| **dE_d1** | **0.0** |
| Result | `SMOKE_OK` |

(1728-atom ASE −5830.92 is a different geometry.)

### Next
1. Force parity vs devices=1 / ASE on same structure.
2. devices=4 smoke; optionally full NaCl6 geometry.

## Burst 6 — 2026-08-08 ~00:50 CDT (F green + NaCl6)

### Force regime (sweep `20925383`)
Best: **force all_reduce SUM** + **`all_reduce` bwd** on `uma_peer::all_reduce_sum` + **grad energy scale 1/world**.
Defaults wired in worker / `AllReduceSumFn` (override with env for diag).

### Gates
| Job | Structure | dE_d1 | max\|ΔF\| | Result |
|-----|-----------|-------|----------|--------|
| `20925398` | nacl64 | 0 | 5.3e-16 | E+F green |
| `20925456` | export n1728 | — | — | `model_mp_w2_n1728_r*.pt` |
| `20925457` | NaCl6 1728 | 1.8e-12 | 5.3e-16 | E+F green; dE_ase≈1.2e-10 |

### Note
MP TS bakes `gp_node_offset` → n-specific artifacts (`UMA_MP_NATOMS`). Keep legacy `model_mp_w2_r*.pt` for n=64.

### Next
1. Optional: unbake node offset for size-agnostic MP TS.
2. devices=4.
