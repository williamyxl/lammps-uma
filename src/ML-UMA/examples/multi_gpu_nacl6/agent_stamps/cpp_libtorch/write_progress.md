# C++ LibTorch track — WRITE progress

REPORT_OWNER=parent (no RESULTS/SUMMARY/MULTIGPU/canvas edits)

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

## Burst 7 — 2026-08-08 ~01:17 CDT (devices=4 E+F green)

### Export
| Job | Artifacts |
|-----|-----------|
| `20925503` | `model_mp_w4_r{0..3}.pt` (n=64) |
| `20925505` | `model_mp_w4_n1728_r{0..3}.pt` |

### Gates (`smoke_mp_w4.slurm`, RECOMPILE=1, force defaults)
| Job | Structure | devices | dE_d1 | max\|ΔF\| | dE_ase |
|-----|-----------|---------|-------|----------|--------|
| `20925504` | nacl64 | 4 | 0 | 6.7e-16 | — |
| `20925506` | NaCl6 1728 | 4 | 1.8e-12 | 5.8e-16 | 1.2e-10 |

Same force regime as w=2; escale=1/world (=0.25) works. No offset unbake required.

### Next (optional)
1. Unbake `gp_node_offset`.
2. LAMMPS `pair_style uma/kk devices 4` integration smoke.

## Burst 8 — 2026-08-08 ~01:20 CDT (Phase 3 LAMMPS wire)

Phase 2b engine/CLI done → Phase 3 end-to-end LAMMPS.

### Wiring fixes
| Change | Why |
|--------|-----|
| `build_lammps_uma.sh` also builds `uma_libtorch_mp_worker` | EXCLUDE_FROM_ALL under LAMMPS cmake |
| `run_multigpu.setup_ld_path` + `UMA_MP_NATOMS` from N atoms | worker path + n1728 shards |
| `pair_uma` GP log → `kokkos_libtorch_vesin` | was misleading `fairchem_eager_python` |
| `lammps_smoke.slurm` | RECOMPILE=1, forbid Ray, force-green env, gate vs `results/ngpu1` |

### Jobs
- devices=2 NaCl6 LAMMPS smoke submitted (see `active_jobid.txt`).
- devices=4 after w2 green.

## Burst 9 — 2026-08-08 ~01:49 CDT (Phase 3 LAMMPS GREEN)

### Rebuild
`build-uma/lmp` now links `libtorch_mp.o` + `peer_context.o` (`LibtorchMpRuntime` symbols present). Worker: `build-cpp-mp/uma_libtorch_mp_worker`.

### LAMMPS gates (NaCl6 1728, `uma_double`, FP64)
| Job | devices | dE_d1 | max\|ΔF\| | dE_ase | pair ms | wall_s |
|-----|---------|-------|----------|--------|---------|--------|
| `20925747` | 2 | 9.1e-13 | 0 | 1.2e-10 | ≈361 | 49.4 |
| `20925801` | 4 | 2.7e-12 | 0 | 1.2e-10 | ≈473 | 64.4 |

Log: `gp=kokkos_libtorch_vesin`. No Ray / no Python GP.

### Timing note
Report pair-path `ms_per_eval` from `run_multigpu` for honesty. SLURM `wall/N_TIMING` inflates (setup + SP dump + NVE).

## Burst 10 — 2026-08-08 ~01:50 CDT (Phase 4 report)

Same-node GP scientifically GREEN through Phase 3. **No new GPU jobs.**

### Landed
| Doc | Role |
|-----|------|
| `results/RESULTS.md` | Canonical Phase 4 — product = Kokkos+LibTorch / `kokkos_libtorch_vesin` |
| `results/SUMMARY.md` + `SUMMARY.json` | Compact + machine-readable |
| Stamps | Phase 3 DONE · Phase 4 report landed · Phase 5 multi-node out of scope |

### Not in this burst
Multi-node · Ray · offset unbake.

## Burst 10 — perf P0 (2026-08-08 ~08:32 CDT)

Goal: improve devices=2/4 pair ms while keeping E+F green vs devices=1.
Hard gate: pair_ms(2)<pair_ms(1) and pair_ms(4)<pair_ms(2).

P0: stop forcing CUDA_LAUNCH_BLOCKING=1 on workers (opt-in UMA_CUDA_LAUNCH_BLOCKING=1).
Worker PERF_TICK timers (ms_fwd/ms_bwd/ms_force_ar). Job 20932843 = perf_p0.slurm devices=1→2→4.

## Burst 10 — P0 result (job 20932843)

Honest pair ms: **320.64 / 330.52 / 382.86** @1/2/4. E+F green (max|ΔF|=0). Self-scale FAIL.
P0 (no CUDA_LAUNCH_BLOCKING) helped vs Phase-3 (~361/~473) but still slower than @1.
Next: P1 CUDA IPC device collectives.

## Burst 11 — perf P1 CUDA IPC (2026-08-08 ~08:50 CDT)

### Landed
| Item | Detail |
|------|--------|
| `shared_peer.h` | `UMA_PEER_TRANSPORT=shm\|cuda_ipc`; device IPC buffers; shm = control + handles + nbytes |
| Worker | `init_cuda_ipc(rank)` after `cudaSetDevice(0)` |
| `libtorch_mp.cpp` | shm size via `map_bytes_for(world, transport)` |
| `perf_p0.slurm` | gate uses honest `uma64`/pair_section ms (not wall/N) |
| `perf_p1.slurm` | devices=1→2→4, `RECOMPILE=1`, `UMA_PEER_TRANSPORT=cuda_ipc` |

Force-green regime unchanged (all_reduce bwd + force SUM + escale 1/world). No Ray.

## Burst 12 — perf P2 pipe-tax (2026-08-08 ~09:15 CDT)

P1 PASS (20932975): 320.34 / 264.96 / 193.32. Soft gap vs ASE/FC @2/@4 remains.

### P2 landed (code)
| Item | Detail |
|------|--------|
| `payload_shm.h` | Shared mmap for pos/z/per-rank edges + rank0 forces |
| `libtorch_mp.cpp` | Publish once; pipe wake = cmd+gen; `PERF_PARENT` timers; cache partition check |
| Worker | Read payload shm; rank0 writes E+F to shm; pipe returns ok only; quiet unless `UMA_MP_VERBOSE=1` |
| `perf_p2.slurm` | devices 1→2→4, RECOMPILE=1, beat-P1 + self-scale + E+F gates |

Goals: keep E+F + self-scale; beat P1 @2/@4; soft-push toward ASE ~194/@2 and ~115–150/@4.

## Burst 13 — P2 result (job 20933393)

Honest pair ms: **321.54 / 190.80 / 140.90**. E+F green (max|ΔF|=0). Self-scale PASS. Beat P1 PASS.
- Soft ASE@2: **OK** (beats ASE/FC). Soft ASE@4 ≤150: **OK**; gap to ASE ~115 remains ~26 ms.
- RESULTS/SUMMARY updated to P2 three-path row.

### P1 result (job `20932975`) — SELF_SCALE_GREEN

Honest pair ms: **320.34 / 264.96 / 193.32** @1/2/4. E+F green (max|ΔF|=0).
vs P0 320.6/330.5/382.9 and Phase-3 ~361/~473. Soft gap vs ASE/FC @4 (~115) remains (optional P2).

## Tick policy (2026-08-08 ~09:59)

On every parent `AGENT_LOOP_TICK_mp_perf`: refresh `results/{RESULTS,SUMMARY,MULTIGPU_REPORT}.md` + canvases `nacl6-multigpu-results` / `nacl6-three-path-compare` from latest honest ms / gates (do not leave stale P1 numbers).

REPORT_OWNER=parent (no RESULTS/SUMMARY/MULTIGPU/canvas edits)

## Burst 14 — P3 close @4 vs ASE/FC (2026-08-08 ~10:05 CDT)

REPORT_OWNER=parent (no RESULTS/SUMMARY/MULTIGPU/canvas edits)

Constraint: stamps + `uma-engine/` only. Target: cut ~26 ms @4 (140.9 → ≲120) without weakening E+F/self-scale.

### P3 code (P3a)
| Item | Detail |
|------|--------|
| `graph_shard::pack_shards_cpu` | One-pass CPU pack; drop `torch::isin` publish tax |
| Worker | Drop mid-path `cudaDeviceSynchronize`; one sync before D2H |
| Parent | `PERF_PARENT` also appended to `UMA_MP_LOG_DIR/parent.log` |
| `perf_p3.slurm` | beat P2 @4 + self-scale + E+F; soft @4 ≤120 |

## Burst 15 — P3 phased plan ack (2026-08-08 ~10:17 CDT)

REPORT_OWNER=parent (no RESULTS/SUMMARY/MULTIGPU/canvas edits). Plan: `uma_vs_ase_fc_perf`.

| Phase | Scope | Status |
|-------|--------|--------|
| **P3a** | `pack_shards_cpu` + sync cuts + `PERF_PARENT`; default `UMA_PEER_TRANSPORT=cuda_ipc`; job `20934280` (`perf_p3.slurm`, RECOMPILE=1) | **in queue** (Priority; scontrol StartTime ~22:02 CDT) |
| **P3b** | After P3a lands: attribute @4 residual from `PERF_PARENT` (`ms_nl`/`ms_pub`/`ms_wait_workers`) vs worker `PERF_TICK`; stamp under `cpp_libtorch/` only | blocked on P3a |
| **P3c** | Opt-in `UMA_PEER_TRANSPORT=nccl` (raw libnccl behind `uma_peer` gather/reduce); cuda_ipc fallback; keep `uma/kk` + `-k on g N`; A/B vs P3a @1/2/4 | **do not code until P3a+P3b stamped** |
| **P4a** | Harden/default nccl **only if P3c wins** | later |
| **P4b** | Else parent NL/publish cuts | later |

Hard gates unchanged: E+F green · self-scale · beat prior @4. Soft: @4 toward ASE ~115. No Ray; no RESULTS/SUMMARY/MULTIGPU/canvas edits.

## Burst 16 — HARD campaign success criteria (2026-08-08 ~10:25 CDT)

REPORT_OWNER=parent (no RESULTS/SUMMARY/MULTIGPU/canvas edits).

**Campaign FAIL** if product `uma/kk` honest pair ms is **slower than ASE FairChem FP64 or FairChem FC LAMMPS** at the same GPU count (NaCl6 1728, FP64), even if self-scale and E+F are green. Soft ≤150 alone does **not** close the campaign.

| Required | Bar |
|----------|-----|
| Accuracy | E+F green vs devices=1 and vs ASE oracle bands (no regress) |
| Perf @2 and @4 | honest `uma64`/pair ms **≤ ASE and ≤ FC** |
| Soft stretch | close to / under ASE @4 (~115) |

P2 status: PASS @1/@2 · **FAIL @4** (uma~140.9 vs ASE~115 / FC~118) → campaign **OPEN**. Continue P3a→P3b→P3c against this bar.
