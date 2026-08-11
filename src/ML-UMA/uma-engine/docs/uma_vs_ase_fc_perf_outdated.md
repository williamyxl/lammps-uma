# uma/kk vs ASE FairChem / FC LAMMPS — multi-GPU design note

**Stamp:** 2026-08-08 ~19:00 CDT · Campaign **PASS** (P3c job `20940474`)  
**Geometry:** NaCl6 1728 atoms, FP64 only · product `pair_style uma/kk … devices N` with `-k on g N`, 1 MPI rank.

## Architecture contrast

| Layer | ASE / FairChem FC | uma/kk (product) |
|-------|-------------------|------------------|
| Parallelism | Ray `ParallelMLIPPredictUnit` + `torch.distributed` NCCL | Fork workers under one LAMMPS rank |
| Neighbor list | Inside FairChem / Ray path | Parent vesin full NL, then shard fan-out |
| Peer transport | NCCL in Ray workers | **NCCL product default** (`select_transport`); override `UMA_PEER_TRANSPORT=cuda_ipc\|shm` |
| Kokkos | N/A | Kept (`-sf kk`); NCCL replaces peer IPC only |

NCCL does **not** drop Kokkos — it only swaps the collective transport under `uma_peer`.

**LIM:** `PayloadShm` fixed mmap caps `kMaxN=8192`, `kMaxWorld=8` (single-node). Multi-node out of scope.

## Attribution path (why @4 lagged ASE)

1. **P3a (cuda_ipc)** `20934280`: **320.6 / 183.57 / 117.63** — beat FC @4; still **+2.4 ms vs ASE 115.2**.
2. **P3b** warm @4: `ms_wait ≈ ms_compute ~106`; `ms_nl+pub ~10`; force gather ≪1 ms. Residual vs ASE lived in worker fwd/bwd collectives, not parent NL alone.
3. **P3c (NCCL)** `20940474`: **321.04 / 183.30 / 112.04** — clears hard bar (**−3.2 vs ASE**, **−6.0 vs FC**) with E+F green and self-scale.

## Bugs hit on the way

- P3c sanity false FAIL: `pipefail` + `grep -q` SIGPIPE → use `nm -D` / `ldd` counts; link libnccl on worker explicitly.
- Hang `20940376`: **NCCL teardown deadlock** (not mid-step). Per-rank `write(cmd=0)+waitpid` left rank0 in `ncclCommDestroy` while rank1 never got shutdown. Fix: broadcast shutdown, then waitpid; rendezvous before `ncclCommDestroy`.

## Product stance

- Hard campaign success = honest pair ms **≤ ASE and ≤ FC** at every GPU count **and** E+F accuracy. Soft ≤150 alone is not enough.
- Keep Ray out of product. NCCL is the product default for peer collectives; cuda_ipc/shm remain fallbacks.
- Water888 @4: spike-free ~**125** vs ASE **118** (~+7). Residual is **worker wait (~110)** + serial parent NL+pub (~15). Next V5: defer NCCL `cudaDeviceSynchronize`, right-size `kMaxE`, stream overlap — `examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/STATUS.md`.
- **LIM:** `PayloadShm` `kMaxN=8192` / `kMaxWorld=8`; world-sized maps; `kMaxE=512k` oversized for water (V5a).

## Canonical numbers

See [`examples/multi_gpu_nacl6/results/RESULTS.md`](../../examples/multi_gpu_nacl6/results/RESULTS.md) and stamp `agent_stamps/cpp_libtorch/perf/summary_p3c_20940474.json`.
