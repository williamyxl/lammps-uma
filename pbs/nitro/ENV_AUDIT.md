# UMA name audit (stage H1)

`uma::runtime_get` returns the in-process map. If the name is absent there, it falls back to `getenv` only when the name is on the allowlist in `runtime_config.cpp`. The fallback value is copied into the map, so later `runtime_set` still wins and the returned pointer stays valid. An empty value is treated as unset.

Rank 0 prints one line on the first `runtime_get`: `NITRO_FLAGS: (none)` or `NITRO_FLAGS: UMA_NITRO_<ID>=...`. Rank comes from `PMI_RANK`, `PMIX_RANK`, `PALS_RANKID`, `SLURM_PROCID`, or `OMPI_COMM_WORLD_RANK`.

## Allowlist (getenv fallback)

| Name | Class | Harness | Reader |
|---|---|---|---|
| `UMA_MP_PERF` | perf | `lib.inc` exports `1` | `libtorch_mp.cpp`, `mpi_peer_predictor.cpp`, `predictor.cpp`, `shared_peer.h` |
| `UMA_MEM_PEAK` | perf | `lib.inc` exports `1` | `pair_uma.cpp` (`runtime_flag_1`) |
| `UMA_MP_NATOMS` | functional | `run_gate.pbs`, `run_timing.pbs`, `lib.inc` | `libtorch_mp.cpp`, `mpi_peer_predictor.cpp` |
| `UMA_MN_CKPT` | functional | `lib.inc` exports `0` (checkpoint off) | `mpi_peer_predictor.cpp` (`mn_checkpoint_enabled`) |
| `UMA_ALLOW_LEGACY_METADATA` | functional | `env.inc` and `lib.inc` export `0` | `metadata.cpp` (`env_flag`, true only for `1`) |
| `UMA_NITRO_*` | arm toggle | unset means BASE | any future reader; logged on rank 0 |

## Keyword names (no getenv fallback)

`PairUMA::apply_runtime_config` writes these from `pair_style` keywords. A process environment value is ignored.

| Name | Keyword |
|---|---|
| `UMA_AC` | `ac off\|chunk\|block\|full` |
| `UMA_DD` | domain decomposition active |
| `UMA_ENGINE_BUILD_GRAPH` | engine rebuilds the graph |
| `UMA_COMPUTE_VIRIAL` | `virial auto\|yes\|no` (`auto` clears the key) |

`lib.inc` also exports `UMA_AC=off`. That export does not reach the engine. The deck keyword does.

## Read by the engine, not allowlisted (env ignored)

These stay map-only. The nitro harness does not export them. Setting them in the environment has no effect.

`UMA_PEER_PERF`, `UMA_DD_DEBUG`, `UMA_DD_HALO_TEST`, `UMA_DD_EDGE_CAP`, `UMA_DD_NO_HALO`, `UMA_CUDA_GRAPH`, `UMA_SKIP_MAXNBR_CAP`, `UMA_CHUNK_RETAIN_K`, `UMA_ALLREDUCE_WITH_GRAD_BWD`, `UMA_SKIP_PRE_BWD_BARRIER`, `UMA_DEBUG_PARTITION_CHECK`, `UMA_CUDA_LAUNCH_BLOCKING`, `UMA_NL_ALLPAIRS`, `UMA_PEER_SPLIT`, `UMA_NO_RECOMPUTE`, `UMA_NO_RECOMPUTE_BLOCK`, `UMA_NO_RECOMPUTE_CHUNK`, `UMA_NO_RECOMPUTE_EDEG`, `UMA_CKPT` (harness unsets it), `UMA_STRUCTURE_NATOMS`, `UMA_LIBTORCH_MP_WORKER`, `UMA_MP_LOG_DIR`, `UMA_PYTHON_GP_WORKER`, `UMA_EAGER_CKPT`, `UMA_PEER_TRANSPORT`.

## Not read by `runtime_get`

Scripts or Python only: `UMA_CHECKPOINT`, `UMA_TASK`, `UMA_ALLOW_FAIRCHEM_MISMATCH` (`trace_patch.py` uses `os.environ`), `UMA_EXPORT_CELL_LIST`, `UMA_HEN_ROOT`.

## Arm toggles

A new behaviour is `UMA_NITRO_<ID>` (non-empty id). Unset means the BASE path. Read it once at construction. Rank 0 logs every `UMA_NITRO_*` variable that is set, so the log shows which arm ran.
