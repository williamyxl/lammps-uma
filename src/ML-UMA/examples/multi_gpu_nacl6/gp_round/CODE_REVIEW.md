# CODE REVIEW — gp_round (uma/kk graph-parallel)

**Reviewer:** REVIEW agent (Grok 4.5 High)  
**Reviewed at:** 2026-08-06T21:10:00Z  
**WRITE stamp:** `gp_round/.write_agent_done.json` (`utc=2026-08-06T21:08:30Z`)  
**C++ changed:** yes (`rebuild_required=true`, `devices_gt1_backend=fairchem_eager_python`)

## Verdict: **PASS**

No hard blockers before `./gp_round/rebuild_and_submit.sh --submit`. Rebuild is mandatory (handled by that script with `RECOMPILE=1`). Residual Ray/NCCL teardown risk is documented; gp_round `ONLY_PATHS=uma_double,uma_mixed` avoids ASE↔FC path sequencing.

---

## Checklist

| # | Item | Status |
|---|------|--------|
| 1 | NL contract in `uma-engine/docs/multi_gpu_graph_parallel.md` vs `Predictor::rebuild_neighbors` (cutoff, max_neighbors, vesin flip) | **Pass** |
| 2 | `devices=1` traced Predictor; `devices>1` `GraphParallelRuntime` + `uma_gp_worker.py` FairChem `workers=N` (not DevicePool) | **Pass** |
| 3 | `pair_style uma/kk precision … devices N` in `pair_uma.cpp`; Kokkos auto devices; single-MPI guard | **Pass** |
| 4 | `parity_gates.py` thresholds match plan (double/mixed) | **Pass** |
| 5 | Risks called out: Ray/NCCL teardown, workers>1 internal NL vs engine NL, rebuild required | **Pass** (residual, not blocking) |
| 6 | Blockers before `--submit` | **None** |

---

## 1. Neighbor-list contract

Doc (`multi_gpu_graph_parallel.md`) matches `predictor.cpp::rebuild_neighbors()`:

| Parameter | Doc / artifact | Code |
|-----------|----------------|------|
| cutoff | 6.0 Å (`metadata.json`) | `metadata_.cutoff` |
| max_neighbors | 300 | `metadata_.max_neighbors` |
| CUDA NL | vesin `full_directed=true` | `vesin_build_graph_cuda(..., true, ...)` |
| Edge flip | vesin (center, neighbor) → FairChem (neighbor, center) | `stack({neighbor_j, center_i}, 0)` |
| CPU fallback | FairChem orientation from `build_neighbor_graph` | present |
| devices>1 | FairChem internal graph (`external_graph_gen=False`); gate on E/F not bit-identical edges | `uma_gp_worker.py` sets `external_graph_gen = workers <= 1` |

Artifacts `uma-s-1p2-omat{-f64}`: cutoff=6.0, max_neighbors=300, `checkpoint_path` → existing `uma-cache/uma-s-1p2.pt`.

---

## 2. devices=1 vs devices>1 backend

- `Predictor::from_artifact(..., num_devices)`: `N==1` loads `model_traced.pt`; `N>1` constructs `GraphParallelRuntime` only (throws if `create` called with `N<=1`).
- `GraphParallelRuntime` forks persistent `uma_gp_worker.py` and inits `load_predict_unit(..., workers=N)` → real FairChem `ParallelMLIPPredictUnit` (Ray + NCCL). Header explicitly forbids a serial DevicePool.
- No `DevicePool` implementation in tree (only a “not a DevicePool” comment).

---

## 3. pair_style parsing / Kokkos / MPI

`pair_uma.cpp::settings`:

- Parses `precision mixed|double` and `devices N` (`N>=1`), sets `devices_explicit`.
- Bare `mixed`/`double` aliases retained.

`load_predictor()`:

- If `devices` omitted and Kokkos `ngpus>1`, auto-sets `devices=ngpus` and logs; `devices 1` forces traced.
- Loads via `Predictor::from_artifact(..., num_devices)`; logs `gp=fairchem_eager_python|traced`.

`compute()`: `comm->nprocs > 1` → hard error (“single MPI rank”). Matches campaign contract.

---

## 4. Parity thresholds

`parity_gates.py` matches doc / DRY_RUN_CHECKLIST / README:

| Mode | \|ΔE\| max | max \|ΔF\| | cosine min |
|------|------------|------------|------------|
| double | 1e-8 | 1e-6 | 1 − 1e-12 |
| mixed | 1e-4 | 1e-5 | 1 − 1e-10 |

Primary gate is uma `devices=N` vs uma `devices=1` at the same precision.

---

## 5. Risks (accepted residuals)

1. **Ray/NCCL teardown:** `uma_gp_worker.py` `shutdown` exits without explicit `ray.shutdown()`. C++ `shutdown_worker` soft-waits ~5s then `SIGTERM`. Known from prior ASE/FC hangs; gp_round is uma-only per SLURM job, so allocation end reclaims GPUs. Residual hang risk on Pair destructor if Ray children stall — mitigated by SIGTERM.
2. **NL parity:** workers>1 uses FairChem internal graph, not engine vesin NL. Correctness is E/F gate, not edge identity — documented in NL contract and worker header.
3. **Rebuild required:** C++/engine/pair changes landed; `build-uma/lmp` must be rebuilt before jobs. `rebuild_and_submit.sh --submit` sets `RECOMPILE=1` and rebuilds.
4. **Nit (non-blocking):** `GraphParallelRuntime::predict` ignores `charge`/`spin` args (hardcodes 0 in JSON). Fine for NaCl6 omat; fix if charged systems are needed later.

---

## 6. Pre-submit blockers

**None.** Proceed with:

```bash
cd /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6
./gp_round/rebuild_and_submit.sh --submit
```

Expect: rebuild → sbatch ngpu1 → ngpu2 (`afterok`). Hold ngpu4 until ngpu2 gates green.

---

## Files reviewed

- `uma-engine/docs/multi_gpu_graph_parallel.md`
- `uma-engine/src/{predictor,graph_parallel,metadata}.cpp` + headers
- `uma-engine/python/uma_gp_worker.py`
- `uma-engine/tests/parity_cli.cpp`, `CMakeLists.txt`
- `pair_uma.{h,cpp}`
- `parity_gates.py`, `gp_round/{rebuild_and_submit.sh,_run_gp_common.sh,DRY_RUN_CHECKLIST.md}`
