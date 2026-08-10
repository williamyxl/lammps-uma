# UMA multi-GPU FP64 perf campaign — agent handoff

**Written:** 2026-08-10 ~13:17 CDT  
**Repo:** `lammps-uma` · **Branch:** `uma-kokkos-mlip` · **HEAD at handoff:** `51a6d8c75f`  
**Workspace root:** `/work/nvme/bfzx/xyan11/workdir/lammps-uma`  
**Living campaign dir:** `src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/`  
**Active plan:** `/u/xyan11/.cursor/plans/v6_w8nk_perf_push_77967fcb.plan.md`  
**Prior plan (historical Tier0–Tier2):** `/u/xyan11/.cursor/plans/v5_max_perf_push_82db7365.plan.md`  
**Chat transcript (searchable):** agent transcript `a1f1708b-33c2-450b-8f4b-60f2b06b99ce`

This document is the single entry point for another LLM agent to continue the campaign without re-deriving context.

---

## 1. North star (do not redefine)

Ship **same-node multi-GPU FP64 UMA** as fast as possible with E/F parity:

```text
pair_style uma precision double devices N
+ LibTorch process-per-rank MP workers
+ stream-ordered NCCL (W8-fix)
+ UMA_USE_KOKKOS=0
```

- ASE / FC matching bars are the **minimum gate**, not the finish line.
- Stop only on a **measured hard ceiling**: two consecutive backlog items each move median **&lt;1 ms** with a flat wait profile.
- **FP64 only** — never silently use FP32 for production/benchmarks/GCMC.
- **1 MPI rank**; no Ray; full parent neighbor list; **do not** skip force reduce.

---

## 2. Product floor (locked) — **W8nk**

Until a wave **promotes**, product stays **W8nk**.

| Knob | Value |
|------|--------|
| Recipe | W8-fix NCCL + `UMA_USE_KOKKOS=0` + `pair_style uma` |
| Artifact | `uma-engine/artifacts/uma-s-1p2-omat-f64-fast` (`umas_fast_pytorch` + `merge_mole`) |
| Oracle | ASE `merge_mole=True` |
| Speed metric | **NVT Pair ms/step** (NaCl `NSTEPS=10`, water `NSTEPS=100`) |

**W8nk NVT Pair bars** (`gate_v5_w8nk_summary.json`):

| Cell | NVT Pair ms | Job |
|------|------------:|-----|
| NaCl@2 | **161.94** | `21010252` |
| NaCl@4 | **92.097** | `21010253` |
| water@2 | **164.82** | `21010254` |
| water@4 | **95.744** | `21010255` |

**Open product gap:** water@4 is still ~**1.2 ms** over ASE ufast **~94.5**. W13 showed wait ≈ model fwd+bwd (~98% of step); force AR ≪1 ms.

---

## 3. Hard rules (ops)

1. **Always rebuild** on first job of a new code version: `RECOMPILE=1`. Dependents: `RECOMPILE=0`. Never dual-submit two `RECOMPILE=1` into the same `build-uma`.
2. Campaign deliverables under gitignored paths: `git add -f` → commit → `git push origin HEAD` on `uma-kokkos-mlip`.
3. Do **not** leave stamps only on disk.
4. Gate E/F vs ASE merge oracle before promoting speed.
5. Do **not** retry closed waves without new evidence: W10 edge-pad promote, W12 skip-barrier, Triton `umas_fast_gpu`, FP32, blind W11 without RNG/shape fixes.

**Env / cluster:** Delta `gpuA100x4`, account `bbpl-delta-gpu`, conda `uma312`, CUDA 12.8.

---

## 4. Key paths

| What | Path |
|------|------|
| Campaign stamps | `.../perf_campaign/{STATUS.md,STATE.json,tick.log,MATRIX.md,GLOSSARY.md}` |
| Gate JSONs | `.../perf_campaign/gate_v5_*.json`, `gate_v6_*.json` |
| ASE merge oracles | `oracle_ase_merge_mole.json`, `oracle_ase_umas_fast_merge.npz` (NaCl); water twins `oracle_ase_water_*` |
| Product art | `uma-engine/artifacts/uma-s-1p2-omat-f64-fast/` |
| CUDA-graph art | `uma-engine/artifacts/uma-s-1p2-omat-f64-fast-cgraph/` |
| MP worker | `uma-engine/tests/uma_libtorch_mp_worker.cpp` |
| NCCL / peer | `uma-engine/include/uma/shared_peer.h` |
| Parent MP | `uma-engine/src/libtorch_mp.cpp` |
| Trace patches (γ=0, MOLE) | `uma-engine/python/trace_patch.py` |
| Export wrapper (shapes) | `uma-engine/python/export_wrapper.py` |
| MP export | `uma-engine/python/export_mp_artifact.py` |
| devices=1 export | `uma-engine/python/w15_export_traced_fast.py` |
| NaCl runner | `examples/multi_gpu_nacl6/run_path_uma.slurm` (+ `.py`) |
| Water runner | `examples/water888/run_path_uma.slurm` |
| W17 export | `perf_campaign/w17_export_cgraph.slurm` |
| W17 matrix helper | `perf_campaign/w17_submit_matrix.sh <afterok_jobid>` |

**Checkpoint:** `/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt`

---

## 5. How to gate a finished cell

1. Find log: `examples/multi_gpu_nacl6/logs/path_uma-<JOB>.out` (water under `examples/water888/logs/`).
2. Results: `results/uma_ngpuN_<JOB>/timing.json` + `forces.npz`.
3. Worker path tags: `UMA_MP_LOG_DIR=.../mp_logs` → `worker_r*.log` look for `path=graph_warmup|graph_capture|graph_replay|graph_fail_eager|eager`.
4. E/F vs merge ASE:
   - Energy: `|E - oracle|` ≲ 1e-6 (typically ~1e-10).
   - Forces: `max|ΔF|` ≲ 1e-5 (W8nk ~5e-7), compare to `oracle_ase_umas_fast_merge.npz` (NaCl) / water twin.
5. Speed: `nvt_pair_ms_per_step` from `timing.json`. Promote only if E/F PASS **and** (beats ASE bar **or** ≥1 ms median win vs W8nk without E/F loss).
6. Stamp `gate_v6_*.json`, update `STATUS.md` / `STATE.json` / `tick.log`, `git add -f`, commit, push.

---

## 6. V6 wave ledger (done → in flight)

| Wave | Outcome | Notes |
|------|---------|-------|
| **W8nk** | **PRODUCT** | NVT bars above; E/F all PASS; water@4 still fails ASE ufast floor |
| **W13** | Profile only | water@4: wait dominates; fwd+bwd ~63% of wait; force_ar negligible → **W16 cancelled** |
| **W14** | **NO_PROMOTE** | Async pin D2H + stream sync; E/F PASS; water@4 gain &lt;1 ms |
| **W15** | Partial | devices=1 NaCl re-export PASS NVT **296.5**; water@1 E/F PASS but NVT floor fail |
| **W16** | Skipped | Not on critical path |
| **W11** (hist.) | STRUCTURAL FAIL | `aten::rand` in Wigner during CUDA graph capture |
| **W17** smoke `21013149`/`21013150` | Export OK; E/F PASS; capture FAIL | γ=0 cleared RNG; fail = **default stream** |
| **W17b** `21014028` | CAPTURE FAIL + job die | Non-default stream OK; fail = `_shape_as_tensor`→`.to(cuda)`; sticky capture poisoned eager |
| **W17c** `21015028`→`21015029` | **IN FLIGHT** (Priority/Dependency at handoff) | Capture-safe shapes + EndCapture hygiene |

### W17 technical stack (what landed in code)

1. **γ=0 at export** (`trace_patch.py`): replace Wigner `torch.rand` with zeros → no `aten::rand` in TS (FairChem CUDA-graph comment pattern).
2. **Non-default stream** (`uma_libtorch_mp_worker.cpp`): `getStreamFromPool` + `CUDAStreamGuard` around capture/replay.
3. **NCCL on current stream when `UMA_CUDA_GRAPH=1`** (`shared_peer.h`): avoid side-stream event waits outside the graph.
4. **Capture-safe shapes** (`export_wrapper.py`): `empty`+`fill_(size)` instead of `_shape_as_tensor` + H2D `.to(cuda)`.
5. **Capture-abort hygiene** (worker): `cudaStreamEndCapture` + `cudaGetLastError` clear if stream still capturing after fail.

Graph art dir: `*-f64-fast-cgraph` (separate from product `*-f64-fast`).

---

## 7. Immediate next actions (start here)

### A. Drain W17c

```bash
squeue -u $USER -o '%.10i %.14j %.8T %.10M %R'
sacct -j 21015028,21015029 --format=JobID,JobName%14,State,ExitCode,Elapsed -n | awk '!/\./'
# Export log:
tail -50 .../perf_campaign/w17_export-21015028.out
# Gate log / mp:
.../logs/path_uma-21015029.out
.../perf_campaign/matrix/nacl6_uma_ufast_w17_ngpu2/mp_logs/worker_r0.log
```

**PASS criteria for NaCl@2:**

- Export: `VERIFY_OK no aten::rand` / `W17_EXPORT_OK`
- Worker: `path=graph_capture` then later `path=graph_replay` (not stuck on `graph_fail_eager`)
- E/F vs ASE merge PASS
- Job exit 0 (hygiene must not kill eager fallback if capture still fails)

**If PASS:**

```bash
bash .../perf_campaign/w17_submit_matrix.sh 21015029
# Then gate @4 NaCl + water@2/@4; promote only if speed wins + E/F hold
```

**If FAIL again:** stamp `gate_v6_w17c_*.json` as structural; **do not** blind-resubmit. Product remains W8nk. Declare W17 closed and apply hard-ceiling rule (W14 already &lt;1 ms; graph stretch exhausted unless a *new* capture-safe design exists — e.g. strip wrapper from captured region, or FairChem-side graph API).

### B. Keep living docs honest

- `STATUS.md` stamp + Queue (live) + Next  
- `STATE.json` `status` / `active_jobs` / `next_action`  
- Append `tick.log`  
- Force-add, commit, push

### C. Wake loop

A 2-min wake may be running (`AGENT_LOOP_TICK_uma_perf`). Re-arm if dead:

```bash
pkill -f 'AGENT_LOOP_TICK_uma_perf' || true
nohup bash -c 'while true; do sleep 120; echo "AGENT_LOOP_TICK_uma_perf {\"prompt\":\"<current next action>\"}"; done' \
  > /tmp/uma_perf_wake.log 2>&1 &
```

Poller log for W17c: `/tmp/uma_w17c_poll.log`.

---

## 8. Decision tree after W17c

```text
W17c NaCl@2
├─ graph_capture + E/F PASS
│   ├─ submit matrix (w17_submit_matrix.sh)
│   ├─ all cells E/F PASS and speed ≥1 ms vs W8nk or clear ASE water@4
│   │    → PROMOTE graph recipe (document env: UMA_CUDA_GRAPH=1 UMA_EDGE_PAD=1 + cgraph art)
│   └─ else NO_PROMOTE; keep W8nk; optional leave graph as opt-in
└─ capture still illegal / job fail
    → STRUCTURAL close W17
    → Product W8nk
    → Hard ceiling: remaining ~1.2 ms water@4 is model/wait (FairChem / algo), not pipe sync
    → Optional follow-ons outside V6 graph: multi-node MPI plan, FairChem merge_mole FP64 FC, devices=1 water NVT floor
```

---

## 9. Explicit non-goals (this campaign)

- Multi-node MPI (see `uma-engine/docs/multinode_mpi_plan.md`) — separate track  
- Fixing FairChem FC `merge_mole` FP64 crash  
- Re-promoting W10 alone  
- FP32 / Triton turbo  
- Changing ASE oracle geometry  

---

## 10. Glossary shortcuts

- **W8nk** = W8-fix NCCL + no Kokkos + `pair_style uma`  
- **ufast** = `umas_fast_pytorch` + `merge_mole=True`  
- **cgraph art** = `*-f64-fast-cgraph` (γ=0 + capture-oriented export)  
- **PERF_PARENT** = parent timing (`ms_wait_workers`, `ms_vesin`, …)  
- **PERF_TICK** = per-rank worker (`ms_fwd`, `ms_bwd`, `ms_force_ar`, `path=…`)  
- **EDGE_PAD** = `UMA_EDGE_PAD=1` fixed edge capacity (required for graph)  

Canonical terms: `GLOSSARY.md`.

---

## 11. Suggested first prompt for the next agent

> Continue the UMA FP64 multi-GPU perf campaign on branch `uma-kokkos-mlip`. Read `perf_campaign/HANDOFF.md` and `STATUS.md`. Product floor is W8nk. Drain W17c jobs `21015028` (export) and `21015029` (NaCl@2 graph). Gate `path=graph_capture` + E/F vs ASE merge; if PASS run `w17_submit_matrix.sh`; if FAIL stamp structural close and keep W8nk. Always FP64; RECOMPILE=1 only on first job of a code version; force-track stamps, commit, push.

---

## 12. Snapshot at handoff time

| Item | Value |
|------|--------|
| Branch / tip | `uma-kokkos-mlip` @ `51a6d8c75f` |
| STATE.status | `w17c_submitted` |
| Queue | `21015028` w17-exp PENDING (Priority); `21015029` afterok Dependency |
| Product | **W8nk** |
| Wake | armed for W17c (verify with `ps aux \| grep AGENT_LOOP_TICK`) |
