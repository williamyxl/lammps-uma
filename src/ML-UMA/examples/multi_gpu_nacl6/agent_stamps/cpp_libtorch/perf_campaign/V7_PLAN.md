# V7 — uma-engine optimization campaign

**Opened:** 2026-08-10 · **Branch:** `uma-kokkos-mlip` · **Product floor:** W8nk (unchanged until a wave promotes)

Successor to V6 (W17 closed structural: FairChem `int(torch.sub(NumToTensor(...)))`
host-sync blocks CUDA-graph capture, 35 sites / 23 files — not fixable by re-export).

---

## 1. Why this campaign exists

W13 profile of water@4 (the authoritative budget):

| Component | ms | % step |
|---|---:|---:|
| parent total | 94.80 | 100% |
| ├─ vesin NL | 1.81 | 1.9% |
| ├─ d2h + pub + pack + shard | 0.59 | 0.6% |
| └─ **wait_workers** | **92.71** | **97.8%** |
| ⠀⠀├─ rank fwd | 42.09 | |
| ⠀⠀├─ rank bwd | 16.91 | |
| ⠀⠀├─ force all-reduce | 0.026 | 0.03% of wait |
| ⠀⠀└─ **UNACCOUNTED** | **33.69** | **35.5% of step** |

Two facts set the scope:

1. **The parent side is done.** NL + all publish machinery is 2.5% combined.
   W1–W14 harvested it. More parent work is dead-end.
2. **33.7 ms/step — a third of the step — has no counter.** Everything else in
   this plan is speculation until that is measured.

Hard ceiling on the whole campaign: fwd+bwd = 59 ms of the 92.7 ms wait, i.e.
**64% is FairChem model math we do not own**. Even a perfect engine leaves
~59 ms/step. Scope accordingly — the target is the 33.7 ms, not the 59 ms.

---

## 2. Where the 33.7 ms can hide (from source reading)

`tests/uma_libtorch_mp_worker.cpp`:

- **Pre-bwd barrier (L355–356)** sits *between* the `t_fwd0` and `t_bwd0`
  timers, so its cost lands in neither. This is load-imbalance absorption.
- **Untimed pre-forward setup (L236–305)**: full-graph H2D, `shard_edges`
  (`isin` + `nonzero` + 2× `index_select` over all E, per rank per step),
  int32→FP64 cast, optional edge pad, small tensor construction.

Prior evidence that constrains hypotheses:

- **W12 already removed the barrier** (`UMA_SKIP_PRE_BWD_BARRIER=1`): no cell
  gained ≥0.5 ms, NaCl@4 *regressed* 2.3 ms. So the barrier is not the cost —
  it *reveals* cost. Remove it and the wait reappears inside the backward NCCL
  collective. **Do not re-run barrier removal as a speed play.**
- Force all-reduce is 0.026 ms. Communication is not the bottleneck at N=4.

---

## 3. Waves

### W18 — instrument the unaccounted block (**do first, gates everything**)

Add worker-side timers, emitted on `PERF_TICK`:

| New counter | Region |
|---|---|
| `ms_h2d` | payload read → all H2D complete |
| `ms_shard` | `shard_edges` (isin/nonzero/index_select) |
| `ms_pad` | edge pad when `UMA_EDGE_PAD=1` |
| `ms_prep` | tensor construction + requires_grad |
| `ms_bar_pre_bwd` | the barrier, timed **separately** from fwd/bwd |
| `ms_post` | force all-reduce → payload publish |

Success = `ms_fwd + ms_bwd + Σ(new) ≈ ms_wait_workers` to within ~2 ms.
**No optimization is attempted before this closes.** Zero risk to E/F: pure
instrumentation.

### W19 — parent-side sharding (hypothesis: redundant H2D + O(E) filter)

Today every rank H2Ds the **entire** edge list, then filters to ~1/N. That is
N× the necessary PCIe traffic plus an O(E) `isin`/`nonzero` per rank per step.

Change: shard on the parent, publish per-rank edge slices, workers H2D only
their own. Expected to scale with atom count (bigger at 5832 than 648) and with
rank count — i.e. it should matter more at @4 than @2.

Gate on W18 numbers: **only pursue if `ms_h2d + ms_shard` is actually large.**

### W20 — edge-balanced partition (hypothesis: load imbalance)

`node_partition` splits by **atom index**; cost is driven by **edges**. W13 fwd
spread is 41.56–42.09 ms (small), but the barrier absorbs whatever exists, so
the true imbalance is only visible after W18.

Change: partition by cumulative edge count instead of atom count. Must preserve
FairChem's `edge_index[1] ∈ node_partition` contract — E/F parity is the gate,
and a wrong partition silently changes forces.

Gate on W18 `ms_bar_pre_bwd` spread across ranks. **Skip if imbalance < 1 ms.**

---

## 4. Gates (every wave)

1. **E/F parity** vs ASE merge oracle, per-atom:
   `|ΔE| ≤ 1e-6 eV` **and** per-atom `max|ΔF| ≤ 1e-5 eV/Å`.
   Net force `|Σ F|` is *not* sufficient — it is bit-identical under sign
   inversion. Use `force_parity.py`.
2. **Speed**: NVT Pair ms/step, warmup excluded. NaCl `NSTEPS=10`,
   water `NSTEPS=100`.
3. **Promote only if** E/F PASS **and** (≥1 ms median win vs W8nk, or clears
   the ASE ufast floor).
4. **Hard-ceiling stop**: two consecutive waves each moving median <1 ms with a
   flat wait profile → close the campaign.

Cells: NaCl@2, NaCl@4, water@2, water@4. NaCl@1 for reference.

**ASE FC and FC LAMMPS are frozen** — their code is unchanged, so reuse locked
bars (`STATE.json:baselines_locked`, `matching_ase_fc`). Do not re-run them.

---

## 5. Ops rules (inherited, non-negotiable)

- `RECOMPILE=1` on the **first** job of a new code version only; dependents
  `RECOMPILE=0`. **Never** dual-submit two `RECOMPILE=1` into `build-uma`
  (this raced CMakeTmp and killed the first W12 submit).
- FP64 only. 1 MPI rank, no Ray, full parent NL, no force-reduce skip.
- Force-add stamps → commit → push; never leave stamps only on disk.
- Do not retry closed waves without new evidence: W10 edge-pad promote,
  W12 barrier removal, W17 CUDA graph, FP32, Triton `umas_fast_gpu`.

---

## 6. Known-open, explicitly out of scope

- **nvalchemi water888 E/F FAIL** (ΔE 6.4e-3, all 648 atoms over tol, while
  NaCl passes at 1.7e-10). Separate bug in the nvalchemi path, not uma-engine.
- **Multi-node**: engine is `/dev/shm` + node-local NCCL. W19/W20 make the
  `multinode_mpi_plan.md` work *more* valuable but do not substitute for it.
- **ASE single-GPU oracle ceiling**: OOMs at 5832 atoms on 40 GB
  (measured, job 21023795 — 37.38 GiB in use, needed +3.21 GiB). Oracles above
  ~4096 atoms need multi-GPU.
