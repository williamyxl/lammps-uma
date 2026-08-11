# Multi-node (2 × 4 A100) — capability report

**System:** NaCl 9×9×9 perturb-then-replicate, 5832 atoms, FP64, NVT 300 K
**Target:** 2 nodes × 4 A100 40 GB = 8 GPUs
**Stamp:** 2026-08-10

Triage rule applied: if a failure was **our setting**, it was fixed and re-run;
if the code **cannot** run multi-node, it is reported as
**FUNCTIONALITY UNSUPPORTED** and not run again.

---

## Summary

| Path | Multi-node | Verdict |
|---|---|---|
| **FC LAMMPS** | **UNSUPPORTED** | `lmp` built `-DBUILD_MPI=OFF` → *"MPI v1.0: LAMMPS MPI STUBS"*. A single process cannot span nodes. Compounded by fairchem's Ray placement being `STRICT_PACK` per node. |
| **ASE FC FP64** | **UNSUPPORTED (as packaged)** | `load_predict_unit(..., workers=N)` exposes no `num_workers_per_node`; it is hardcoded to a default of 8, so `num_nodes = ceil(8/8) = 1`. All 8 workers are placed on one node. |
| **ALCHEMI** | **SUPPORTED, but out of memory at this size** | Multi-node machinery is real (SLURM-aware `DistributedManager`). Both decomposition strategies exceed 40 GB/GPU for 5832 atoms at 2 ranks, so 8 ranks cannot help. |
| LibTorch UMA LAMMPS | **UNSUPPORTED (by design, same-node)** | `/dev/shm` + node-local NCCL. Multi-node is an explicit non-goal; see `uma-engine/docs/multinode_mpi_plan.md` (still PLAN). |

**No 2-node measurement was obtained for any path.**

---

## FC LAMMPS — UNSUPPORTED

Job `21023815`, 2 nodes, TIMEOUT after 45:20. Recorded evidence:

```
=== BLOCKER 1: does lmp have real MPI? ===
MPI v1.0: LAMMPS MPI STUBS for LAMMPS version 22 Jul 2025
=== BLOCKER 2: FC FP64 + merge_mole on this box ===
INFO: Started a local Ray instance
INFO: Creating placement groups with [8] workers on cuda
*** CANCELLED DUE TO TIME LIMIT ***
```

Two independent blockers:

1. **MPI STUBS.** The binary has no real MPI, so one `lmp` process cannot span
   nodes. Would require rebuilding with `-DBUILD_MPI=ON` plus re-validation.
2. **Ray placement.** fairchem builds one placement group per node with
   `strategy="STRICT_PACK"`, and the local Ray instance had only 4 GPUs, so the
   8-worker request could never be satisfied. It hung until the wall clock.

Not a setting we can fix from our side. **Not re-run.**

## ASE FC FP64 — UNSUPPORTED as packaged

Job `21023814` — **cancelled without running**, because inspecting the API
showed it cannot succeed:

```python
load_predict_unit(path, ..., workers: int = 1)      # no per-node control
ParallelMLIPPredictUnit(..., num_workers: int = 1,
                        num_workers_per_node: int = 8)   # not plumbed through
```

`num_nodes = math.ceil(num_workers / num_workers_per_node)`. With `workers=8`
and the default `num_workers_per_node=8`, `num_nodes = 1`: all workers are
packed onto a single node regardless of the allocation. `num_workers_per_node`
is not reachable through `load_predict_unit`, so this cannot be fixed by a
setting on our side.

Also note the campaign rule *"1 MPI rank; no Ray"* — the Ray path would not
produce numbers comparable to the locked ASE bars even if it ran.

## ALCHEMI — supported, but OOM at 5832 atoms

Two genuine **our-side** bugs were found and fixed:

| Bug | Fix |
|---|---|
| `device="cuda:N"` → `AssertionError: device must be either 'cpu' or 'cuda'`; all 8 ranks died at model load (job `21023813`) | pin with `torch.cuda.set_device(device)`, pass `device.type` |
| Standalone single point called `model(batch)` on the **full** system on every rank *before* `DomainParallel` existed, so `--strategy` never applied | skip the undecomposed SP above `ALCHEMI_SP_ATOM_CAP` (2048 atoms) and let the decomposed MD block produce E/F |

After both fixes the strategy is correctly applied (`"strategy":
"graph_partition"`, `sp_skipped` recorded), but per-rank memory at 2 ranks is:

| Strategy | per-rank VRAM | result |
|---|---:|---|
| `halo` | 37.18 GiB | OOM |
| `graph_partition` | 39.26 GiB | OOM |

Both exceed the 39.49 GiB usable on an A100 40 GB. `graph_partition` *"holds
the full geometry replicated"* per rank by design, so per-rank memory does not
fall with rank count for this system — **8 ranks would not change the outcome**,
which is why the 2-node job was not resubmitted.

ALCHEMI multi-node remains **unverified**, not disproven: the machinery is
present, the system size is simply too large for this GPU at these rank counts.
A smaller box (e.g. NaCl 6³, 1728 atoms at ~24.7 GiB) would test it.

---

## Reproducers

- `/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/smoke_1node.slurm`
  — cheap 1-node/2-rank rehearsal of the same `srun` + `DistributedManager` +
  `DomainParallel` path. Catching the OOM here avoided burning further 2-node
  allocations.
- `/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/run_fclammps_nacl9.py`
  — probes and records the MPI-STUBS blocker as evidence rather than assertion.
