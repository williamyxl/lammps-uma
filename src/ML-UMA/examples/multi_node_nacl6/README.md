# Multi-node NaCl campaign (`multi_node_nacl6`)

**Handoff / live status:** Cursor plan `multi-node_mpi_uma_c20e7566` (section **Session handoff**). Spec: [`../../uma-engine/docs/multi_node_mpi.md`](../../uma-engine/docs/multi_node_mpi.md).

**Precision:** UMA = **`uma_double` / FP64 only**. `uma_mixed` is disabled.

## Phase G1 — OOM sweep (status 2026-08-07 ~17:42 CDT)

Canonical table: [`results/geom_sweep/SWEEP.md`](results/geom_sweep/SWEEP.md) · JSON: [`SWEEP.json`](results/geom_sweep/SWEEP.json)  
Canvas: [`uma-oom-sweep-nacl`](/u/xyan11/.cursor/projects/work-nvme-bfzx-xyan11-workdir-lammps-uma/canvases/uma-oom-sweep-nacl.canvas.tsx)

| N | natoms | ase | fc | uma_double | all_pass |
|--:|------:|:---:|:--:|:----------:|:--------:|
| 8 | 4096 | PASS | PASS | PASS | YES |
| 10 | 8000 | PASS | PASS | PASS | YES |
| 12 | 13824 | **OOM** | PENDING/`MISSING` | PENDING/`MISSING` | no |

**N\*** (largest all-pass) = **10**.

- Jobs done: `20911082`–`20911088`, `20911090` (N12 ase OOM).
- Still queued: `20911091` (N12 fc), `20911094` (N12 uma_double). Mixed jobs scanceled.
- N12 ASE OOM: CUDA OOM on A100 40GB during FairChem forward (~13824 atoms).

```bash
cd src/ML-UMA/examples/multi_node_nacl6
RECOMPILE=1 ./submit_oom_sweep.sh --n 8,10,12   # parallel; ase/fc/uma_double
# after jobs finish:
python merge_oom_sweep.py
```

Each job writes only `results/geom_sweep/NXX/<path>/` (no shared merge during jobs).
