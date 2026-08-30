# Phase G1 OOM sweep

_Stamp:_ auto-merged from `probe_status.json` under `/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_node_nacl6/results/geom_sweep`.

**Precision:** ase / fc / `uma_double` (FP64). `uma_mixed` disabled.

| N | natoms | ase | fc | uma_double | all_pass |
|--:|------:|-----|----|------------|----------|
| 8 | 4096 | PASS | PASS | PASS | YES |
| 10 | 8000 | PASS | PASS | PASS | YES |
| 12 | 13824 | OOM | MISSING | MISSING | no |

**N\*** (largest all-pass over ase/fc/uma_double) = **10**

Notes:

- N=12 ASE = OOM on A100 40GB (13824 atoms).
- N=12 fc / uma_double may still be PENDING in the queue.

