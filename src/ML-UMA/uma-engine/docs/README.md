# uma-engine docs index

**Start here: [`CAMPAIGN_SUMMARY.md`](CAMPAIGN_SUMMARY.md)** — authoritative
summary of the Polaris LAMMPS+UMA scaling campaign (single-node validation,
multi-node, capacity/checkpointing, offload, 2-node investigation, and the
scalability verdict).

## Current docs
| doc | topic |
|---|---|
| `CAMPAIGN_SUMMARY.md` | **authoritative** overview + conclusions + doc index |
| `multinode_mpi_plan.md` | overall plan, phases, Phase P0 |
| `../../alcf_polaris_single_node.md` | P0 1/2/4-GPU parity + timing report |
| `multinode_impl_polaris.md` | MPI edge-parallel (M3) design + result |
| `activation_checkpointing.md` | activation checkpointing method (capacity lever) |
| `cpu_offload_plan.md` | CPU-memory offload ladder (A1/A3/C1 measured) |
| `capacity_findings_4xa100.md` | 1/4-GPU capacity ceilings (baseline vs ckpt) |
| `lammps_checkpointing_Nge20.md` | checkpointing in LAMMPS incl. N=20/21 + multi-node |
| `two_node_investigation.md` | where 8-GPU/2-node breaks + improvement levers |

## Superseded (kept for history)
`*_outdated.md` — earlier point-in-time notes, superseded by the above.
