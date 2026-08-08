# Multi-GPU graph parallel for uma-engine / uma/kk

Same-node in-model graph parallel (`devices N`) for UMA inference behind
`pair_style uma/kk`. Scientific weights stay `uma-s-1p2`; neighbor-list scheme
and energy/force parity vs `devices=1` are hard gates.

## Neighbor-list contract (must match)

Implementation reference: `Predictor::rebuild_neighbors()` in
`uma-engine/src/predictor.cpp`.

| Parameter | Source | Value (omat artifacts) |
|-----------|--------|------------------------|
| cutoff | `metadata.json` | 6.0 Å |
| max_neighbors | `metadata.json` | 300 (per-center cap) |
| PBC wrap | vesin / CPU NL | positions wrapped into cell before NL |
| Edge orientation | FairChem | `edge_index[0]=neighbor`, `edge_index[1]=center` |

**CUDA path (devices=1 traced):** vesin `cell_list` (`full_directed=true`), then
flip rows from vesin’s (center, neighbor) to FairChem (neighbor, center).

**CPU fallback:** `build_neighbor_graph` already emits FairChem orientation.

**devices>1 (eager FairChem GP):** workers>1 uses FairChem *internal* graph
generation (`external_graph_gen=False`) — same as ASE multi-GPU. Assert NL
scheme match via E/F parity gates rather than bit-identical edges. Long-term
native LibTorch GP should feed shards from the engine NL above.

Multi-GPU must not change cutoff / max_neighbors / orientation.

## Partition key (FairChem-compatible)

Match FairChem `eSCNMD` / `gp_utils` / `filter_edges_by_node_partition`:

1. `node_partition = tensor_split(arange(N), world_size)[rank]` (contiguous atom chunks).
2. Keep edges where **target** `edge_index[1] ∈ node_partition`.
3. Message-passing gathers full node states as needed; energy/forces all-reduce / gather.

## Load path decision

| `devices` | Legacy (opt-in) | **Product** ([`native_kokkos_libtorch_gp.md`](native_kokkos_libtorch_gp.md)) |
|-----------|-----------------|-----------------------------------------------------------------------------|
| `1` | TorchScript `model_traced.pt` + vesin CUDA NL | unchanged |
| `N>1` | `UMA_ALLOW_RAY_GP=1` → FairChem Ray / `UMA_PYTHON_GP_WORKER=1` → Python GP | **C++** `LibtorchMpRuntime` + `model_mp_w*_n*_r*.pt` + `uma_peer` **CUDA IPC** + vesin shards; keep `uma/kk` + `-k on g N`. **No Ray.** |

Set `UMA_FORBID_RAY_GP=1` to reject Ray. Default peer transport: `UMA_PEER_TRANSPORT=cuda_ipc`.

Artifacts (export with `python/export_mp_artifact.py` / `export_artifact.py --dtype float64`):

- double / active: `artifacts/uma-s-1p2-omat-f64/` (`model_mp_w{N}_n{NATOMS}_r{R}.pt`)
- mixed: `artifacts/uma-s-1p2-omat/` — **disabled** for campaigns

**Landed (2026-08-08):** devices=2/4 E+F green · self-scale **≈320 / ≈265 / ≈193 ms** (job `20932975`). See `examples/multi_gpu_nacl6/results/RESULTS.md`.

## Parity thresholds (`devices=N` vs oracle)

Frozen geometry: `examples/multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz`.

**Permanent ground truth (record once):** ASE FairChem **FP64**, `workers=1`, no
ParallelMLIPPredictUnit — energy + forces cached at
`examples/multi_gpu_nacl6/results/gp_round/oracle_ase_fp64_w1.{json,npz}`
(E = −5830.9237201666 eV on NaCl6). Recompute only if geometry or checkpoint changes.

| Mode | Oracle (gp_round as run) | \|ΔE\| | max \|ΔF\| | force cosine |
|------|--------------------------|--------|------------|--------------|
| double | uma traced `devices=1` (future: ASE FP64@1 cache) | ≲ 1e-8 eV (prefer ~1e-10) | ≲ 1e-6 eV/Å | ≥ 1 − 1e-12 |
| mixed | ASE FairChem `float32` `workers=1` (future: ASE FP64@1 with looser band) | ≲ 5e-4 eV | ≲ 1e-5 eV/Å | ≥ 1 − 1e-10 |

**Why mixed ≠ traced `devices=1`:** FairChem eager `base_precision_dtype=float32`
(including `workers=1` and GP `workers=N`) stays within ~1e-4 of FP64 on NaCl6
(`E≈−5830.9237`), while traced mixed artifact energy is `E≈−5830.9819`
(`|ΔE|≈0.058`). Evidence: `examples/multi_gpu_nacl6/gp_round/f32_diag.json`.

Secondary: uma double vs ASE FP64@1 cache (~1e-10 class energy).

## Engine / CLI

```bash
# devices=1 (traced)
uma_parity_cli artifacts/uma-s-1p2-omat-f64 structure.txt
uma_parity_cli artifacts/uma-s-1p2-omat-f64 structure.txt --devices 1

# devices>1 (FairChem eager GP; needs GPU node + uma312)
uma_parity_cli artifacts/uma-s-1p2-omat-f64 structure.txt --devices 2
```

Env: `UMA_CHECKPOINT`, `UMA_PYTHON` (default `python3`), `UMA_GP_WORKER`.

## LAMMPS usage

```
pair_style uma/kk precision double devices 2
pair_coeff * * <artifact_dir> Na Cl
```

Launch: `--ntasks=1`, `lmp -k on g N -sf kk`, trust SLURM `CUDA_VISIBLE_DEVICES`.
If `devices` omitted and Kokkos sees ngpus>1, pair auto-sets `devices=ngpus` and logs.

Rebuild after WRITE:

```bash
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh && conda activate uma312
module load cuda/12.8 cmake/3.31.8
bash scripts/build_lammps_uma.sh
```

## Forbidden

- Approximate spatial tiling that drops long-range message passing
- Silent turbo / FP32 on paths labeled double
- Fake multi-GPU that only runs full evals serially
- MPI multi-rank domain-decomp MLIP in this round
