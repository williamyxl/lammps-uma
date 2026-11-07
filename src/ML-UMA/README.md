# ML-UMA — LibTorch UMA pair style for LAMMPS

`pair_style uma` is a native LAMMPS pair style backed by vendored
[`uma-engine/`](uma-engine/) (GPU-persistent LibTorch, forces via autograd).
It is not a Kokkos `/kk` variant and not FairChem's `fix external` Python
callback.

## User surface

```lammps
pair_style uma precision double virial auto ac off decomp graph neigh lammps
pair_coeff * * /path/to/uma-s-1p2-omat-f64 Na Cl
newton off
```

| Keyword | Values | Default |
|---|---|---|
| `precision` | `double` | `double` (FP64 only; `mixed` is refused) |
| `devices` | N | 1 (same-node fork workers; multi-rank uses one GPU per rank) |
| `virial` | `auto` / `yes` / `no` | `auto` (on when a barostat is present) |
| `ac` | `off` / `chunk` / `block` / `full` | `off` (required for NPT) |
| `decomp` | `graph` / `spatial` | `graph` (MPI graph-parallel) |
| `neigh` | `lammps` / `engine` | `lammps` |

`pair_coeff` names a TorchScript **artifact directory**. LAMMPS does not
import FairChem at run time. Runtime knobs are pair_style keywords only.

One MPI rank: `export_artifact.py` (``model_traced.pt`` in that directory).
W MPI ranks, `decomp graph`, `devices 1`: `export_blocks_xpu.py` with
`EXPORT_WORLD=W` and `EXPORT_RANK=0..W-1`, which writes `w{W}/r{R}/` under
the same directory `pair_coeff` names. A single-rank artifact will not
load on `mpiexec -n W`. The atom count is part of a W>1 export.

Full syntax: [`doc/src/pair_uma.rst`](../../doc/src/pair_uma.rst).

## Build

Requires CMake, MPI (recommended), and LibTorch. Traditional make is not
supported.

```bash
cmake -S cmake -B build -D PKG_ML-UMA=on \
  -D CMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build
```

Default GPU backend is CUDA. Intel GPU: `-D UMA_ENGINE_USE_XPU=on`.
Do not enable the KOKKOS package for this pair style. Graph-parallel uses
MPI ranks (one device per rank), not `lmp -k on -sf kk`.

See [`doc/src/Build_extras.rst`](../../doc/src/Build_extras.rst) (ML-UMA).

Export a one-rank artifact with
[`uma-engine/python/export_artifact.py`](uma-engine/python/export_artifact.py)
(`--task omat`, `--dtype float64`, `--checkpoint $UMA_CHECKPOINT`).
Graph-parallel (one rank per GPU) uses
[`uma-engine/python/export_blocks_xpu.py`](uma-engine/python/export_blocks_xpu.py);
see [`examples/PACKAGES/uma`](../../examples/PACKAGES/uma).

## Examples

[`examples/PACKAGES/uma`](../../examples/PACKAGES/uma) — NaCl NVT and NPT;
artifact path via `-var uma_art`. Weights are not in the tree.

## License

C++ is GPL-2 (LAMMPS). Vesin is BSD-3; nlohmann JSON is MIT. LibTorch and
fairchem-core are user-installed. **Do not ship** `uma-s-1p2.pt` or
exported artifacts (FAIR Chemistry License). See [`LICENSE`](LICENSE).
