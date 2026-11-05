# ML-UMA — LibTorch UMA pair style for LAMMPS

Provides `pair_style uma` backed by vendored [`uma-engine/`](uma-engine/)
(GPU-persistent LibTorch, forces via autograd). This is a native pair style, not
a Kokkos `/kk` variant and not FairChem's `fix external` Python callback.

## Precision (FP64 only)

**Supported / default:** `precision double` with artifact `uma-s-1p2-omat-f64`.
`precision mixed` is refused.

```lammps
pair_style uma precision double virial auto ac off decomp graph
pair_coeff * * /path/to/uma-s-1p2-omat-f64 Na Cl
newton off
```

`pair_coeff` points at a TorchScript **artifact directory** (exported once from a
FairChem UMA checkpoint). The LAMMPS run does not import FairChem or any sibling
research tree. The only extra-repo file is the checkpoint used at **export** time
(`UMA_CHECKPOINT`), or the artifact directory at **run** time.

## Build

Requires CMake, MPI (optional but recommended), and LibTorch.

```bash
./scripts/build_lammps_uma.sh
```

Do not enable the KOKKOS package for this pair style. Graph-parallel uses MPI
ranks (one device per rank), not `lmp -k on -sf kk`.

Export artifacts with
[`uma-engine/python/export_artifact.py`](uma-engine/python/export_artifact.py)
(`--task omat`, `--dtype float64`, `--checkpoint $UMA_CHECKPOINT`).

## Examples

LAMMPS-tree examples: [`examples/PACKAGES/uma`](../../examples/PACKAGES/uma)
(NaCl NVT and NPT; artifact path via `-var uma_art`).
In-package smoke: [`examples/getting_started/`](examples/getting_started/).
