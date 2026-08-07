# ML-UMA — LibTorch UMA pair style for LAMMPS

Provides `pair_style uma` and `pair_style uma/kk` backed by vendored
[`uma-engine/`](uma-engine/) (GPU-persistent LibTorch, task **omat**, forces via
autograd).

## Precision (FP64 only)

**Supported / default:** `precision double` with artifact `uma-s-1p2-omat-f64`.

| Mode | Positions | Energy | Forces | Status |
|------|-----------|--------|--------|--------|
| `double` | FP64 | FP64 | FP64 | **use this** |
| `mixed` | FP32 | FP32 | FP64 storage (autograd still FP32) | **disabled** |

```lammps
pair_style uma/kk precision double
pair_coeff * * /path/to/ML-UMA/uma-engine/artifacts/uma-s-1p2-omat-f64 Na Cl
newton off
```

`precision mixed` remains in the C++ parser for historical scripts, but
**campaigns, getting_started, and submit defaults must not use it.** Mixed runs
an FP32 Torch energy graph and only upcasts forces to FP64 — that is not
force-accurate FP64. Re-enable only with an explicit owner request.

This LAMMPS tag lacks `KOKKOS_PREC=mixed`. Do **not** pass
`-DPREC_POS/FORCE/ENERGY` (breaks Kokkos on this tag). Pair/engine precision
is selected solely via the `pair_style` keyword above.

## Build (with Kokkos CUDA)

```bash
./scripts/build_lammps_uma.sh
# or: sbatch src/ML-UMA/examples/getting_started/build_uma.slurm
```

Export TorchScript artifacts with
[`uma-engine/python/export_artifact.py`](uma-engine/python/export_artifact.py)
(`--task omat` or `--all-tasks`, `--dtype float64`). See
[`uma-engine/README.md`](uma-engine/README.md).

## Examples and reports

Start here: [`examples/getting_started/`](examples/getting_started/).

Parity drivers and reports: [`examples/`](examples/) (see
[`examples/README.md`](examples/README.md)).
