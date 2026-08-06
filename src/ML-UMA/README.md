# ML-UMA — LibTorch UMA pair style for LAMMPS

Provides `pair_style uma` and `pair_style uma/kk` backed by vendored
[`uma-engine/`](uma-engine/) (GPU-persistent LibTorch, task **omat**, forces via
autograd).

## Precision (runtime, like GPU/INTEL packages)

| Mode | Positions | Energy | Forces |
|------|-----------|--------|--------|
| `mixed` (default) | FP32 | FP32 | FP64 |
| `double` | FP64 | FP64 | FP64 |

```lammps
pair_style uma/kk precision mixed
pair_style uma/kk precision double
# bare aliases also accepted:
pair_style uma/kk mixed
pair_style uma/kk double
```

The TorchScript artifact dtype must match the selected mode
(`uma-s-1p2-omat` for mixed, `uma-s-1p2-omat-f64` for double).

This LAMMPS tag lacks `KOKKOS_PREC=mixed`. Do **not** pass
`-DPREC_POS/FORCE/ENERGY` (breaks Kokkos on this tag). Pair/engine precision
is selected solely via the `pair_style` keyword above.

## Build (with Kokkos CUDA)

```bash
./scripts/build_lammps_uma.sh
```

## Input

```lammps
pair_style uma/kk precision mixed
pair_coeff * * /path/to/ML-UMA/uma-engine/artifacts/uma-s-1p2-omat Na Cl
newton off
```

## Examples and reports

Parity drivers, LAMMPS inputs, frozen structures, and JSON reports live in
[`examples/`](examples/) (tracked with this package inside the `lammps/` git tree).

See [`examples/README.md`](examples/README.md).
