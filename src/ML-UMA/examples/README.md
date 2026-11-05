# ML-UMA examples

Inputs for `pair_style uma` (FP64). Launch with `lmp -in in.*` — no Kokkos
suffix, no `-k on`.

```
getting_started/   short NaCl 5x5x5 NVT (start here)
min-sample/        NaCl NVT with data file + artifact extract helper
si_sp/ si4_sp/     Si diamond single-point
al_sp/             Al fcc single-point
```

```bash
lmp -in in.nacl5_nvt
```
