# Si / Al / Si4 pair_style uma single-point examples

Companion systems for covalent (Si diamond) and metallic (Al fcc) bonding.

| Dir | System | Atoms | Input |
|-----|--------|------:|-------|
| `si_sp/` | Si diamond 3×3×3 | 216 | `in.si_sp` |
| `si4_sp/` | Si diamond 4×4×4 | 512 | `in.si4_sp` |
| `al_sp/` | Al fcc 3×3×3 | 108 | `in.al_sp` |

```bash
lmp -in in.si_sp
lmp -in in.si4_sp
lmp -in in.al_sp
```

For FP64 energy matching ASE, use `precision double` and artifact
`uma-s-1p2-omat-f64`.
