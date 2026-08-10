# Campaign glossary — FairChem knobs only

**Rule:** Prefer writing the FairChem settings and artifact path names below.
Do **not** invent new campaign synonyms (`turbo`, `fast+merge`, `merge-only`, …)
in new STATUS/STATE text. If an older stamp used those phrases, decode them
here; do not extend the vocabulary.

**Scope:** FairChem UMA-S-1.2 **omat** FP64. C++ path: `uma/kk`.

---

## FairChem settings (canonical)

| Setting | Values used | CLI |
|---------|-------------|-----|
| `InferenceSettings.execution_mode` | `general` · `umas_fast_pytorch` | `--execution-mode` |
| `InferenceSettings.merge_mole` | `False` · `True` | `--merge-mole` |

Constraint from FairChem: `umas_fast_pytorch` requires `merge_mole=True`
(Tier1c `fastnomole` export failed). `umas_fast_gpu` (Triton) is out of scope
for TorchScript `uma/kk`.

FairChem also has an **`InferenceSettings` preset named `turbo`**. That preset
is **not** used in this campaign and is **not** an alias for
`umas_fast_pytorch`. Product parity README: do not use that preset.

---

## Artifact combinations (canonical)

Name a run by its settings and artifact directory — nothing else.

| `execution_mode` | `merge_mole` | Artifact dir | Campaign role |
|------------------|--------------|--------------|---------------|
| `general` | `False` | `uma-engine/artifacts/uma-s-1p2-omat-f64` | Tier0 / product |
| `general` | `True` | `…/uma-s-1p2-omat-f64-merge` | Tier1c isolate (MOLE fuse only) |
| `umas_fast_pytorch` | `True` | `…/uma-s-1p2-omat-f64-fast` | Tier1+ / Wave A / Tier2 speed path |

Export for the speed path:

```bash
python export_mp_artifact.py --execution-mode umas_fast_pytorch --merge-mole ...
```

---

## ASE E/F oracles and speed bars

Speed and E/F use **matching** FairChem settings: the ASE/FC bar for a uma
artifact uses the same `execution_mode` / `merge_mole` as that artifact.

| Gate against | ASE/FC settings | Used for |
|--------------|-----------------|----------|
| ASE/FC `general` | `execution_mode=general`, `merge_mole=False` | Tier0 `*-f64`; **locked** multi-GPU table (do not re-run) |
| ASE/FC `general`+`merge_mole` | `general`, `merge_mole=True` | `*-f64-merge` (measure once, then lock) |
| ASE/FC `umas_fast_pytorch`+`merge_mole` | `umas_fast_pytorch`, `merge_mole=True` | `*-f64-fast` (measure once, then lock) |

**Hard speed gate (Tier1+):** uma ms ≤ matching ASE **and** ≤ matching FC on
the gated metric. Clearing only the locked `general` table is **necessary but
not sufficient** once `merge_mole` / `umas_fast_pytorch` bars exist.

E/F oracles (ASE FP64 @1, frozen geometry):

| Oracle | Settings | Artifact |
|--------|----------|----------|
| ASE `general` | `general`, `merge_mole=False` | Tier0 `*-f64` |
| ASE `merge_mole=True` | stamps `oracle_ase_merge_mole.json` / water twin | `*-f64-fast`, `*-f64-merge` |

Oracle stamps (reuse): NaCl job `20983514`, water `20984160`. Rows:
`general_merge`, `umas_fast_merge`.

On NaCl6, ASE `general_merge` and ASE `umas_fast_merge` agree to ~10⁻¹³ eV;
both sit ~2.1×10⁻⁵ eV above ASE `general` (expected MOLE-fuse residual).

| Comparison | Typical `\|ΔE\|` |
|------------|------------------|
| `*-f64-fast` or `*-f64-merge` vs ASE `general` | ≈ 2×10⁻⁵ eV (expected; do not FAIL) |
| same vs ASE `merge_mole=True` | ∼ 10⁻¹⁰–10⁻¹¹ eV (PASS bar) |

Env knobs for ASE/FC reference jobs: `FAIRCHEM_EXECUTION_MODE`,
`FAIRCHEM_MERGE_MOLE` (see `run_multigpu.py` / water `path_common.py`).
Defaults remain product `general` / `merge_mole=False`.

---

## Legacy phrase decode (do not reuse in new prose)

| Legacy phrase | Means (write this instead) |
|---------------|----------------------------|
| turbo / Tier1 turbo / campaign turbo | `execution_mode=umas_fast_pytorch`, `merge_mole=True` (`*-f64-fast`). **Not** FairChem `InferenceSettings` turbo. |
| fast+merge / umas_fast+merge | same as above |
| merge-only / general+merge | `execution_mode=general`, `merge_mole=True` (`*-f64-merge`) |
| general ASE | ASE with `execution_mode=general`, `merge_mole=False` |
| merge ASE | ASE with `merge_mole=True` (oracle stamps above) |
