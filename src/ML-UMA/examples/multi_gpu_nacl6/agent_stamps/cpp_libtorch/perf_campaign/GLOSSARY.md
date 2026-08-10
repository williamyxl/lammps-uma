# Campaign glossary — artifact modes & E/F oracles

**Audience:** `STATUS.md`, `STATE.json`, Tier0/Tier1/Wave A/Tier2 gate notes.  
**Scope:** FairChem UMA-S-1.2 **omat** FP64 only. These terms are **not** interchangeable.

---

## Two independent FairChem axes

FairChem inference is configured with (at least) two orthogonal knobs. Campaign
names combine them.

| Axis | FairChem setting | Values used here |
|------|------------------|------------------|
| **Execution backend** | `InferenceSettings.execution_mode` / export `--execution-mode` | `general` · `umas_fast_pytorch` |
| **MOLE fuse** | `InferenceSettings.merge_mole` / export `--merge-mole` | `False` · `True` |

**Hard constraint:** `umas_fast_pytorch` **requires** `merge_mole=True` (MOLE
experts have no fused `.weight` otherwise). `umas_fast` + `merge_mole=False`
is **illegal** for export/trace (Tier1c `fastnomole` failed).

`umas_fast_gpu` (Triton) is **out of scope** — not TorchScript-traceable into
the C++ `uma/kk` runtime.

---

## Named modes (use these spellings)

### Product / Tier0 — **general** (not turbo)

| Field | Value |
|-------|--------|
| Also called | product path, Tier0 art, `*-f64` |
| `execution_mode` | `general` |
| `merge_mole` | `False` |
| Artifact dir | `uma-engine/artifacts/uma-s-1p2-omat-f64` |
| E/F oracle | **general ASE** (below) |

Default product export in `common.py` / historical multi-GPU parity README.

### **merge-only** (also: **general+merge**)

| Field | Value |
|-------|--------|
| `execution_mode` | `general` |
| `merge_mole` | `True` |
| Artifact dir | `…/uma-s-1p2-omat-f64-merge` |
| Meaning | Fuse MOLE experts for fixed composition/charge/spin; keep **general** SO2/radial backend |
| Role | Tier1c isolate — proves ~2×10⁻⁵ eV vs general ASE is from **MOLE fuse**, not from `umas_fast` |

### **fast+merge** (also: **umas_fast+merge**, campaign **turbo** path)

| Field | Value |
|-------|--------|
| `execution_mode` | `umas_fast_pytorch` |
| `merge_mole` | `True` |
| Artifact dir | `…/uma-s-1p2-omat-f64-fast` |
| Meaning | Block-diagonal SO2 GEMM + batched radial MLP (**PyTorch**, TS-traceable) **and** fused MOLE |
| Role | Tier1+ / Wave A / Tier2 default speed path for `uma/kk` MP shards |

Export: `export_mp_artifact.py --execution-mode umas_fast_pytorch --merge-mole`.

---

## ASE oracles (E/F only — not speed bars)

Speed gates always use **locked** ASE/FC multi-GPU timings (general FairChem /
FC LAMMPS). Do **not** re-run those for campaign iterations.

E/F oracles are **ASE FairChem FP64 @1 worker** on the same frozen geometry,
with calculator settings matching the uma artifact under test.

### **general ASE**

- ASE `FAIRChemCalculator` / predict unit: `execution_mode=general`, `merge_mole=False`, dtype float64.
- Ground truth for **Tier0 / product `*-f64`**.
- Cached: NaCl6 ASE@1 energy in parity / P3c stamps; water ASE@1 from path jobs.

### **merge ASE** (short for **ASE FP64 with `merge_mole=True`**)

- ASE calculator with **`merge_mole=True`**, dtype float64.
- Sub-rows (NaCl job `20983514`, water `20984160`):
  - **`general_merge`:** `execution_mode=general`, `merge_mole=True`
  - **`umas_fast_merge`:** `execution_mode=umas_fast_pytorch`, `merge_mole=True`
- On NaCl6 those two ASE energies agree to ~10⁻¹³ eV; both sit ~2.1×10⁻⁵ eV
  above **general ASE** (expected **MOLE-fuse residual**, not an `uma/kk` bug).
- Ground truth for **Tier1 turbo / fast+merge / merge-only** uma energies:
  gate `|ΔE|` / forces vs **merge ASE**, not vs general ASE.

Stamps: `oracle_ase_merge_mole.json`, `oracle_ase_water_merge_mole.json`.

---

## Campaign slang: **turbo**

| Use | Meaning |
|-----|---------|
| **Tier1 turbo** / **turbo path** (this campaign) | The **fast+merge** uma artifact + E/F vs **merge ASE**. Synonym for Tier1 speed push. |
| FairChem **`InferenceSettings` “turbo”** (elsewhere) | A separate FairChem preset (often looser / faster settings). **Not** what STATUS means by turbo. |

`examples/multi_gpu_nacl6/README.md` “Do **not** use turbo settings” refers to
FairChem InferenceSettings turbo for the **product parity** recipe — it does
**not** ban the campaign Tier1 **fast+merge** path documented here.

---

## Dual-oracle policy (summary)

| uma artifact | Gate E/F against | Gate speed against |
|--------------|------------------|--------------------|
| `*-f64` (general) | **general ASE** | locked ASE/FC ms (general) |
| `*-f64-fast` (fast+merge) | **merge ASE** | locked ASE/FC ms (same locked table) |
| `*-f64-merge` (merge-only) | **merge ASE** | locked ASE/FC ms |

Typical residuals:

- fast+merge or merge-only vs **general ASE:** `|ΔE| ≈ 2×10⁻⁵` eV → **expected**, do not FAIL.
- fast+merge or merge-only vs **merge ASE:** `|ΔE| ∼ 10⁻¹⁰`–`10⁻¹¹` eV → PASS bar.

---

## Quick decode table

| Phrase in STATUS | Means |
|------------------|--------|
| merge ASE | ASE FP64, `merge_mole=True` (oracle jobs above) |
| general ASE | ASE FP64, `merge_mole=False`, `execution_mode=general` |
| turbo | Campaign Tier1 **fast+merge** path (not FairChem InferenceSettings.turbo) |
| fast+merge / umas_fast+merge | `umas_fast_pytorch` + `merge_mole=True` → art `*-f64-fast` |
| merge-only / general+merge | `general` + `merge_mole=True` → art `*-f64-merge` |
