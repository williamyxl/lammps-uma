# UMA multi-GPU NaCl 6×6×6 parity report (`gp_round`)

**Stamp:** 2026-08-07 · Delta A100-SXM4-40GB · uma GP **DONE** · ASE FP64@1 **cached**

## Ground truth

ASE FairChem FP64 `workers=1` (no ParallelMLIP): **E = −5830.9237201666 eV**, forces `(1728,3)` in  
`oracle_ase_fp64_w1.{json,npz}` (promoted from `results/ngpu1/`).

## Setup

- **System:** NaCl 6×6×6 rocksalt, 1728 atoms
- **Structure (immutable):** `src/ML-UMA/examples/delta_parity/structures/nacl6_rattle_fixed.extxyz`
- **Perturbation:** uniform_box δ=0.1 Å seed=0 (fixed — never re-rattle)
- **ASE@1 energy (legacy ASE path):** −5830.9237201666 eV
- **Precision:** uma double = FP64; mixed = explicit lower-precision path
- **Parity gates:** 4/4 passed (`all_passed=True`)

### Backend

| `devices` | Mechanism |
|-----------|-----------|
| 1 | Traced LibTorch `Predictor` (`pair_style uma/kk`) |
| N>1 | FairChem eager GP via `GraphParallelRuntime` → `uma_gp_worker.py` (`pair_style uma`; Ray owns GPUs) |

**Final speedup is not 1.00×.** The flat ~1× figure applied only to the earlier Kokkos `-k on g N` recipe where LibTorch stayed single-device. Graph-parallel `devices=N` scales.

### Oracles / comparison policy

| Mode | E/F comparison reference |
|------|--------------------------|
| **All paths** | ASE FairChem FP64 `workers=1` (`oracle_ase_fp64_w1.*`) |

Historical campaign also gated double vs traced `devices=1` and mixed vs ASE
float32@1; **do not use those for reported |ΔE| / max|ΔF|** going forward.

## Timing (ms/eval) — final

| Path | ngpu1 | ngpu2 | ngpu4 | Speedup 1→2 | Speedup 1→4 |
|------|------:|------:|------:|------------:|------------:|
| uma double | 322.2 | 192.4 | 112.6 | **1.67×** | **2.86×** |
| uma mixed | 246.4 | 148.7 | 91.2 | **1.66×** | **2.70×** |

## Energy / force parity (vs ASE FP64@1)

| Path | ngpu | Energy (eV) | ms/eval | \|ΔE\| vs ASE FP64 | max \|ΔF\| | gate |
|------|------|-------------|---------|--------------------|------------|------|
| uma double | 1 | −5830.9237201667 | 322.2 | 1.3×10⁻¹⁰ | 5.0×10⁻⁷ | PASS |
| uma double | 2 | −5830.9237201666 | 192.4 | ~10⁻¹² | 5.0×10⁻⁷ | PASS |
| uma double | 4 | −5830.9237201666 | 112.6 | ~10⁻¹² | 5.0×10⁻⁷ | PASS |
| uma mixed | 1 | −5830.9819335938 | 246.4 | **5.82×10⁻²** | 7.2×10⁻⁶ | FAIL |
| uma mixed | 2 | −5830.9234143138 | 148.7 | 3.06×10⁻⁴ | 7.0×10⁻⁶ | PASS |
| uma mixed | 4 | −5830.9235703731 | 91.2 | 1.50×10⁻⁴ | 7.0×10⁻⁶ | PASS |

## Jobs

| Config | Job ID | Outcome |
|--------|--------|---------|
| devices=1 | `20901312` | COMPLETED |
| devices=2 double | `20903160` | PASS |
| devices=2 mixed | `20903538` | PASS |
| devices=4 (hung) | `20904146` | TIMEOUT mid mixed NVE |
| devices=4 | `20907648` | PASS (~4.4 min) |

## Notes

- Machine-readable: `gp_round/SUMMARY.json`, `ngpu{1,2,4}/parity.json`, `forces.npz`.
- Narrative summary: `gp_round/RESULTS.md`.
- Next policy: **done** — gate / report E+F vs cached ASE FairChem FP64 `workers=1`.
