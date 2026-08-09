# uma/kk perf campaign — living status

**Stamp:** 2026-08-09T15:00:00 CDT · Loop **armed** (2 min) · **Tier 0 COMPLETE — speed + E/F parity documented**
**State:** `STATE.json` · Plan: `v5_max_perf_push_82db7365.plan.md` (**CURRENT**)
**Parity stamp:** `gate_v5_tier0_parity_summary.json`

## Headline bars (product FP64) — Tier 0 speed

| Suite | Metric | @1 (pre) | @2 Tier0 | @4 Tier0 | ASE @4 | Gate |
|-------|--------|---------:|---------:|---------:|-------:|------|
| NaCl6 | pair ms | **321** | **172.9** | **100.2** | 115.2 | **PASS** |
| water888 | NVT Pair ms/step | **333** | **178.3** | **104.2** | **118.0** | **PASS** (was FAIL ~126) |

Margins @4 vs ASE: NaCl **−15.0 ms**; water **−13.8 ms**.

Jobs: NaCl `20981757/59` · water `20981758/60`.

## Tier 0 energy + force parity

Formal gates (NaCl vs `uma@1` d1): `|ΔE| ≤ 1e-8`, `max|ΔF| ≤ 1e-6`, `cosine ≥ 1−1e-12`. Water first-frame vs ASE@1 FP64 oracle (same residual class as pre-Tier0 `uma@1/@2/@4` in `water888/results/COMPARE.md`).

### NaCl6 — `uma/kk precision double` (jobs 20981757 @2, 20981759 @4)

| ngpu | E (eV) | \|ΔE\| vs ASE | \|ΔE\| vs uma@1 | max\|ΔF\| vs ASE | max\|ΔF\| vs uma@1 | cosine ASE | cosine d1 | gate |
|------|--------|--------------:|----------------:|-----------------:|-------------------:|-----------:|----------:|------|
| 2 | −5830.92372016672 | 1.24e-10 | 1.82e-12 | 5.00e-07 | **0** | 0.999999999999 | **1.0** | **PASS** |
| 4 | −5830.92372016672 | 1.25e-10 | 1.82e-12 | 5.00e-07 | **0** | 0.999999999999 | **1.0** | **PASS** |

Source: `multi_gpu_nacl6/results/ngpu{2,4}/parity.json` (`parity_gates_summary.all_passed=true`).

### water888 — first-frame E+F vs ASE@1 (jobs 20981758 @2, 20981760 @4)

| ngpu | E (eV) | \|ΔE\| vs ASE@1 | max\|ΔF\| | force MAE | cosine | SP ms | NVT ms |
|------|--------|----------------:|----------:|----------:|-------:|------:|-------:|
| 2 | −3143.38935611817 | 3.00e-11 | 4.96e-06 | 7.41e-07 | 0.999999999998 | 178.7 | **178.3** |
| 4 | −3143.38935611817 | 3.00e-11 | 4.96e-06 | 7.41e-07 | 0.999999999998 | 102.8 | **104.2** |

ASE@1 oracle: E=−3143.3893561182 (`ase_ngpu1_20948821`). Force residual **matches** historical uma@1/@2/@4 (`max|ΔF|≈4.96e-06`, cosine≈1−1.5e-12) — not a Tier0 regression. Sources: `water888/results/uma_ngpu{2,4}_209817{58,60}/{timing.json,forces.npz}`.

## Tier 0 landed

I0 · W1 cover-once · W2 GPU degree skip · W3 AG equal-shard · W4 NCCL/H2D sync diet · W5 deferred.

## Next

**Tier 1:** re-export MP + devices=1 with `merge_mole=True` + `execution_mode=umas_fast_pytorch`; parity vs ASE FP64 oracle; re-gate @2/@4. Export job `20982262` → `artifacts/uma-s-1p2-omat-f64-fast/`.

## Constraints

FP64 · `uma/kk` + Kokkos · 1 MPI · no Ray · full parent NL · no force-reduce skip.
