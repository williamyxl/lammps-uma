# PENDING — report close-out

> **Precision:** `uma/kk` mixed is **disabled**. Active UMA path = double/FP64 only.  
> **Stamp:** 2026-08-07 ~17:42 CDT

| # | Item | Jobs | Artifact | Status |
|---|------|------|----------|--------|
| 1 | ASE FairChem **FP64** ground truth (`workers=1`) | promoted / `20898588` | `gp_round/oracle_ase_fp64_w1.{json,npz}` | **DONE** |
| 2 | ASE FairChem FP64 **@ 4 GPU** timing/E/F | `20909845` | `results/ngpu4/parity.json` + `forces.npz` | **DONE** (117.7 ms) |
| 3 | FairChem **FC LAMMPS** @ 4 GPU | `20909846` cancelled; `20910353` pending | merge into `results/ngpu4/` | **PENDING** |
| 4 | uma double @ 4 GPU (gp_round) | `20907648` | `gp_round/ngpu4/` | **DONE** |
| 5 | Optional requeue ASE/double path-isolation @4 | `20910352` / `20910354` | — | PENDING (redundant if #2/#4 kept) |
| — | ~~uma mixed~~ | scanceled | — | **DISABLED** |
| G1 | OOM sweep N=8/10/12 | see `multi_node_nacl6` | `SWEEP.md` **N\*=10**; N12 ase OOM | **N\* known**; N12 fc/uma still queued |

Already in hand:

- uma GP **double** 1/2/4 timing + E/F gates → `results/gp_round/`
- ASE 1/2/4 timing → `results/ngpu{1,2,4}/` + oracle
- FC 1/2 → `results/ngpu{1,2}/`

When FC @4 is green: refresh RESULTS/SUMMARY/canvas with full ASE/FC/uma 1→2→4 table; stop pending poll loops.
