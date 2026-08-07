# gp_round dry-run checklist (TEST-PREP)

Do **not** submit heavy jobs until WRITE lands `pair_style uma/kk ... devices N` and
`build-uma/lmp` is rebuilt.

**Precision:** `uma_double` / FP64 only. **`uma_mixed` disabled.**

## Pre-submit (login node)

```bash
cd /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6
./gp_round/rebuild_and_submit.sh   # syntax + geometry + pair_style line check
```

- [ ] `.write_agent_done.json` exists with `rebuild_required: true` (or run rebuild manually)
- [ ] `build-uma/lmp` rebuilt after WRITE C++ changes
- [ ] `python -m py_compile run_multigpu.py parity_gates.py collect_results.py`
- [ ] `pair_style uma/kk precision double devices 2` accepted by `lmp -h` or smoke `run 0`

## Submit order (path-isolated — one path per job)

VRAM isolation: **one** `ONLY_PATHS=uma_double` per allocation. Do not submit mixed.

1. **ngpu1 uma_double** — devices=1 baseline
2. **ngpu2 uma_double** — parity vs ngpu1 / ASE FP64
3. **ngpu4** — only after ngpu2 green

```bash
./gp_round/rebuild_and_submit.sh --submit
# optional 4-GPU after 2-GPU green:
./gp_round/rebuild_and_submit.sh --submit --ngpu4
# or directly:
RECOMPILE=1 ./submit_path_jobs.sh --gp --ngpus 1,2,4
```

## Parity gates (vs devices=1, double only)

| Mode   | \|ΔE\| max | max \|ΔF\| | cosine min |
|--------|------------|------------|------------|
| double | 1e-8       | 1e-6       | 1 − 1e-12  |

Results: `results/gp_round/ngpu{N}/parity.json`  
Reports: `results/gp_round/{SUMMARY,MULTIGPU_REPORT,RESULTS}.md`
