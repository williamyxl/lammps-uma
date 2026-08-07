# gp_round dry-run checklist (TEST-PREP)

Do **not** submit heavy jobs until WRITE lands `pair_style uma/kk ... devices N` and
`build-uma/lmp` is rebuilt.

## Pre-submit (login node)

```bash
cd /work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6
./gp_round/rebuild_and_submit.sh   # syntax + geometry + pair_style line check
```

- [ ] `.write_agent_done.json` exists with `rebuild_required: true` (or run rebuild manually)
- [ ] `build-uma/lmp` rebuilt after WRITE C++ changes
- [ ] `python -m py_compile run_multigpu.py parity_gates.py collect_results.py`
- [ ] `pair_style uma/kk precision double devices 2` accepted by `lmp -h` or smoke `run 0`

## Submit order

1. **ngpu1** — establishes `devices=1` baseline (`ONLY_PATHS=uma_double,uma_mixed`)
2. **ngpu2** — parity gate vs ngpu1 (`devices=2`)
3. **ngpu4** — only after ngpu2 gates green

```bash
./gp_round/rebuild_and_submit.sh --submit
# optional 4-GPU after 2-GPU green:
./gp_round/rebuild_and_submit.sh --submit --ngpu4
```

## Parity gates (vs devices=1, same precision)

| Mode   | \|ΔE\| max | max \|ΔF\| | cosine min |
|--------|------------|------------|------------|
| double | 1e-8       | 1e-6       | 1 − 1e-12  |
| mixed  | 1e-4       | 1e-5       | 1 − 1e-10  |

Results: `results/gp_round/ngpu{N}/parity.json`  
Merge: `results/gp_round/SUMMARY.json`, `MULTIGPU_REPORT.md`

## SLURM contract

- `--account=bbpl-delta-gpu --partition=gpuA100x4`
- `--ntasks=1 --gpus-per-node=N`
- `module unload cudatoolkit`; `module load cuda/12.8`
- `conda activate uma312`
- `RECOMPILE=1` default in gp_round
