#!/bin/bash
# DEPRECATED — use gp_round/run_ngpu1_uma_double.slurm and ..._uma_mixed.slurm
echo "ERROR: gp_round/run_ngpu1.slurm is deprecated (multi-path VRAM isolation)." >&2
echo "  Use: ./submit_path_jobs.sh --gp" >&2
echo "    or: sbatch gp_round/run_ngpu1_uma_double.slurm" >&2
exit 2
