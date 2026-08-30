#!/bin/bash
# DEPRECATED — use gp_round/run_ngpu2_uma_double.slurm and ..._uma_mixed.slurm
echo "ERROR: gp_round/run_ngpu2.slurm is deprecated (multi-path VRAM isolation)." >&2
echo "  Use: ./submit_path_jobs.sh --gp" >&2
exit 2
