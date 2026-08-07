#!/bin/bash
# DEPRECATED — use gp_round/run_ngpu4_uma_double.slurm and ..._uma_mixed.slurm
echo "ERROR: gp_round/run_ngpu4.slurm is deprecated (multi-path VRAM isolation)." >&2
echo "  Use: ./submit_path_jobs.sh --gp --ngpus 4" >&2
exit 2
