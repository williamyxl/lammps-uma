#!/bin/bash
# DEPRECATED multi-path wrapper — VRAM isolation requires one path per job.
# Use:  ./submit_path_jobs.sh
#   or: sbatch run_ngpu1_ase.slurm | run_ngpu1_fc.slurm | run_ngpu1_uma_double.slurm | ...
echo "ERROR: run_ngpu1.slurm is deprecated (multi-path contaminates GPU VRAM)." >&2
echo "  Use ./submit_path_jobs.sh  or  sbatch run_ngpu1_<ase|fc|uma_double|uma_mixed>.slurm" >&2
exit 2
