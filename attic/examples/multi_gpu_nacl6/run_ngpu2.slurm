#!/bin/bash
# DEPRECATED multi-path wrapper — VRAM isolation requires one path per job.
echo "ERROR: run_ngpu2.slurm is deprecated (multi-path contaminates GPU VRAM)." >&2
echo "  Use ./submit_path_jobs.sh  or  sbatch run_ngpu2_<ase|fc|uma_double|uma_mixed>.slurm" >&2
exit 2
