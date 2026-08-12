#!/usr/bin/env bash
# Polaris per-rank GPU affinity wrapper for mpiexec (Cray PALS).
#
# Usage:  mpiexec -n <N> --ppn <ppn> ... polaris/gpu_affinity_polaris.sh <cmd> [args...]
#
# Polaris nodes have 4 A100. Cray PALS exposes the node-local rank as
# PMI_LOCAL_RANK. We pin each rank to one GPU via CUDA_VISIBLE_DEVICES so the
# process sees exactly one device (torch index 0).
#
# IMPORTANT for pair_uma.cpp: its per-rank binding reads
# SLURM_LOCALID / OMPI_COMM_WORLD_LOCAL_RANK / MV2_COMM_WORLD_LOCAL_RANK /
# LOCAL_RANK -- none of which PALS sets. We export LOCAL_RANK too so the pair
# style's own logic agrees with the mask (belt and suspenders). With the mask
# applied, torch device_count()==1 and idx = LOCAL_RANK % 1 = 0, which is the
# single visible GPU -- correct either way.

NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
LR="${PMI_LOCAL_RANK:-0}"
GPU=$(( LR % NGPUS_PER_NODE ))

export CUDA_VISIBLE_DEVICES="${GPU}"
export LOCAL_RANK="${LR}"

if [[ "${UMA_AFFINITY_VERBOSE:-1}" == "1" ]]; then
  echo "[affinity] PMI_RANK=${PMI_RANK:-?} PMI_LOCAL_RANK=${LR} -> CUDA_VISIBLE_DEVICES=${GPU} host=$(hostname)"
fi

exec "$@"
