#!/bin/bash
#SBATCH --job-name=n8-alchemi-1n
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/logs/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/logs/%x-%j.out

# ALCHEMI on 1 node x 4 A100 = 4 GPUs. Same box and same measurements as the
# 2-node job, so the pair isolates the inter-node effect: identical engine,
# identical system, only the node count differs.
#
# Direct comparison point: LibTorch UMA LAMMPS ran this exact box (NaCl 8x8x8,
# 4096 atoms) on 1 node x 4 A100 at 24.4/40 GiB (61%), 189.7 ms/step NVT.
set -uo pipefail
EX=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node
module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate nvalchemi312

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29900 + RANDOM % 90))
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=8

echo "JOBID=${SLURM_JOB_ID} ALCHEMI 1-node nodes=${SLURM_JOB_NUM_NODES} ntasks=${SLURM_NTASKS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "${EX}"
export NACL_TAG=${NACL_TAG:-nacl8rep}
# The standalone single point runs the FULL system on every rank (no
# decomposition); 4096 atoms needs ~32 GiB, which fits a 40 GiB A100.
N_FROM_TAG=${NACL_TAG#nacl}; N_FROM_TAG=${N_FROM_TAG%rep}
python build_nacl9_replicated.py --n "${N_FROM_TAG}"

srun --cpu-bind=cores python run_alchemi_nacl9.py \
  --ngpu "${SLURM_NTASKS}" --nsteps "${NSTEPS:-10}" \
  --strategy "${ALCHEMI_STRATEGY:-graph_partition}"
echo "ALCHEMI_1NODE_RC=$?"
