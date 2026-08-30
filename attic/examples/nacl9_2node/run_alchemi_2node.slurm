#!/bin/bash
#SBATCH --job-name=n9-alchemi-2n
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/logs/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/logs/%x-%j.out

# ALCHEMI on 2 nodes x 4 A100 = 8 GPUs. Box selected by NACL_TAG (default nacl8rep, 4096 atoms).
# One task per GPU; DistributedManager picks up SLURM_PROCID / NPROCS / LOCALID
# and rendezvouses via MASTER_ADDR, so this spans nodes without torchrun.
set -uo pipefail
EX=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node
module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate nvalchemi312

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29600 + RANDOM % 400))
# NCCL over the inter-node fabric; without a sane interface list the halo
# exchange can hang on a management NIC.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export OMP_NUM_THREADS=8

echo "JOBID=${SLURM_JOB_ID} ALCHEMI 2-node nodes=${SLURM_JOB_NUM_NODES} ntasks=${SLURM_NTASKS}"
echo "MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
scontrol show hostnames "$SLURM_JOB_NODELIST"

cd "${EX}"
# NACL_TAG selects the box: nacl8rep (4096) or nacl9rep (5832).
export NACL_TAG=${NACL_TAG:-nacl8rep}
# No SP cap anymore: the runner does the single point through DomainParallel,
# so it is decomposed across all ranks like MD. The old undecomposed
# model(batch) that OOMed a full node is gone.
N_FROM_TAG=${NACL_TAG#nacl}; N_FROM_TAG=${N_FROM_TAG%rep}
python build_nacl9_replicated.py --n "${N_FROM_TAG}"

srun --cpu-bind=cores python run_alchemi_nacl9.py \
  --ngpu "${SLURM_NTASKS}" --nsteps "${NSTEPS:-10}" \
  --strategy "${ALCHEMI_STRATEGY:-graph_partition}"
echo "ALCHEMI_2NODE_RC=$?"
