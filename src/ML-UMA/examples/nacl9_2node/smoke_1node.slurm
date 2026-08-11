#!/bin/bash
#SBATCH --job-name=n9-alch-smoke
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --time=00:40:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/logs/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/logs/%x-%j.out
# Cheap 1-node/2-rank rehearsal of the SAME srun+DistributedManager+DomainParallel
# path used by the 2-node job. Validates device handling, partition(), halo
# exchange and NoseHoover-under-DD before spending a 2-node allocation.
set -uo pipefail
EX=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node
module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate nvalchemi312
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29800 + RANDOM % 200))
export OMP_NUM_THREADS=8
echo "JOBID=${SLURM_JOB_ID} smoke nodes=1 ntasks=${SLURM_NTASKS}"
cd "${EX}"
export NACL_TAG=${NACL_TAG:-nacl8rep}
export ALCHEMI_SP_ATOM_CAP=${ALCHEMI_SP_ATOM_CAP:-4096}
N_FROM_TAG=${NACL_TAG#nacl}; N_FROM_TAG=${N_FROM_TAG%rep}
python build_nacl9_replicated.py --n "${N_FROM_TAG}"
srun --cpu-bind=cores python run_alchemi_nacl9.py --ngpu "${SLURM_NTASKS}" --nsteps 10 --strategy graph_partition
echo "SMOKE_RC=$?"
