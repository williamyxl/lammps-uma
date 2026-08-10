#!/bin/bash
#SBATCH --job-name=nvalch
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nvalchemi_path/logs/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nvalchemi_path/logs/%x-%j.out

# One nvalchemi matrix cell.
#   sbatch --gpus-per-node=N --job-name=nvalch-<sys>-n<N> \
#          --export=ALL,SYS=nacl6,NG=N run_matrix_cell.slurm
#
# 1 GPU  -> direct python (integrator runs in-process)
# 2/4 GPU-> torchrun, DomainParallel spatial domain decomposition
set -uo pipefail
EX=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nvalchemi_path
SYS=${SYS:?set SYS=nacl6|water888}
NG=${NG:?set NG=1|2|4}

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate nvalchemi312

echo "JOBID=${SLURM_JOB_ID} nvalchemi sys=${SYS} ngpu=${NG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "${EX}"
if [[ "${NG}" == "1" ]]; then
  python run_nvalchemi.py --sys "${SYS}" --ngpu 1
else
  # Rendezvous on a free port; single node so localhost is fine.
  torchrun --nproc_per_node="${NG}" --nnodes=1 \
    --rdzv-backend=c10d --rdzv-endpoint="localhost:$((29500 + RANDOM % 1000))" \
    run_nvalchemi.py --sys "${SYS}" --ngpu "${NG}"
fi
echo "CELL_RC=$? sys=${SYS} ngpu=${NG}"
