#!/bin/bash
#SBATCH --job-name=nvalch-smoke
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=00:40:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nvalchemi_path/logs/smoke-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nvalchemi_path/logs/smoke-%j.out

# Validate the nvalchemi path end-to-end on one GPU before scaling to 2/4.
set -uo pipefail
EX=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nvalchemi_path
module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate nvalchemi312

echo "JOBID=${SLURM_JOB_ID} nvalchemi smoke"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import nvalchemi,torch;print('nvalchemi',nvalchemi.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())"

cd "${EX}"
python run_nvalchemi.py --sys nacl6 --ngpu 1 --nsteps 10
echo "SMOKE_RC=$?"
