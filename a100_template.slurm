#SBATCH --job-name=uma-a100
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Template shell. Full 4-path parity (export+build+run):
#   sbatch src/ML-UMA/examples/delta_parity/run_parity.slurm

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312
