#!/bin/bash
#SBATCH --job-name=uma-gp-rebuild-g2
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=gp_round/%x-%j.out
#SBATCH --error=gp_round/%x-%j.out

set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
EX=$ROOT/src/ML-UMA/examples/multi_gpu_nacl6
GP=$EX/gp_round

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "=== rebuild (avoid BUILD env collision) ==="
export BUILD_DIR=$ROOT/build-uma
# module may set BUILD=...
bash $ROOT/scripts/build_lammps_uma.sh

echo "=== smoke parity_cli devices=2 ==="
ART=$ROOT/src/ML-UMA/uma-engine/artifacts/uma-s-1p2-omat-f64
# minimal structure from ngpu1 data if available
$ROOT/build-uma/uma_parity_cli "$ART" --devices 2 || true

echo "=== run ngpu2 campaign ==="
export NGPUS=2
export RECOMPILE=0
export ONLY_PATHS=uma_double,uma_mixed
export UMA_DEVICES=2
source $GP/_run_gp_common.sh
