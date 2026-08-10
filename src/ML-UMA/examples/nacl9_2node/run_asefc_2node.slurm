#!/bin/bash
#SBATCH --job-name=n9-asefc-2n
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/logs/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node/logs/%x-%j.out

# ASE FairChem FP64 across 2 nodes x 4 A100 = 8 workers.
#
# EXPECTED-RISK RUN. fairchem's workers>1 path is Ray-based and creates one
# STRICT_PACK placement group per node, so it needs a Ray cluster spanning both
# nodes -- started explicitly below (head on node 0, worker on node 1).
# This deliberately violates the campaign's "no Ray / 1 MPI rank" rule, so the
# resulting numbers are NOT comparable to the locked ASE bars; they are a
# multi-node capability probe only.
set -uo pipefail
EX=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/nacl9_2node
module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD=${NODES[0]}
HEAD_IP=$(getent hosts "$HEAD" | awk '{print $1}' | head -1)
PORT=$((6400 + RANDOM % 300))
echo "JOBID=${SLURM_JOB_ID} ASE FC 2-node head=${HEAD}(${HEAD_IP}):${PORT} nodes=${NODES[*]}"

cleanup() { srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
              bash -c 'ray stop --force >/dev/null 2>&1 || true' || true; }
trap cleanup EXIT

# Head node
srun --nodes=1 --ntasks=1 -w "$HEAD" \
  ray start --head --node-ip-address="$HEAD_IP" --port="$PORT" \
            --num-gpus=4 --block &
sleep 25

# Worker node(s)
for ((i=1; i<${#NODES[@]}; i++)); do
  srun --nodes=1 --ntasks=1 -w "${NODES[$i]}" \
    ray start --address="${HEAD_IP}:${PORT}" --num-gpus=4 --block &
done
sleep 25

export RAY_ADDRESS="${HEAD_IP}:${PORT}"
cd "${EX}"
python build_nacl9_replicated.py
python - <<'PY'
import ray, os
ray.init(address=os.environ["RAY_ADDRESS"])
r = ray.cluster_resources()
print("RAY_CLUSTER_RESOURCES:", {k: r.get(k) for k in ("CPU", "GPU")})
print("RAY_NODES:", len([n for n in ray.nodes() if n["Alive"]]))
PY

python run_asefc_nacl9.py --workers "${WORKERS:-8}"
echo "ASEFC_2NODE_RC=$?"
