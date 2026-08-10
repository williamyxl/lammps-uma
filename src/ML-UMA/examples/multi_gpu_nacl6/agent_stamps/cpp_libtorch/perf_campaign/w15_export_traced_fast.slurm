#!/bin/bash
#SBATCH --job-name=w15-traced
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/w15_export-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/w15_export-%j.out

# W15: devices=1 model_traced.pt via MP-style path (NaCl atoms + umas_fast + merge_mole).
# Prior attempt used export_artifact.py (Fe sample / HF atom_refs) → FAILED.
set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART=${ENG}/artifacts/uma-s-1p2-omat-f64-fast
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
NACL_XYZ=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID:-none} W15 devices=1 traced fast (MP-style, NaCl atoms)"
test -f "${CKPT}"
test -f "${NACL_XYZ}"
mkdir -p "${ART}"

# Ensure a fallback general copy exists if we must roll back
if [[ -f "${ART}/model_traced.pt" && ! -f "${ART}/model_traced.pt.general_copy_bak_20260810T090821" ]]; then
  cp -a "${ART}/model_traced.pt" "${ART}/model_traced.pt.pre_w15_${SLURM_JOB_ID}"
fi

PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" python "${ENG}/python/w15_export_traced_fast.py" \
  --checkpoint "${CKPT}" \
  --output "${ART}" \
  --atoms "${NACL_XYZ}" \
  --dtype float64 \
  --task omat \
  --device cuda \
  --out-name model_traced.pt \
  --backup-existing

python - <<PY
import hashlib, json
from pathlib import Path
art = Path("${ART}")
tr = art / "model_traced.pt"
bak = art / "model_traced.pt.general_copy_bak_20260810T090821"
assert tr.is_file(), tr
h = hashlib.md5();
with open(tr, "rb") as f: h.update(f.read(1024*1024))
hb = None
if bak.is_file():
  hb = hashlib.md5();
  with open(bak, "rb") as f: hb.update(f.read(1024*1024))
  assert h.hexdigest() != hb.hexdigest(), "traced still identical to general bak"
meta = json.loads((art/"metadata.json").read_text())
inf = meta.get("inference_settings") or {}
print("traced_size", tr.stat().st_size, "md5_1MB", h.hexdigest())
print("execution_mode", inf.get("execution_mode"), "merge_mole", inf.get("merge_mole"))
assert inf.get("execution_mode") == "umas_fast_pytorch"
assert inf.get("merge_mole") is True
print("W15_EXPORT_OK job=${SLURM_JOB_ID} art=${ART}")
PY
