#!/bin/bash
#SBATCH --job-name=w17-exp
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/w17_export-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf_campaign/w17_export-%j.out

# W17 unblock: re-export MP + devices=1 traced with fixed Wigner gamma=0
# (trace_patch eliminates aten::rand for CUDA graph capture).
set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART_SRC=${ENG}/artifacts/uma-s-1p2-omat-f64-fast
ART=${ENG}/artifacts/uma-s-1p2-omat-f64-fast-cgraph
CKPT=${UMA_CHECKPOINT:-/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt}
NACL_XYZ=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz
WATER_EX=${ROOT}/src/ML-UMA/examples/water888
WATER_DATA=${WATER_EX}/water_nvt_300K_atomic_metal.data
WATER_XYZ=${WATER_EX}/water_nvt_300K.extxyz

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID:-none} W17 fixed-gamma cgraph export -> ${ART}"
mkdir -p "${ART}"
cp -n "${ART_SRC}/metadata.json" "${ART}/" || cp "${ART_SRC}/metadata.json" "${ART}/"

python - <<PY
from ase.io import read, write
from pathlib import Path
p = Path("${WATER_XYZ}")
if not p.is_file():
    a = read("${WATER_DATA}", format="lammps-data", atom_style="atomic", units="metal")
    a.pbc = True
    write(str(p), a)
    print("wrote", p, "natoms=", len(a))
else:
    print("have", p)
PY

export_world() {
  local W="$1"
  local ATOMS="$2"
  local TAG="$3"
  echo "=== W17 export_mp world=${W} atoms=${ATOMS} (${TAG}) fixed_gamma ==="
  for R in $(seq 0 $((W - 1))); do
    CUDA_VISIBLE_DEVICES=$R PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" \
      python "${ENG}/python/export_mp_artifact.py" \
        --checkpoint "${CKPT}" \
        --output "${ART}" \
        --world "${W}" \
        --ranks "${R}" \
        --dtype float64 \
        --task omat \
        --device cuda \
        --atoms "${ATOMS}" \
        --execution-mode umas_fast_pytorch \
        --merge-mole
  done
}

# NaCl6 MP @2/@4 then water @2/@4
export_world 2 "${NACL_XYZ}" nacl1728
export_world 4 "${NACL_XYZ}" nacl1728
export_world 2 "${WATER_XYZ}" water648
export_world 4 "${WATER_XYZ}" water648

# devices=1 NaCl traced (same fixed-gamma patches via w15 script / trace_patch)
PYTHONPATH="${ENG}/python:${PYTHONPATH:-}" python "${ENG}/python/w15_export_traced_fast.py" \
  --checkpoint "${CKPT}" \
  --output "${ART}" \
  --atoms "${NACL_XYZ}" \
  --dtype float64 \
  --task omat \
  --device cuda \
  --out-name model_traced.pt \
  --backup-existing

# Verify no aten::rand in a representative shard + traced
python - <<PY
import torch
import zipfile
from pathlib import Path
art = Path("${ART}")
job = "${SLURM_JOB_ID:-none}"
paths = [
  art / "model_mp_w2_n1728_r0.pt",
  art / "model_traced.pt",
]
for p in paths:
  assert p.is_file(), p
  with zipfile.ZipFile(p) as z:
    hits = 0
    for n in z.namelist():
      b = z.read(n)
      hits += b.count(b"aten::rand")
    print(p.name, "aten::rand raw_hits", hits)
    # Allow 0 only; rand_emb strings do not contain aten::rand
    assert hits == 0, f"{p.name} still contains aten::rand (hits={hits})"
m = torch.jit.load(str(art / "model_traced.pt"), map_location="cpu")
g = str(m.inlined_graph)
assert "aten::rand" not in g, "inlined graph has aten::rand"
print("VERIFY_OK no aten::rand in model_traced / MP zip")
print(f"W17_EXPORT_OK job={job} art={art}")
PY
