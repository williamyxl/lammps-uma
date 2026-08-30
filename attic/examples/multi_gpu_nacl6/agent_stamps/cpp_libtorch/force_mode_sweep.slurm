#!/bin/bash
#SBATCH --job-name=uma-cpp-fmodes
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=01:30:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/force_sweep-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/force_sweep-%j.out

# Sweep force/autograd regimes vs devices=1 on nacl64.
set -euo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
ENG=${ROOT}/src/ML-UMA/uma-engine
ART=${ENG}/artifacts/uma-s-1p2-omat-f64
STAMP=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch
STRUCT="${STRUCT:-${ENG}/artifacts/nacl64.txt}"

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export RECOMPILE="${RECOMPILE:-1}"

mkdir -p "${ENG}/build-cpp-mp"
cd "${ENG}/build-cpp-mp"
cmake .. -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" -DUMA_ENGINE_USE_CUDA=ON
cmake --build . -j "${SLURM_CPUS_PER_TASK:-8}"

export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${ENG}/third_party/vesin/lib:${LD_LIBRARY_PATH:-}"
export UMA_LIBTORCH_MP_WORKER="${ENG}/build-cpp-mp/uma_libtorch_mp_worker"
export PATH="${ENG}/build-cpp-mp:${PATH}"

WORKDIR="${STAMP}/force_sweep_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}"
F1="${WORKDIR}/f_d1.bin"

echo "=== devices=1 baseline ==="
"${ENG}/build-cpp-mp/uma_parity_cli" "${ART}" "$STRUCT" --devices 1 --write-forces "$F1" | tee "${WORKDIR}/d1.txt"

run_mode() {
  local name="$1"
  local skip="$2"   # 0/1
  local ar_bwd="$3" # 0/1
  local escale="$4" # float
  local dir="${WORKDIR}/${name}"
  mkdir -p "$dir"
  export UMA_MP_LOG_DIR="${dir}/logs"
  mkdir -p "${UMA_MP_LOG_DIR}"
  export UMA_SKIP_FORCE_GP_REDUCE="$skip"
  export UMA_ALLREDUCE_WITH_GRAD_BWD="$ar_bwd"
  export UMA_GRAD_ENERGY_SCALE="$escale"
  echo "=== MODE $name skipF=$skip arBwd=$ar_bwd escale=$escale ==="
  set +e
  "${ENG}/build-cpp-mp/uma_parity_cli" "${ART}" "$STRUCT" --devices 2 --write-forces "${dir}/f_d2.bin" | tee "${dir}/d2.txt"
  local rc=$?
  set -e
  python - <<PY
import re, struct, math, json
from pathlib import Path
wd = Path("${dir}")
base = Path("${WORKDIR}")
d1 = (base / "d1.txt").read_text()
d2 = (wd / "d2.txt").read_text()
e1 = float(re.search(r"energy=([-+0-9.eE]+)", d1).group(1))
e2 = float(re.search(r"energy=([-+0-9.eE]+)", d2).group(1))
n = int(re.search(r"n=(\d+)", d1).group(1))
f1 = struct.unpack(f"{3*n}d", (base / "f_d1.bin").read_bytes())
f2 = struct.unpack(f"{3*n}d", (wd / "f_d2.bin").read_bytes()) if (wd/"f_d2.bin").exists() else None
rec = {"mode": "${name}", "rc": ${rc}, "E_d1": e1, "E_d2": e2 if f2 else None, "dE": abs(e2-e1) if f2 else None}
if f2:
    max_df = max(abs(a-b) for a,b in zip(f1,f2))
    max_dfa = 0.0
    for i in range(n):
        dx,dy,dz = f1[3*i]-f2[3*i], f1[3*i+1]-f2[3*i+1], f1[3*i+2]-f2[3*i+2]
        max_dfa = max(max_dfa, math.sqrt(dx*dx+dy*dy+dz*dz))
    rec["max_dF"] = max_df
    rec["max_dF_atom"] = max_dfa
print("RESULT", json.dumps(rec))
(wd / "gate.json").write_text(json.dumps(rec, indent=2)+"\n")
open(base/"summary.jsonl","a").write(json.dumps(rec)+"\n")
PY
}

# name | skip_force_reduce | allreduce_bwd | energy_scale
run_mode sumF_idBwd_s1          0 0 1.0
run_mode skipF_idBwd_s1         1 0 1.0
run_mode sumF_arBwd_s1          0 1 1.0
run_mode skipF_arBwd_s1         1 1 1.0
run_mode sumF_idBwd_s0p5        0 0 0.5
run_mode skipF_idBwd_s0p5       1 0 0.5
run_mode sumF_arBwd_s0p5        0 1 0.5
run_mode skipF_arBwd_s0p5       1 1 0.5

echo "=== SUMMARY ==="
cat "${WORKDIR}/summary.jsonl"
python - <<PY
import json
from pathlib import Path
rows=[json.loads(l) for l in Path("${WORKDIR}/summary.jsonl").read_text().splitlines() if l.strip()]
rows=sorted(rows, key=lambda r: (r.get("max_dF") is None, r.get("max_dF") or 1e99))
print("best_by_max_dF:")
for r in rows[:5]:
    print(r)
best=rows[0]
ok = best.get("max_dF") is not None and best["max_dF"] <= 1e-6 and (best.get("dE") or 0) <= 1e-9
print("SWEEP_BEST", best["mode"], "max_dF", best.get("max_dF"), "F_GREEN" if ok else "F_NOT_GREEN")
PY
echo "SWEEP_DONE job=$SLURM_JOB_ID"
