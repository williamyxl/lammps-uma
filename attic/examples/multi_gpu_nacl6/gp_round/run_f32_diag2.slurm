#!/bin/bash
#SBATCH --job-name=uma-gp-f32-diag2
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=gp_round/%x-%j.out
#SBATCH --error=gp_round/%x-%j.out

set -euo pipefail
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312
export WANDB_MODE=disabled WANDB_DISABLED=true

ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/gp_round
STRUCT=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_rattle_fixed.extxyz
CKPT=/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt

# One subprocess per config so Ray/CUDA state cannot poison the next run.
run_one() {
  local dtype=$1 workers=$2
  python3 - "$dtype" "$workers" "$STRUCT" "$CKPT" <<'PY'
import sys, json
import numpy as np
import torch
from ase.io import read
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import inference_settings_default

dtype, workers, struct, ckpt = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
settings = inference_settings_default()
settings.base_precision_dtype = getattr(torch, dtype)
settings.external_graph_gen = workers <= 1
settings.activation_checkpointing = False
settings.execution_mode = "general"
pred = load_predict_unit(ckpt, device="cuda", inference_settings=settings, workers=workers)
calc = FAIRChemCalculator(pred, task_name="omat")
atoms = read(struct)
atoms.calc = calc
e = float(atoms.get_potential_energy())
f = np.asarray(atoms.get_forces(), dtype=np.float64)
print(f"RESULT dtype={dtype} workers={workers} E={e:.12f} fmax={float(np.abs(f).max()):.6e}", flush=True)
print("JSON "+json.dumps({"dtype":dtype,"workers":workers,"E":e,"fmax":float(np.abs(f).max())}), flush=True)
PY
}

declare -a LINES=()
for dtype in float64 float32; do
  for workers in 1 2; do
    echo "=== run $dtype workers=$workers ==="
    # capture RESULT line; allow failure without killing whole job
    if out=$(run_one "$dtype" "$workers" 2>&1); then
      echo "$out" | tail -40
      line=$(echo "$out" | grep '^JSON ' | tail -1 | sed 's/^JSON //')
      LINES+=("$line")
    else
      echo "$out" | tail -60
      LINES+=("{\"dtype\":\"$dtype\",\"workers\":$workers,\"error\":true}")
    fi
    # hard reset Ray between runs
    ray stop --force >/dev/null 2>&1 || true
    sleep 2
  done
done

python3 - <<PY
import json
from pathlib import Path
raw = '''${LINES[*]}'''
# rebuild from stdout file instead
PY

# Parse RESULT lines from this job's out via collecting above into file
printf '%s\n' "${LINES[@]}" > "$ROOT/f32_diag_raw.jsonl"
python3 - <<PY
import json
from pathlib import Path
p = Path("$ROOT/f32_diag_raw.jsonl")
rows = []
for line in p.read_text().splitlines():
    line=line.strip()
    if not line: continue
    rows.append(json.loads(line))
by = {f"{r.get('dtype')}_w{r.get('workers')}": r for r in rows}
print("=== summary ===")
for k,v in by.items():
    print(k, v)
def E(k):
    return by[k]["E"]
pairs = [("float64_w1","float64_w2"),("float32_w1","float32_w2"),("float32_w1","float64_w1"),("float32_w2","float64_w2")]
print("=== deltas ===")
for a,b in pairs:
    if "E" in by.get(a,{}) and "E" in by.get(b,{}):
        print(f"|E({a})-E({b})| = {abs(E(a)-E(b)):.6e}")
    else:
        print(f"{a} vs {b}: missing")
# vs traced mixed / double from campaign
traced_mixed = -5830.98193359375
traced_double = -5830.923720166723
for k in ("float32_w1","float32_w2"):
    if "E" in by.get(k,{}):
        print(f"|E({k})-traced_mixed| = {abs(E(k)-traced_mixed):.6e}")
        print(f"|E({k})-traced_double| = {abs(E(k)-traced_double):.6e}")
out = Path("$ROOT/f32_diag.json")
out.write_text(json.dumps({"rows":rows,"by":by}, indent=2)+"\n")
print("wrote", out)
PY
