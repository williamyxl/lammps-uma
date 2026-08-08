#!/bin/bash
#SBATCH --job-name=uma-perf-p1
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf/perf_p1-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/perf/perf_p1-%j.out

# P1: CUDA IPC device collectives (UMA_PEER_TRANSPORT=cuda_ipc).
# Chain devices=1,2,4; E+F green; hard gate pair_ms(2)<pair_ms(1) and pair_ms(4)<pair_ms(2).
set -euo pipefail

ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
EX=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6
ENG=${ROOT}/src/ML-UMA/uma-engine
STAMP=${EX}/agent_stamps/cpp_libtorch
PERF=${STAMP}/perf
ART=${ENG}/artifacts/uma-s-1p2-omat-f64
mkdir -p "${PERF}"
export PERF_DIR="${PERF}"

export ONLY_PATHS=uma_double
export RECOMPILE="${RECOMPILE:-1}"
export MERGE_RESULTS=1
export SKIP_DIST_DESTROY=1
export UMA_FORBID_RAY_GP=1
export N_TIMING="${N_TIMING:-5}"
export UMA_ALLREDUCE_WITH_GRAD_BWD=1
export UMA_MP_NATOMS=1728
export UMA_STRUCTURE_NATOMS=1728
# P1 product transport
export UMA_PEER_TRANSPORT="${UMA_PEER_TRANSPORT:-cuda_ipc}"
# Honest pair ms: do not overwrite with SLURM wall/N_TIMING
export USE_SLURM_TIMING=0
unset UMA_CUDA_LAUNCH_BLOCKING || true
unset CUDA_LAUNCH_BLOCKING || true
unset UMA_PYTHON_GP_WORKER || true
unset UMA_ALLOW_RAY_GP || true
unset UMA_SKIP_FORCE_GP_REDUCE || true
unset UMA_GRAD_ENERGY_SCALE || true

export UMA_LIBTORCH_MP_WORKER="${ENG}/build-cpp-mp/uma_libtorch_mp_worker"
export FIXED_STRUCTURE="${EX}/structures/nacl6_rattle_fixed.extxyz"
export RESULTS_PARENT="${EX}/results"
export PYTHONUNBUFFERED=1

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "PERF_P1 JOBID=$SLURM_JOB_ID host=$(hostname) RECOMPILE=$RECOMPILE UMA_PEER_TRANSPORT=$UMA_PEER_TRANSPORT"
nvidia-smi -L || true

if [[ "${RECOMPILE}" == "1" ]]; then
  bash "${ROOT}/scripts/build_lammps_uma.sh"
  pushd "${ENG}/build-cpp-mp" >/dev/null
  cmake --build . -j 8 --target uma_libtorch_mp_worker
  popd >/dev/null
fi
export LMP_UMA="${ROOT}/build-uma/lmp"
test -x "${LMP_UMA}"
test -x "${UMA_LIBTORCH_MP_WORKER}"
export RECOMPILE=0

echo "=== worker CUDA IPC refs ==="
nm -C "${UMA_LIBTORCH_MP_WORKER}" 2>/dev/null | grep -i cudaIpc | head -10 || true

run_one() {
  local d="$1"
  export NGPUS="$d"
  export UMA_DEVICES="$d"
  export RESULTS_DIR="${EX}/results/perf_p1_ngpu${d}_${SLURM_JOB_ID}"
  export UMA_MP_LOG_DIR="${PERF}/worker_logs_p1_w${d}_${SLURM_JOB_ID}"
  export UMA_KK_LAUNCH="lmp -k on g ${d} -sf kk; pair_style uma/kk ... devices ${d}"
  mkdir -p "${RESULTS_DIR}" "${UMA_MP_LOG_DIR}"
  echo "=== PERF_P1 devices=${d} RESULTS_DIR=${RESULTS_DIR} transport=${UMA_PEER_TRANSPORT} ==="
  if [[ "$d" -gt 1 ]]; then
    ls -lh "${ART}"/model_mp_w${d}_n1728_r*.pt
  fi
  cd "${EX}"
  local t0 t1 wall rc
  t0=$(date +%s.%N)
  set +e
  python "${EX}/run_multigpu.py"
  rc=$?
  set -e
  t1=$(date +%s.%N)
  wall=$(awk -v a="${t0}" -v b="${t1}" 'BEGIN { printf "%.3f", b-a }')
  echo "run_multigpu devices=${d} rc=${rc} wall_s=${wall}"
  if [[ ${rc} -ne 0 ]]; then
    echo "FAIL devices=${d}"
    for f in "${UMA_MP_LOG_DIR}"/worker_r*.log; do
      [[ -f "$f" ]] || continue
      echo "----- $f -----"; tail -40 "$f"
    done
    exit ${rc}
  fi
  python - <<PY
import json, os, re
from pathlib import Path
import numpy as np

ex = Path("${EX}")
results = Path(os.environ["RESULTS_DIR"])
devices = int("${d}")
job = os.environ["SLURM_JOB_ID"]
perf = Path("${PERF}")
n_timing = int(os.environ.get("N_TIMING", "5"))

parity = json.loads((results / "parity.json").read_text())
row = next(r for r in parity["rows"] if r.get("key") == "uma_double")
e = float(row["energy_eV"])
f = np.asarray(np.load(results / "forces.npz")["forces_uma_double"], dtype=np.float64)

# Honest pair ms: prefer pair_section_s, then uma64 line in run.log, never SLURM wall.
ms = None
if row.get("pair_section_s") is not None:
    ms = float(row["pair_section_s"]) / max(n_timing, 1) * 1e3
log = results / "run.log"
if log.is_file():
    for line in log.read_text().splitlines():
        m = re.search(r"uma64 E=\S+\s+([0-9.]+)\s+ms", line)
        if m:
            ms = float(m.group(1))
# Fallback: parity ms only if it looks like pair-scale (< 5000 ms)
if ms is None:
    cand = row.get("ms_per_eval")
    if cand is not None and float(cand) < 5000:
        ms = float(cand)

base = ex / "results" / "ngpu1"
brow = next(r for r in json.loads((base / "parity.json").read_text())["rows"] if r.get("key") == "uma_double")
e1 = float(brow["energy_eV"])
f1 = np.asarray(np.load(base / "forces.npz")["forces_uma_double"], dtype=np.float64)
dE = abs(e - e1)
dF = float(np.max(np.abs(f - f1)))
ok_e, ok_f = dE <= 1e-8, dF <= 1e-6

ms_fwd = ms_bwd = ms_force = None
wlog = Path(os.environ["UMA_MP_LOG_DIR"]) / "worker_r0.log"
ipc_ok = None
if wlog.is_file():
    text = wlog.read_text()
    ipc_ok = "cuda_ipc ready" in text
    for line in text.splitlines():
        if "PERF_TICK" in line:
            m = re.search(r"ms_fwd=([0-9.]+).*ms_bwd=([0-9.]+).*ms_force_ar=([0-9.]+)", line)
            if m:
                ms_fwd, ms_bwd, ms_force = map(float, m.groups())

gate = {
    "phase": "P1_perf",
    "devices": devices,
    "jobid": job,
    "E": e,
    "dE_d1": dE,
    "max_dF": dF,
    "E_green": ok_e,
    "F_green": ok_f,
    "ms_per_eval": ms,
    "ms_source": "pair_section_or_uma64_log",
    "wall_s": float("${wall}"),
    "ms_fwd": ms_fwd,
    "ms_bwd": ms_bwd,
    "ms_force_ar": ms_force,
    "peer_transport": os.environ.get("UMA_PEER_TRANSPORT"),
    "cuda_ipc_ready_log": ipc_ok,
}
out = perf / f"gate_p1_w{devices}_{job}.json"
out.write_text(json.dumps(gate, indent=2) + "\n")
print(f"PERF_GATE devices={devices} ms={ms} dE_d1={dE:.3e} max|dF|={dF:.3e} E_green={ok_e} F_green={ok_f} ipc_log={ipc_ok}")
print("wrote", out)
if not (ok_e and ok_f):
    raise SystemExit(2)
if ms is None:
    raise SystemExit("missing honest pair ms")
PY
}

run_one 1
run_one 2
run_one 4

python - <<PY
import json, os
from pathlib import Path
perf = Path(os.environ.get("PERF_DIR", "${PERF}"))
job = os.environ["SLURM_JOB_ID"]
rows = {}
for d in (1, 2, 4):
    p = perf / f"gate_p1_w{d}_{job}.json"
    rows[d] = json.loads(p.read_text())
m1, m2, m4 = (float(rows[d]["ms_per_eval"]) for d in (1, 2, 4))
ok_scale = m2 < m1 and m4 < m2
summary = {
    "phase": "P1_perf",
    "jobid": job,
    "peer_transport": os.environ.get("UMA_PEER_TRANSPORT", "cuda_ipc"),
    "ms": {"1": m1, "2": m2, "4": m4},
    "dE_d1": {str(d): rows[d]["dE_d1"] for d in (1, 2, 4)},
    "max_dF": {str(d): rows[d]["max_dF"] for d in (1, 2, 4)},
    "E_F_green": all(rows[d]["E_green"] and rows[d]["F_green"] for d in (1, 2, 4)),
    "self_scale_green": bool(ok_scale),
    "hard_gate": "ms(2)<ms(1) and ms(4)<ms(2)",
    "p0_baseline_ms": {"1": 320.64, "2": 330.52, "4": 382.86},
    "note": "P1 = CUDA IPC device payloads; control still in shm",
}
(perf / f"summary_p1_{job}.json").write_text(json.dumps(summary, indent=2) + "\n")
print("PERF_SUMMARY", json.dumps(summary))
print("SELF_SCALE_GREEN" if ok_scale else "SELF_SCALE_NOT_GREEN")
if not summary["E_F_green"]:
    raise SystemExit(2)
# Soft: still report scale; hard-fail scale so campaign sees the gate clearly.
if not ok_scale:
    raise SystemExit(3)
PY

echo "PERF_P1_DONE job=$SLURM_JOB_ID"
