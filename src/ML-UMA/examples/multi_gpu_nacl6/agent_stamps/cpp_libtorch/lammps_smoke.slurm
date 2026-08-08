#!/bin/bash
#SBATCH --job-name=uma-cpp-lmp
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/lammps_smoke-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/examples/multi_gpu_nacl6/agent_stamps/cpp_libtorch/lammps_smoke-%j.out

# Phase 3: LAMMPS end-to-end (single MPI rank)
#   lmp -k on g N -sf kk + pair_style uma/kk precision double devices N
# Override: sbatch --gres=gpu:4 --export=ALL,NGPUS=4,UMA_DEVICES=4,...
set -euo pipefail

ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
EX=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6
ENG=${ROOT}/src/ML-UMA/uma-engine
STAMP=${EX}/agent_stamps/cpp_libtorch
ART=${ENG}/artifacts/uma-s-1p2-omat-f64

export NGPUS="${NGPUS:-2}"
export UMA_DEVICES="${UMA_DEVICES:-${NGPUS}}"
export ONLY_PATHS=uma_double
export RECOMPILE="${RECOMPILE:-1}"
export MERGE_RESULTS=1
export SKIP_DIST_DESTROY=1
export UMA_FORBID_RAY_GP=1
export N_TIMING="${N_TIMING:-3}"
# Sibling of results/ngpu1 so run_multigpu can load devices=1 baseline.
export RESULTS_DIR="${RESULTS_DIR:-${EX}/results/ngpu${NGPUS}}"
export RESULTS_PARENT="${EX}/results"

# C++ LibTorch MP product path (not Python/Ray).
unset UMA_PYTHON_GP_WORKER || true
unset UMA_ALLOW_RAY_GP || true
unset UMA_SKIP_FORCE_GP_REDUCE || true
unset UMA_GRAD_ENERGY_SCALE || true
# Force-green regime (engine defaults ON; do not force identity bwd).
export UMA_ALLREDUCE_WITH_GRAD_BWD="${UMA_ALLREDUCE_WITH_GRAD_BWD:-1}"
export UMA_MP_NATOMS="${UMA_MP_NATOMS:-1728}"
export UMA_STRUCTURE_NATOMS="${UMA_STRUCTURE_NATOMS:-1728}"
export UMA_LIBTORCH_MP_WORKER="${UMA_LIBTORCH_MP_WORKER:-${ENG}/build-cpp-mp/uma_libtorch_mp_worker}"
export UMA_MP_LOG_DIR="${STAMP}/lammps_worker_logs_w${UMA_DEVICES}_${SLURM_JOB_ID:-manual}"
mkdir -p "${UMA_MP_LOG_DIR}" "${RESULTS_DIR}" "${STAMP}/lammps_gate_w${UMA_DEVICES}_${SLURM_JOB_ID:-manual}"

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8 cmake/3.31.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=$SLURM_JOB_ID host=$(hostname) NGPUS=$NGPUS UMA_DEVICES=$UMA_DEVICES"
echo "UMA_MP_NATOMS=$UMA_MP_NATOMS WORKER=$UMA_LIBTORCH_MP_WORKER"
echo "RESULTS_DIR=$RESULTS_DIR RECOMPILE=$RECOMPILE"
ls -lh "${ART}"/model_mp_w${UMA_DEVICES}_n1728_r*.pt
nvidia-smi -L || true

if [[ "${RECOMPILE}" == "1" ]]; then
  echo "=== RECOMPILE=1 build_lammps_uma.sh ==="
  bash "${ROOT}/scripts/build_lammps_uma.sh"
fi
export LMP_UMA="${ROOT}/build-uma/lmp"
test -x "${LMP_UMA}"
test -x "${UMA_LIBTORCH_MP_WORKER}"

echo "=== libuma_engine.a MP objects ==="
ar t "${ROOT}/build-uma/uma-engine/libuma_engine.a" | grep -E 'libtorch_mp|peer_context' || {
  echo "ERROR: libuma_engine.a missing libtorch_mp/peer_context after rebuild"
  exit 1
}
nm -C "${LMP_UMA}" | grep -c 'LibtorchMpRuntime' | awk '{print "LibtorchMpRuntime_syms="$1; if($1<1) exit 1}'

export FIXED_STRUCTURE="${EX}/structures/nacl6_rattle_fixed.extxyz"
export UMA_KK_LAUNCH="lmp -k on g ${NGPUS} -sf kk; pair_style uma/kk ... devices ${UMA_DEVICES}"
export PYTHONUNBUFFERED=1
export USE_SLURM_TIMING=1

cd "${EX}"
_t0=$(date +%s.%N)
set +e
python "${EX}/run_multigpu.py"
_rc=$?
set -e
_t1=$(date +%s.%N)
_wall=$(awk -v a="${_t0}" -v b="${_t1}" 'BEGIN { printf "%.3f", b-a }')
echo "run_multigpu rc=${_rc} wall_s=${_wall}"

if [[ ${_rc} -ne 0 ]]; then
  echo "=== worker logs (tail) ==="
  for f in "${UMA_MP_LOG_DIR}"/worker_r*.log; do
    [[ -f "$f" ]] || continue
    echo "----- $f -----"; tail -60 "$f"
  done
  exit ${_rc}
fi

if [[ -f "${EX}/stamp_slurm_timing.py" ]]; then
  python "${EX}/stamp_slurm_timing.py" \
    --results-dir "${RESULTS_DIR}" \
    --wall-s "${_wall}" \
    --n-timing "${N_TIMING}" \
    --paths "${ONLY_PATHS}" || true
fi

export STAMP EX WALL_S="${_wall}"
GATE_DIR="${STAMP}/lammps_gate_w${UMA_DEVICES}_${SLURM_JOB_ID}"
mkdir -p "${GATE_DIR}"
python - <<'PY'
import json, os, sys
from pathlib import Path
import numpy as np

ex = Path(os.environ["EX"])
results = Path(os.environ["RESULTS_DIR"])
stamp = Path(os.environ["STAMP"])
devices = int(os.environ["UMA_DEVICES"])
gate_dir = stamp / f"lammps_gate_w{devices}_{os.environ['SLURM_JOB_ID']}"

parity = json.loads((results / "parity.json").read_text())
forces = np.load(results / "forces.npz")
row = next(r for r in parity["rows"] if r.get("key") == "uma_double")
e = float(row["energy_eV"])
f = np.asarray(forces["forces_uma_double"], dtype=np.float64)

base = ex / "results" / "ngpu1"
brow = next(r for r in json.loads((base / "parity.json").read_text())["rows"]
            if r.get("key") == "uma_double")
e1 = float(brow["energy_eV"])
f1 = np.asarray(np.load(base / "forces.npz")["forces_uma_double"], dtype=np.float64)

dE = abs(e - e1)
dF = float(np.max(np.abs(f - f1)))
dF_atom = float(np.max(np.linalg.norm(f - f1, axis=1)))

ase_path = ex / "results" / "gp_round" / "oracle_ase_fp64_w1.json"
dE_ase = None
ase_e = None
if ase_path.is_file():
    ase = json.loads(ase_path.read_text())
    ase_e = float(ase.get("energy_eV") or ase.get("E") or ase.get("energy"))
    dE_ase = abs(e - ase_e)

# Match engine Phase-2 gates (loose vs ASE; tight vs d1 for E+F).
ok_e = dE <= 1e-8
ok_f = dF <= 1e-6
ms = row.get("ms_per_eval") or row.get("ms_per_step")
timing = None
tpath = results / "timing_slurm.json"
if tpath.is_file():
    timing = json.loads(tpath.read_text())

print(f"LAMMPS_GATE devices={devices} E={e:.12f} E_d1={e1:.12f} dE_d1={dE:.3e}")
if dE_ase is not None:
    print(f"LAMMPS_GATE dE_ase={dE_ase:.3e} ASE={ase_e:.12f}")
print(f"LAMMPS_GATE max|dF|={dF:.6e} max|dF_atom|={dF_atom:.6e}")
print(f"LAMMPS_GATE ms_per_eval={ms} wall_s={os.environ.get('WALL_S')}")
print("E_GREEN" if ok_e else "E_NOT_GREEN")
print("F_GREEN" if ok_f else "F_NOT_GREEN")

# Confirm GP label in log if present
gp_label = None
logp = results / "work" / "uma_double" / "log.sp"
if logp.is_file():
    for line in logp.read_text().splitlines():
        if "devices=" in line and "gp=" in line:
            gp_label = line.strip()
            break
print("LOG:", gp_label)

gate = {
    "phase": "3_lammps",
    "devices": devices,
    "E": e,
    "E_d1": e1,
    "dE_d1": dE,
    "dE_ase": dE_ase,
    "max_dF": dF,
    "max_dF_atom": dF_atom,
    "E_green": ok_e,
    "F_green": ok_f,
    "ms_per_eval": ms,
    "wall_s": float(os.environ.get("WALL_S") or "nan"),
    "timing_slurm": timing,
    "gp_log": gp_label,
    "results_dir": str(results),
    "jobid": os.environ.get("SLURM_JOB_ID"),
}
(gate_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
(results / "lammps_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
print("wrote", gate_dir / "gate.json")
sys.exit(0 if (ok_e and ok_f) else 2)
PY

echo "LAMMPS_SMOKE_OK job=$SLURM_JOB_ID devices=$UMA_DEVICES wall_s=${_wall}"
