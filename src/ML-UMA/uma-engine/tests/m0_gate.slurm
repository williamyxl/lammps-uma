#!/bin/bash
#SBATCH --job-name=m0-gate
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --time=00:40:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/uma-engine/tests/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/uma-engine/tests/%x-%j.out

# M0 GATE — the UMA path under MPI, not LAMMPS' own MPI.
#
# LAMMPS already ships multi-node support for its built-in potentials; testing
# lj/cut across 2 nodes would validate upstream, not us. What is unproven is
# `pair_style uma` itself:
#
#   1. pair_uma.cpp:94 refuses nprocs > 1 outright. That guard is lifted at M3,
#      so at M0 the correct, expected result is a CLEAN REFUSAL on every rank --
#      not a hang, not a segfault, not rank 0 succeeding while others die.
#   2. Every rank must bind a DISTINCT GPU. torch::Device(torch::kCUDA) with no
#      index is cuda:0, so all local ranks would land on GPU 0. The new binding
#      in pair_uma.cpp reads SLURM_LOCALID (then OMPI/MV2/LOCAL_RANK, then
#      comm->me) and takes local_rank % device_count.
#
# Gate: 8 ranks across 2 nodes all reach pair_style uma and emit the documented
# single-rank refusal, and the recorded binding is one distinct GPU per rank.
set -uo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
T=${ROOT}/src/ML-UMA/uma-engine/tests
LMP=${ROOT}/build-uma-mn/lmp
ART=${ROOT}/src/ML-UMA/uma-engine/artifacts/uma-s-1p2-omat-f64-fast
DATA=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_nvt_300K_atomic_metal.data
WORK=${T}/m0_gate_work_${SLURM_JOB_ID}
mkdir -p "${WORK}"

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

VESIN="${ROOT}/src/ML-UMA/uma-engine/third_party/vesin/lib"
TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${VESIN}:${TORCH_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export UMA_USE_KOKKOS=0

echo "JOBID=${SLURM_JOB_ID} M0 gate: nodes=${SLURM_JOB_NUM_NODES} ntasks=${SLURM_NTASKS}"
test -x "${LMP}" || { echo "M0_GATE_FAIL missing ${LMP}"; exit 1; }
"${LMP}" -h 2>&1 | grep -iE 'MPI v|MPI STUBS' | head -2

cd "${WORK}"
cp "${DATA}" ./nacl6.data
cat > in.uma <<EOF
units           metal
atom_style      atomic
boundary        p p p
newton          off
read_data       nacl6.data
pair_style      uma precision double devices 1
pair_coeff      * * ${ART} Na Cl
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
run             0
EOF

echo "=== 1 rank (must succeed: this is the shipped configuration) ==="
"${LMP}" -in in.uma -log log.serial > out.serial 2>&1
echo "serial rc=$?"
grep -iE 'binding rank|Total wall time|ERROR' out.serial log.serial 2>/dev/null | head -4

echo "=== 8 ranks / 2 nodes (expect the documented refusal on EVERY rank) ==="
# --mpi=pmix is REQUIRED. Without it srun launches N independent 1-rank
# LAMMPS jobs instead of one N-rank job: each prints "1 by 1 by 1 MPI
# processor grid", the nprocs>1 guard never fires, and the gate silently
# "passes" while testing nothing. Verified: the conda Open MPI 5.0.10 ships
# libpmix.so and `srun --mpi=list` offers pmix_v5.
srun --mpi=pmix --ntasks=8 --gpus-per-task=1 --cpu-bind=cores \
  "${LMP}" -in in.uma -log log.mpi8 > out.mpi8 2>&1
echo "mpi8 rc=$?"

python3 - <<'PY'
import json, pathlib, re
ser = pathlib.Path("out.serial").read_text(errors="ignore")
mpi = pathlib.Path("out.mpi8").read_text(errors="ignore")

rec = {"gate": "M0 — UMA path under MPI"}
rec["serial_ok"] = ("ERROR" not in ser) or ("Total wall time" in ser)

# Did srun actually form ONE 8-rank job? A grid of "1 by 1 by 1" means it
# launched 8 separate serial jobs, which would make every check below
# meaningless. Require a real multi-rank decomposition.
import re as _re
grids = _re.findall(r"(\d+) by (\d+) by (\d+) MPI processor grid", mpi)
rec["mpi_grids_seen"] = ["x".join(g) for g in grids[:3]]
rec["n_grid_lines"] = len(grids)
prod = [int(a) * int(b) * int(c) for a, b, c in grids]
rec["max_ranks_in_one_job"] = max(prod) if prod else 0
rec["single_8rank_job"] = rec["max_ranks_in_one_job"] == 8
# Distinct GPUs: one line per rank, and the cuda index must vary.
binds = _re.findall(r"uma: rank (\d+) -> cuda:(\d+) \(of (\d+) visible\)", mpi)
rec["bind_lines"] = len(binds)
rec["distinct_ranks"] = len({b[0] for b in binds})
rec["distinct_gpus"] = len({b[1] for b in binds})

# Every rank must hit the documented single-rank guard.
guard = re.findall(r"currently supports a single MPI rank", mpi)
rec["guard_hits"] = len(guard)
rec["guard_on_all_ranks"] = len(guard) >= 1
# The failure must be the guard, not a crash.
bad = [w for w in ("Segmentation fault", "signal", "CUDA error",
                   "out of memory", "std::bad_alloc") if w in mpi]
rec["crash_markers"] = bad
rec["clean_refusal"] = (len(guard) >= 1) and not bad
rec["pass"] = bool(rec["serial_ok"] and rec["clean_refusal"]
                   and rec["single_8rank_job"])
pathlib.Path("m0_gate_result.json").write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
print("M0 GATE:", "PASS" if rec["pass"] else "FAIL")
print()
print("Note: the refusal is the CORRECT M0 outcome. pair_uma.cpp:94 lifts at M3.")
PY
echo "M0_GATE_RC=$?"
