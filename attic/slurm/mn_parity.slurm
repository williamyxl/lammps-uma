#!/bin/bash
#SBATCH --job-name=mn-parity
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --time=00:40:00
#SBATCH --output=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/uma-engine/tests/%x-%j.out
#SBATCH --error=/work/nvme/bfzx/xyan11/workdir/lammps-uma/src/ML-UMA/uma-engine/tests/%x-%j.out

# M3 GATE -- multi-rank UMA parity: 4 MPI ranks x 1 GPU each, NaCl 6x6x6
# (1728 atoms), single-point E, vs the devices=1 single-GPU reference.
#
#   reference (job 21011180, devices=1, traced model): E = -5830.923741338 eV
#
# N=6 on purpose: devices=1 loads model_traced.pt, and the traced export only
# fits <= N=6 on one A100 (N=8 exceeds it). This isolates the MPI TRANSPORT
# (Allgatherv + tag sort) against a known single-GPU number. Larger boxes and
# the w8 MP path come after M3 passes.
#
# This run uses FOUR MPI RANKS with devices=1 each -- a completely different
# transport (MPI_Allgatherv + tag sort, no fork, no /dev/shm payload) that must
# reproduce the same physics. Start on ONE node so only the transport changes;
# 2 nodes follows once this is clean, and needs a w8 re-export.
set -uo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
T=${ROOT}/src/ML-UMA/uma-engine/tests
LMP=${ROOT}/build-uma-mn/lmp
ART=${ROOT}/src/ML-UMA/uma-engine/artifacts/uma-s-1p2-omat-f64-fast
DATA=${ROOT}/src/ML-UMA/examples/multi_gpu_nacl6/structures/nacl6_nvt_300K_atomic_metal.data
WORK=${T}/mn_parity_work_${SLURM_JOB_ID}
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
export UMA_MP_NATOMS=1728
export UMA_STRUCTURE_NATOMS=1728

echo "JOBID=${SLURM_JOB_ID} mn-parity nodes=${SLURM_JOB_NUM_NODES} ntasks=${SLURM_NTASKS}"
test -x "${LMP}" || { echo "MN_FAIL missing ${LMP}"; exit 1; }
test -f "${DATA}" || { echo "MN_FAIL missing ${DATA}"; exit 1; }

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
thermo          1
thermo_style    custom step temp pe etotal
thermo_modify   norm no format float %.17g
run             0
print "MN_ENERGY = \$(pe:%.17g)"
EOF

echo "=== ${SLURM_NTASKS} MPI ranks, devices 1 each ==="
srun --mpi=pmix --ntasks="${SLURM_NTASKS}" --gpus-per-task=1 --cpu-bind=cores \
  "${LMP}" -in in.uma -log log.mn > out.mn 2>&1
echo "srun rc=$?"

python3 - <<'PY'
import json, pathlib, re
txt = pathlib.Path("out.mn").read_text(errors="ignore")
rec = {"gate": "multi-rank UMA parity vs 4-GPU reference"}
REF_E = -5830.923741338

grids = re.findall(r"(\d+) by (\d+) by (\d+) MPI processor grid", txt)
prod = [int(a)*int(b)*int(c) for a, b, c in grids]
rec["max_ranks_in_one_job"] = max(prod) if prod else 0
rec["single_job"] = rec["max_ranks_in_one_job"] > 1

binds = re.findall(r"uma: rank (\d+) -> cuda:(\d+) \(of (\d+) visible\)", txt)
rec["bind_lines"] = len(binds)
rec["distinct_ranks"] = len({b[0] for b in binds})

m = re.findall(r"MN_ENERGY = (\S+)", txt)
rec["energy_eV"] = float(m[-1]) if m else None
rec["reference_eV"] = REF_E
if rec["energy_eV"] is not None:
    rec["abs_dE"] = abs(rec["energy_eV"] - REF_E)
    rec["rel_dE"] = rec["abs_dE"] / abs(REF_E)
    # Transport change only: the arithmetic is identical, so agreement should
    # be at float64 round-off, not merely "close".
    rec["parity_pass"] = rec["abs_dE"] <= 1e-6
rec["errors"] = re.findall(r"ERROR[^\n]{0,100}", txt)[:3]
rec["pass"] = bool(rec.get("parity_pass") and rec["single_job"])
pathlib.Path("mn_parity_result.json").write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
print("\nMN PARITY:", "PASS" if rec["pass"] else "FAIL")
PY
echo "MN_PARITY_RC=$?"
