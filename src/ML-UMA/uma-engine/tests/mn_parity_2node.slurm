#!/bin/bash
#SBATCH --job-name=mn-parity-2n
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

# M3 GATE (real target): NaCl 8x8x8 (4096 atoms) on 2 nodes x 4 A100 = 8 ranks,
# single-point E + per-atom F, vs the 4-GPU ground truth (job 21026029):
#   E = -13821.798173425354 eV, forces (4096,3), |F|max 0.536847
#
# Each rank runs pair_style uma with the w8 MP shard for its rank: every rank
# holds all atoms and 1/8 of the edges (edge-parallel), the same design as the
# 4-GPU reference but split 8 ways across 2 nodes. The multi-node compute()
# assembles the tag-ordered global system, predicts, and keeps owned forces.
set -uo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
T=${ROOT}/src/ML-UMA/uma-engine/tests
LMP=${ROOT}/build-uma-mn/lmp
ART=${ROOT}/src/ML-UMA/examples/nacl_nsweep/artifacts/nacl8
DATA=${ROOT}/src/ML-UMA/examples/nacl_nsweep/structures/nacl8_nvt_300K_atomic_metal.data
REF=${ROOT}/src/ML-UMA/examples/nacl_nsweep/results/n8_21026029/forces.npz
WORK=${T}/mn_parity2n_work_${SLURM_JOB_ID}
mkdir -p "${WORK}"

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

VESIN="${ROOT}/src/ML-UMA/uma-engine/third_party/vesin/lib"
TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${VESIN}:${TORCH_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export UMA_USE_KOKKOS=0 UMA_MP_NATOMS=4096 UMA_STRUCTURE_NATOMS=4096
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29700 + RANDOM % 300))

echo "JOBID=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} ntasks=${SLURM_NTASKS}"
test -x "${LMP}" || { echo "MN_FAIL missing ${LMP}"; exit 1; }
ls "${ART}"/model_mp_w8_n4096_r*.pt >/dev/null 2>&1 || { echo "MN_FAIL missing w8 shards"; exit 1; }

cd "${WORK}"
cp "${DATA}" ./nacl8.data
cat > in.uma <<EOF
units           metal
atom_style      atomic
boundary        p p p
newton          off
read_data       nacl8.data
pair_style      uma precision double devices 8
pair_coeff      * * ${ART} Na Cl
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
thermo          1
thermo_style    custom step temp pe
thermo_modify   norm no format float %.17g
dump            1 all custom 1 mn_forces.dump id type x y z fx fy fz
dump_modify     1 sort id format float %.17g
run             0
print "MN_ENERGY = \$(pe:%.17g)"
EOF

echo "=== 8 ranks / 2 nodes, devices 8 (w8 MP shards) ==="
srun --mpi=pmix --ntasks=8 --gpus-per-task=1 --cpu-bind=cores \
  "${LMP}" -in in.uma -log log.mn > out.mn 2>&1
echo "srun rc=$?"

python3 - <<PYEOF
import json, re, pathlib
import numpy as np
txt = pathlib.Path("out.mn").read_text(errors="ignore")
rec = {"gate": "M3 2-node parity, NaCl 8x8x8 vs 4-GPU ground truth"}
REF_E = -13821.798173425354

grids = re.findall(r"(\d+) by (\d+) by (\d+) MPI processor grid", txt)
prod = [int(a)*int(b)*int(c) for a,b,c in grids]
rec["max_ranks_in_one_job"] = max(prod) if prod else 0
rec["single_8rank_job"] = rec["max_ranks_in_one_job"] == 8

m = re.findall(r"MN_ENERGY = (\S+)", txt)
rec["energy_eV"] = float(m[-1]) if m else None
rec["reference_eV"] = REF_E
if rec["energy_eV"] is not None:
    rec["abs_dE"] = abs(rec["energy_eV"] - REF_E)
    rec["energy_pass"] = rec["abs_dE"] <= 1e-6

# per-atom force parity vs ground truth
dump = pathlib.Path("mn_forces.dump")
ref = pathlib.Path("${REF}")
if dump.is_file() and ref.is_file():
    lines = dump.read_text().splitlines()
    s = next(i for i,l in enumerate(lines) if l.startswith("ITEM: ATOMS")) + 1
    rows = [list(map(float, l.split()[5:8])) for l in lines[s:] if len(l.split())>=8]
    F = np.array(rows)
    F0 = np.load(str(ref))["forces"]
    if F.shape == F0.shape:
        mag = np.linalg.norm(F-F0, axis=1)
        rec["force_max_per_atom"] = float(mag.max())
        rec["force_mean_per_atom"] = float(mag.mean())
        rec["n_atoms_over_tol"] = int((mag > 1e-5).sum())
        rec["force_pass"] = rec["force_max_per_atom"] <= 1e-5
    else:
        rec["force_shape"] = [list(F.shape), list(F0.shape)]

rec["errors"] = re.findall(r"ERROR[^\n]{0,100}", txt)[:3]
rec["pass"] = bool(rec.get("single_8rank_job") and rec.get("energy_pass")
                   and rec.get("force_pass"))
pathlib.Path("mn_parity2n_result.json").write_text(json.dumps(rec, indent=2)+"\n")
print(json.dumps(rec, indent=2))
print("\nM3 2-NODE PARITY:", "PASS" if rec["pass"] else "FAIL")
PYEOF
echo "MN_PARITY_2N_RC=$?"
