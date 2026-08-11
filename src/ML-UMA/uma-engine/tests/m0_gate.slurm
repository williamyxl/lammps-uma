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

# M0 GATE (plan sec 9): "8-rank LJ NVT reproduces the serial trajectory."
#
# Deliberately uses a NON-UMA pair style (lj/cut). The point is to prove the
# multi-node launch scaffold works -- MPI build, one GPU per rank,
# 2 nodes x 4 ranks -- BEFORE any UMA code is involved.
#
# NO KOKKOS. The product recipe is 'no Kokkos' (UMA_USE_KOKKOS=0), and plan
# sec 8 states Kokkos is not needed for multi-node correctness. It is
# introduced only at M6 for the halo exchange. Testing the scaffold with
# '-k on -sf kk' here would validate a configuration we do not ship.
# If this fails, nothing downstream can be trusted, and the failure is in the
# scaffold rather than in the potential.
set -uo pipefail
ROOT=/work/nvme/bfzx/xyan11/workdir/lammps-uma
T=${ROOT}/src/ML-UMA/uma-engine/tests
LMP=${ROOT}/build-uma-mn/lmp
WORK=${T}/m0_gate_work_${SLURM_JOB_ID}
mkdir -p "${WORK}"

module unload cudatoolkit 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.8
source /u/xyan11/miniforge3-x86_64/etc/profile.d/conda.sh
conda activate uma312

echo "JOBID=${SLURM_JOB_ID} M0 gate: nodes=${SLURM_JOB_NUM_NODES} ntasks=${SLURM_NTASKS}"
test -x "${LMP}" || { echo "M0_GATE_FAIL missing ${LMP} (run M0 build first)"; exit 1; }
"${LMP}" -h 2>&1 | grep -iE 'MPI v|MPI STUBS' | head -2

cd "${WORK}"
# Plain LJ melt: no UMA, no external potential file, fully reproducible.
cat > in.lj <<'EOF'
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 1.44 87287 loop geom
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    delay 0 every 20 check no
fix             1 all nvt temp 1.44 1.44 0.5
thermo          10
thermo_style    custom step temp pe etotal press
thermo_modify   format float %.17g
run             100
print "M0_FINAL_TEMP = $(temp:%.17g)"
print "M0_FINAL_PE   = $(pe:%.17g)"
EOF

echo "=== serial reference (1 rank) ==="
"${LMP}" -in in.lj -log log.serial 2>&1 | tail -3
echo "=== 8 ranks, 2 nodes, one GPU per rank (no Kokkos) ==="
srun --ntasks=8 --gpus-per-task=1 --cpu-bind=cores \
  "${LMP}" -in in.lj -log log.mpi8 2>&1 | tail -3

python3 - <<'PY'
import re, json, pathlib
def grab(p):
    t = pathlib.Path(p).read_text(errors="ignore")
    out = {}
    for k, pat in (("temp", r"M0_FINAL_TEMP = (\S+)"),
                   ("pe",   r"M0_FINAL_PE   = (\S+)")):
        m = re.findall(pat, t)
        if m:
            out[k] = float(m[-1])
    return out
s, m = grab("log.serial"), grab("log.mpi8")
rec = {"serial": s, "mpi8": m}
ok = bool(s and m)
if ok:
    for k in ("temp", "pe"):
        if k in s and k in m:
            d = abs(s[k] - m[k]) / max(abs(s[k]), 1e-30)
            rec[f"rel_diff_{k}"] = d
            # LJ MD is chaotic; 100 steps of a different domain decomposition
            # will not be bit-identical. Require agreement to ~1e-6 relative,
            # which is far tighter than any decomposition bug would allow.
            ok = ok and d <= 1e-6
rec["pass"] = ok
pathlib.Path("m0_gate_result.json").write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
print("M0 GATE:", "PASS" if ok else "FAIL")
PY
echo "M0_GATE_RC=$?"
