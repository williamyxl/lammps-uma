#!/usr/bin/env bash
# Multi-node edge-parallel run body: build-uma-mn/lmp under mpiexec, W ranks
# (1 GPU/rank) across nodes. Each rank = one MpiPeerPredictor; NCCL-over-MPI
# does the force all-reduce. NaCl 8x8x8 (4096) SP (E + per-atom F) + NVT timing.
#
# Env in: NRANKS (=total GPUs), PPN (ranks/node=4), TAG (e.g. r8), RESULTS dir.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/polaris/env_polaris.sh"

: "${NRANKS:?}"; : "${PPN:?}"; : "${TAG:?}"; : "${RESULTS:?}"
NSTEPS="${M3_NVT_STEPS:-10}"
SYSTEM="${M3_SYSTEM:-nacl4096}"
NAT="${M3_NATOMS:-4096}"
LMP="${ROOT}/build-uma-mn/lmp"
ENG="${ROOT}/src/ML-UMA/uma-engine"
ART="${UMA_ARTIFACT_DIR}"
P0="${ROOT}/src/ML-UMA/examples/polaris_p0"
M3="${ROOT}/src/ML-UMA/examples/polaris_m3"
export PYTHONPATH="${P0}:${ENG}/python:${PYTHONPATH:-}"

test -x "${LMP}" || { echo "M3_FAIL missing ${LMP} (run MG4)"; exit 1; }
test -f "${ART}/model_traced.pt" || { echo "M3_FAIL missing artifact"; exit 1; }
for R in $(seq 0 $((NRANKS-1))); do
  [[ -f "${ART}/model_mp_w${NRANKS}_n${NAT}_r${R}.pt" ]] || {
    echo "M3_FAIL missing shard model_mp_w${NRANKS}_n${NAT}_r${R}.pt"; exit 1; }
done

mkdir -p "${RESULTS}"
WORK="${RESULTS}/work_${TAG}"; mkdir -p "${WORK}"

DATA="$(python -c "import p0_common as c; print(c.SYSTEMS['${SYSTEM}']['data'])")"
ELEMS="$(python -c "import p0_common as c; print(' '.join(c.SYSTEMS['${SYSTEM}']['elements']))")"
DUMP="${WORK}/mn_forces.dump"
cat > "${WORK}/in.mn" <<EOF
units metal
atom_style atomic
boundary p p p
newton off
read_data ${DATA}
# one rank per GPU; multi-node edge-parallel triggers on nprocs>1
pair_style uma precision double devices 1
pair_coeff * * ${ART} ${ELEMS}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep 0.001
thermo 1
thermo_style custom step temp pe
thermo_modify norm no format float %.17g
dump 1 all custom 1 ${DUMP} id type x y z fx fy fz
dump_modify 1 sort id format float %.17g
run 0
undump 1
print "MN_SP_PE = \$(pe:%.17g)"
velocity all create 300.0 12345 mom yes rot yes dist gaussian
fix 1 all nvt temp 300.0 300.0 \$(100.0*dt)
run ${NSTEPS}
print "MN_NVT_PE = \$(pe:%.17g) T = \$(temp:%.17g)"
EOF

# per-rank shard selection + peer transport
export UMA_MP_NATOMS="${NAT}"
export UMA_STRUCTURE_NATOMS="${NAT}"
export UMA_PEER_TRANSPORT=nccl
export UMA_GPUS_PER_NODE="${PPN}"
export UMA_FORBID_RAY_GP=1
export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$((29700 + RANDOM % 300))"

echo "=== mpiexec: ${NRANKS} ranks / ${PPN} per node, edge-parallel w${NRANKS}, tag=${TAG} ==="
mpiexec -n "${NRANKS}" --ppn "${PPN}" \
  "${ROOT}/polaris/gpu_affinity_polaris.sh" \
  "${LMP}" -in "${WORK}/in.mn" -log "${WORK}/log.mn" > "${WORK}/out.mn" 2>&1
echo "mpiexec rc=$?"
tail -8 "${WORK}/out.mn" || true

# parse (reuse the gp parser: same output tokens minus prefix)
python - "${WORK}" "${TAG}" "${NAT}" "${NSTEPS}" "${RESULTS}" <<'PY'
import sys, re, json
from pathlib import Path
import numpy as np
work, tag, nat, nsteps, results = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
text = (work/"log.mn").read_text(errors="ignore") + "\n" + (work/"out.mn").read_text(errors="ignore")
e = None
for m in re.finditer(r"MN_SP_PE = (-?\d[\d.eE+-]*)", text): e = float(m.group(1))
grids = [int(a)*int(b)*int(c) for a,b,c in re.findall(r"(\d+) by (\d+) by (\d+) MPI processor grid", text)]
loop_re = re.compile(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps")
pair_re = re.compile(r"^Pair\s+\|\s+([0-9.eE+-]+)\s+\|", re.M)
blocks=[]
for m in loop_re.finditer(text):
    nxt=loop_re.search(text,m.end()); chunk=text[m.end():nxt.start() if nxt else len(text)]
    pm=pair_re.search(chunk); n=int(m.group(3)); ls=float(m.group(1)); ps=float(pm.group(1)) if pm else None
    blocks.append({"nsteps":n,"loop_ms":ls/n*1e3 if n else None,"pair_ms":ps/n*1e3 if (ps and n) else None})
nvt=blocks[-1] if blocks else {}
nvt_ms = nvt.get("pair_ms") or nvt.get("loop_ms")
dump=work/"mn_forces.dump"; f=None
if dump.is_file():
    L=dump.read_text().splitlines(); s=next(i for i,l in enumerate(L) if l.startswith("ITEM: ATOMS"))
    cols=L[s].split()[2:]; iid,ifx=cols.index("id"),cols.index("fx"); rows=[]
    for l in L[s+1:]:
        if l.startswith("ITEM:"): break
        p=l.split()
        if len(p)>=len(cols): rows.append((float(p[iid]),float(p[ifx]),float(p[ifx+1]),float(p[ifx+2])))
    a=np.array(rows); a=a[np.argsort(a[:,0])]; f=a[:,1:4]
rec={"system":"nacl4096","natoms":nat,"tag":tag,"energy_eV":e,"ranks":max(grids) if grids else 0,
     "nvt_steps":nsteps,"nvt_ms_per_step":nvt_ms,
     "force_abs_max":(float(np.abs(f).max()) if f is not None else None),
     "force_net":([float(x) for x in f.sum(0)] if f is not None else None)}
out=results/f"m3_{tag}.json"; out.write_text(json.dumps(rec,indent=2)+"\n")
if f is not None: np.savez(out.with_suffix(".npz"),forces=f,energy_eV=np.array(e if e is not None else np.nan))
print(json.dumps(rec,indent=2))
PY
echo "M3_MN_RUN_OK tag=${TAG} ranks=${NRANKS}"
