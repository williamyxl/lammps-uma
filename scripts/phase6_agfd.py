#!/usr/bin/env python3
"""Phase-6 AG=FD check for the XCCL graph-parallel pair_style uma run.

For a given rundir (with data.nacl, in.sp, symbols.txt, positions.npy), displace
one coordinate by +/- eps, re-run LAMMPS (same launch: mpiexec -n W + tiles),
read step-0 PotEng, and compare central-difference force to the autograd force
from forces_step0.dump. Gate max|F_AG - F_FD| <= AG_FD_TOL over sampled atoms.

Driven by the PBS wrapper which passes the exact launch command via env
LMP_LAUNCH (a shell command template that runs `<in>` in cwd and writes
log.lammps). We reuse the already-generated in.sp (run 0 + dump).

Env:
  DIR (rundir), W, FD_EPS=1e-4, AG_FD_TOL=1e-5, FD_SAMPLE=100
  LMP_LAUNCH  : shell command run in DIR that executes LAMMPS on in_fd.lmp and
                writes log_fd.lammps (must reference $INFILE, $LOGFILE)
"""
from __future__ import annotations
import os, re, subprocess, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uma_gates  # P1.5: single source of truth for tolerances


def pe0(log):
    txt = Path(log).read_text()
    m = re.search(r"^\s*Step\b.*PotEng.*$", txt, re.M)
    if not m:
        return None
    hdr = m.group(0).split(); pe = hdr.index("PotEng")
    for ln in txt[m.end():].splitlines():
        s = ln.split()
        if len(s) >= len(hdr) and s and s[0] == "0":
            return float(s[pe])
    return None


def read_forces(path, nat):
    lines = Path(path).read_text().splitlines()
    f = np.zeros((nat, 3))
    for i, ln in enumerate(lines):
        if ln.startswith("ITEM: ATOMS"):
            cols = ln.split()[2:]; ix = {c: k for k, c in enumerate(cols)}
            for j in range(nat):
                v = lines[i + 1 + j].split(); a = int(v[ix["id"]]) - 1
                f[a] = [float(v[ix["fx"]]), float(v[ix["fy"]]), float(v[ix["fz"]])]
            break
    return f


def write_data(path, syms, pos, cell):
    Z = {"Na": 11, "Cl": 17}; MASS = {"Na": 22.989769, "Cl": 35.453}
    L = cell[0, 0]
    lines = ["fd", "", f"{len(syms)} atoms", "2 atom types", "",
             f"0.0 {L:.16e} xlo xhi", f"0.0 {L:.16e} ylo yhi",
             f"0.0 {L:.16e} zlo zhi", "", "Masses", "",
             f"1 {MASS['Na']}", f"2 {MASS['Cl']}", "", "Atoms # atomic", ""]
    for i, s in enumerate(syms):
        t = 1 if s == "Na" else 2
        lines.append(f"{i+1} {t} {pos[i,0]:.16e} {pos[i,1]:.16e} {pos[i,2]:.16e}")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    d = Path(os.environ["DIR"])
    # P1.5: tolerances from the shared uma_gates table (env-overridable there).
    eps = uma_gates.fd_eps()
    tol = uma_gates.agfd_tol()
    nsamp = int(os.environ.get("FD_SAMPLE", str(uma_gates.min_sample())))
    launch = os.environ["LMP_LAUNCH"]
    syms = (d / "symbols.txt").read_text().split(); nat = len(syms)
    pos0 = np.load(d / "positions.npy"); cell = np.load(d / "cell.npy")
    f_ag = read_forces(d / "forces_step0.dump", nat)

    idx = sorted(set(np.linspace(0, nat - 1, min(nsamp, nat)).astype(int)) | {0, nat - 1})
    # FD input: same as in.sp but reads data_fd.nacl and no dump needed.
    base_in = (d / "in.sp").read_text().replace("data.nacl", "data_fd.nacl")
    base_in = re.sub(r"dump\s+f0.*\n", "", base_in)
    base_in = re.sub(r"dump_modify\s+f0.*\n", "", base_in)
    base_in = base_in.replace("undump          f0\n", "")
    (d / "in_fd.lmp").write_text(base_in)

    def energy_at(pos):
        write_data(d / "data_fd.nacl", syms, pos, cell)
        cmd = launch.replace("$INFILE", "in_fd.lmp").replace("$LOGFILE", "log_fd.lammps")
        # P1.1: check the LAMMPS return code — a crashed FD run must NOT be silently
        # treated as a missing-energy skip that leaves the gate able to PASS.
        rc = subprocess.run(cmd, shell=True, cwd=str(d), check=False).returncode
        if rc != 0:
            return None
        return pe0(d / "log_fd.lammps")

    # P1.1: minimum number of successful FD samples required to trust a PASS.
    # min_sample counts (atom,component) pairs; default = the atom sample count.
    min_sample = int(os.environ.get("MIN_SAMPLE", str(len(idx))))
    max_agfd = 0.0; sum_agfd = 0.0; cnt = 0; failed = 0
    for ia in idx:
        for ic in range(3):
            p = pos0.copy(); p[ia, ic] += eps; ep = energy_at(p)
            p = pos0.copy(); p[ia, ic] -= eps; em = energy_at(p)
            if ep is None or em is None:
                print(f"  FD run failed at atom {ia} comp {ic}", flush=True)
                failed += 1
                continue
            f_fd = -(ep - em) / (2 * eps)
            dd = abs(float(f_ag[ia, ic]) - f_fd)
            max_agfd = max(max_agfd, dd); sum_agfd += dd; cnt += 1
    mean = sum_agfd / cnt if cnt else float("nan")
    # P1.1: fail closed. Zero samples, too few samples, or ANY failed FD run means
    # the gate CANNOT report PASS (the old `ok = max_agfd <= tol` passed on cnt==0
    # because max_agfd stayed 0.0).
    if cnt == 0:
        print(f"[AG=FD W={os.environ.get('W','?')}] FAIL: 0 successful FD samples "
              f"(all {failed} runs failed)", flush=True)
        return 2
    if cnt < min_sample:
        print(f"[AG=FD W={os.environ.get('W','?')}] FAIL: only {cnt} successful FD "
              f"samples < MIN_SAMPLE={min_sample} ({failed} failed)", flush=True)
        return 2
    if failed > 0:
        print(f"[AG=FD W={os.environ.get('W','?')}] FAIL: {failed} FD run(s) failed "
              f"(fail-closed; got {cnt} good samples)", flush=True)
        return 2
    ok = max_agfd <= tol
    print(f"[AG=FD W={os.environ.get('W','?')}] sampled_atoms={len(idx)} "
          f"good_samples={cnt} eps={eps} "
          f"max|AG-FD|={max_agfd:.3e} mean={mean:.3e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
