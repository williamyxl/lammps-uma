#!/usr/bin/env python3
"""Binary-search driver for the NaCl NxNxN 4-GPU capacity ceiling.

Protocol (per user): probe N=8, then N=16.
  - 16 OK   -> double the bracket (16..32) and keep going
  - 16 OOM  -> bisect (8,16) -> 12, then bisect the failing half
Terminates when hi == lo + 1: lo is the largest working N, hi the smallest OOM.

The driver only submits and reads cell.json / parity.json; it never guesses a
result. Run it repeatedly (or from the poller) — it is idempotent and resumes
from whatever cells are already on disk.

  python nsweep_driver.py --status     # show the ladder so far
  python nsweep_driver.py --next       # submit the next N the search needs
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

EX = Path(__file__).resolve().parent
RESULTS = EX / "results"
STATE = EX / "nsweep_state.json"


def load_cells() -> dict[int, dict]:
    cells: dict[int, dict] = {}
    for f in sorted(RESULTS.glob("n*_*/cell.json")):
        try:
            c = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        n = int(c.get("n", -1))
        if n < 0:
            continue
        p = f.parent / "parity.json"
        if p.is_file():
            try:
                c["parity_rec"] = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        # Prefer a successful run if the same N was retried.
        if n not in cells or (c.get("functional") and not cells[n].get("functional")):
            cells[n] = c
    return cells


def ok(c: dict) -> bool:
    return bool(c.get("functional")) and not c.get("oom")


def next_n(cells: dict[int, dict], lo_seed: int, hi_seed: int,
           cap: int) -> tuple[int | None, str]:
    """Return the next N to test and why."""
    done = {n: ok(c) for n, c in cells.items()}
    if not done:
        return lo_seed, "seed probe"
    good = sorted(n for n, v in done.items() if v)
    bad = sorted(n for n, v in done.items() if not v)
    lo = max(good) if good else None
    hi = min(bad) if bad else None

    if lo is None:
        # Even the seed failed: walk down toward a known-good size.
        smallest = min(done)
        cand = max(6, smallest // 2)
        return (cand, "seed failed; halving down") if cand not in done else (None, "no working N found")
    if hi is None:
        # Everything passes so far: double the bracket.
        cand = min(cap, lo * 2)
        if cand == lo or cand in done:
            return None, f"ceiling >= {lo} (cap {cap} reached)"
        return cand, f"all pass up to {lo}; doubling"
    if hi <= lo + 1:
        return None, f"converged: max working N={lo}, first OOM N={hi}"
    mid = (lo + hi) // 2
    if mid in done:
        return None, f"converged: max working N={lo}, first OOM N={hi}"
    return mid, f"bisect ({lo},{hi})"


# Measured single-GPU ceiling on an A100 40GB: N=6 (1728 atoms) fits, N=8
# (4096) OOMs in the traced export at 37.10/39.49 GiB. Any cell above this must
# run multi-GPU, where the MP path shards edges across ranks.
SINGLE_GPU_MAX_N = 6


def submit(n: int, nsteps: int, dep: str | None, devices: int = 4) -> str:
    if devices == 1 and n > SINGLE_GPU_MAX_N:
        raise SystemExit(
            f"refusing N={n} on 1 GPU: single-GPU max is N={SINGLE_GPU_MAX_N} "
            f"on A100 40GB (measured OOM at N=8)")
    cmd = ["sbatch", "--parsable",
           f"--job-name=nsw-n{n}",
           f"--export=ALL,NVAL={n},NSTEPS={nsteps},UMA_DEVICES={devices}"]
    if dep:
        cmd.append(f"--dependency=afterany:{dep}")
    cmd.append(str(EX / "run_nsweep_cell.slurm"))
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout.strip().split(";")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=8, help="seed probe")
    ap.add_argument("--hi", type=int, default=16, help="second probe")
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--next", action="store_true", help="submit next N")
    ap.add_argument("--dep", default=None, help="afterany dependency jobid")
    a = ap.parse_args()

    cells = load_cells()
    if a.status or not a.next:
        print(f"{'N':>4} {'natoms':>8} {'status':>14} {'oom':>5} {'nvt_ms':>9} "
              f"{'T_K':>6} {'VRAM_GiB':>16} {'use%':>6} {'parity':>14}")
        for n in sorted(cells):
            c = cells[n]
            pr = c.get("parity_rec", {})
            ms = c.get("nvt_pair_ms_per_step")
            T = c.get("final_T_K")
            pk = c.get("vram_peak_MiB_max")
            tot = c.get("vram_total_MiB")
            vram = f"{pk / 1024:.1f}/{tot / 1024:.0f}" if pk and tot else "-"
            uf = c.get("vram_util_frac")
            print(f"{n:>4} {c.get('natoms', 0):>8} {str(c.get('status')):>14} "
                  f"{str(c.get('oom')):>5} {(f'{ms:.1f}' if ms else '-'):>9} "
                  f"{(f'{T:.0f}' if T else '-'):>6} {vram:>16} "
                  f"{(f'{100 * uf:.0f}' if uf else '-'):>6} "
                  f"{str(pr.get('parity', '-')):>14}")
        good = [n for n, c in cells.items() if ok(c)]
        bad = [n for n, c in cells.items() if not ok(c)]
        if good:
            print(f"\nmax working N = {max(good)} ({8 * max(good) ** 3} atoms)")
        if bad:
            print(f"smallest failing N = {min(bad)}")
        par = [n for n, c in cells.items()
               if c.get("parity_rec", {}).get("parity") == "PASS"]
        if par:
            print(f"max parity-verified N = {max(par)} ({8 * max(par) ** 3} atoms)")
        # VRAM headroom on the largest passing N tells us whether the ceiling
        # is near, independently of waiting for the next OOM.
        if good:
            top = cells[max(good)]
            pk, tot = top.get("vram_peak_MiB_max"), top.get("vram_total_MiB")
            if pk and tot:
                print(f"peak VRAM at N={max(good)}: {pk / 1024:.1f} GiB / "
                      f"{tot / 1024:.0f} GiB ({100 * pk / tot:.0f}%), "
                      f"headroom {(tot - pk) / 1024:.1f} GiB")
        nxt, why = next_n(cells, a.lo, a.hi, a.cap)
        print(f"\nnext: {nxt if nxt else 'DONE'}  ({why})")

    if a.next:
        # Honour the explicit two-probe seed before bisecting.
        if not cells:
            nxt, why = a.lo, "seed probe"
        elif len(cells) == 1 and a.lo in cells and ok(cells[a.lo]):
            nxt, why = a.hi, "second probe"
        else:
            nxt, why = next_n(cells, a.lo, a.hi, a.cap)
        if nxt is None:
            print(f"DONE: {why}")
            return 0
        jid = submit(nxt, a.nsteps, a.dep)
        print(f"submitted N={nxt} job={jid} ({why})")
        STATE.write_text(json.dumps(
            {"last_submit_n": nxt, "job": jid, "reason": why}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
