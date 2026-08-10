# NaCl N×N×N max-size sweep — 4×A100, FP64, NVT 300 K

Finds the largest NaCl `N×N×N` rocksalt supercell (`8·N³` atoms) that runs on
**4 A100 GPUs** under the current W8nk product recipe, with two separate
answers reported:

| Number | Meaning |
|--------|---------|
| **max parity-verified N** | largest N that runs **and** matches ASE FP64 ground truth (E/F) |
| **max functional N** | largest N that runs NVT 300 K without OOM (capacity ceiling) |

These differ because the ground-truth oracle is **single-GPU** and OOMs well
before the 4-GPU path does. Above the oracle limit the capacity ceiling is
still measured, but is explicitly **not** parity-verified
(`parity="NO_ORACLE_OOM"`). Do not quote the capacity number as validated.

## Recipe (matches product)

```text
pair_style uma precision double devices 4
UMA_USE_KOKKOS=0 · NCCL · FP64 · umas_fast_pytorch + merge_mole
```

## Gates per cell

1. **Functional** — NVT block completes, final T finite and in (100, 900) K.
2. **Force sanity** — `|Σ F|` < 1e-6 eV/Å. A periodic cell must have zero net
   force; a large residual means the multi-GPU reduction dropped a shard.
3. **Capacity** — no CUDA OOM in LAMMPS stdout or any worker log.
4. **Parity** — vs ASE FP64 `umas_fast_pytorch`+`merge_mole`:
   `|ΔE| ≤ 1e-6 eV` and `max|ΔF| ≤ 1e-5 eV/Å`.

## Search

Binary search: probe **N=8**, then **N=16**. If 16 survives, double the bracket
(→32); if it OOMs, bisect (8,16)→12 and continue. Converges in ~5–7 jobs
instead of one job per N. Verified against simulated ceilings 7…40.

## Why one job per N

MP TorchScript **bakes partition offsets per atom count**
(`model_mp_w4_n{natoms}_r*.pt`), so each N needs its own ~5 min 4-rank export.
Artifacts land in `artifacts/nacl{N}/`, never in the product artifact dir.

## Usage

```bash
python nsweep_driver.py --status     # ladder so far + next N
python nsweep_driver.py --next       # submit the next N the search needs
bash   nsweep_poll.sh &              # auto-advance the bisect
```

`run_nsweep_cell.slurm` never rebuilds (`RECOMPILE=0`) so it cannot race the
W17c wave, which owns `build-uma` with `RECOMPILE=1`.

OOM exits 0 by design: it is a measurement, not a broken job.

## Files

| File | Role |
|------|------|
| `build_structure.py` | frozen NaCl N³ + MB velocities @300 K (N=6 reproduces `nacl6_rattle_fixed` to ~5e-9 Å) |
| `run_nsweep_cell.slurm` | one cell: structure → per-N export → LAMMPS NVT |
| `run_cell.py` | runs LAMMPS, classifies OOM, writes `cell.json` |
| `parity_cell.py` | ASE FP64 single-GPU oracle → `parity.json` |
| `nsweep_driver.py` | bisect driver (`--status` / `--next`) |
| `nsweep_poll.sh` | auto-advance poller |
