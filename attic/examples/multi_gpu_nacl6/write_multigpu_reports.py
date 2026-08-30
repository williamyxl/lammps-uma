#!/usr/bin/env python3
"""Write MULTIGPU_REPORT.md from results/SUMMARY.json (or collect first).

Usage:
  python write_multigpu_reports.py
  python write_multigpu_reports.py /path/to/SUMMARY.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from parity_gates import PARITY_THRESHOLDS

EX = Path(__file__).resolve().parent
RESULTS_PARENT = Path(
    os.environ.get("RESULTS_PARENT", str(EX / "results"))
).expanduser().resolve()
DEFAULT_SUMMARY = RESULTS_PARENT / "SUMMARY.json"
OUT_MD = RESULTS_PARENT / "MULTIGPU_REPORT.md"


def _fmt_e(x, prec: int = 10) -> str:
    if x is None:
        return "—"
    return f"{float(x):.{prec}f}"


def _fmt_sci(x) -> str:
    if x is None:
        return "—"
    x = float(x)
    if abs(x) == 0.0:
        return "0"
    return f"{x:.3e}"


def _fmt_ms(x) -> str:
    if x is None:
        return "—"
    x = float(x)
    if x >= 1000:
        return f"{x / 1000:.2f} s"
    return f"{x:.1f}"


def ensure_summary(summary_path: Path) -> dict:
    if not summary_path.is_file():
        # Try collect
        collect = EX / "collect_results.py"
        if collect.is_file():
            import subprocess

            rc = subprocess.call([sys.executable, str(collect)])
            if rc != 0 or not summary_path.is_file():
                raise SystemExit(
                    f"ERROR: missing {summary_path}; collect_results.py failed (rc={rc})"
                )
        else:
            raise SystemExit(f"ERROR: missing {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_markdown(summary: dict, path: Path) -> None:
    rows = summary.get("rows") or []
    ngpus = sorted({int(r["ngpu"]) for r in rows})
    paths = []
    for r in rows:
        if r["path"] not in paths:
            paths.append(r["path"])

    lines: list[str] = []
    lines.append("# UMA multi-GPU NaCl 6×6×6 parity report")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- **System:** NaCl 6×6×6 rocksalt, {summary.get('natoms', 1728)} atoms")
    lines.append(f"- **Structure (immutable):** `{summary.get('structure', '')}`")
    lines.append(
        "- **Perturbation:** uniform_box δ=0.1 Å seed=0 "
        "(fixed extxyz — never re-rattle)"
    )
    lines.append(f"- **Reference:** {summary.get('reference', 'ASE FP64 @ ngpu1')}")
    lines.append(
        f"- **ASE@ngpu1 energy:** `{_fmt_e(summary.get('ase_ref_energy_eV_ngpu1'))}` eV"
    )
    lines.append(
        f"- **uma/kk launch:** `{summary.get('kokkos_launch', 'lmp -k on g N -sf kk')}`"
    )
    lines.append(
        f"- **uma pair_style:** `{summary.get('uma_pair_style', 'pair_style uma/kk ... devices N')}`"
    )
    lines.append(
        f"- **ASE/FC:** `{summary.get('fairchem_workers', 'workers=N')}`"
    )
    lines.append(f"- **Precision:** {summary.get('precision', 'FP64 for ASE + uma double')}")
    gs = summary.get("parity_gates_summary") or {}
    if gs.get("n_checked"):
        lines.append(
            f"- **Parity gates (uma vs devices=1):** "
            f"{gs.get('n_passed')}/{gs.get('n_checked')} passed "
            f"(all_passed={gs.get('all_passed')})"
        )
    lines.append("")
    lines.append(
        "Force metrics vs ASE are secondary oracle. **Primary GP gate:** "
        "uma/kk `devices=N` vs `devices=1` at same precision."
    )
    lines.append("")
    lines.append("### Parity thresholds (vs devices=1)")
    lines.append("")
    lines.append("| Mode | |ΔE| max | max |ΔF| | cosine min |")
    lines.append("|------|----------|------------|------------|")
    th_src = summary.get("parity_thresholds") or PARITY_THRESHOLDS
    for mode, th in th_src.items():
        lines.append(
            f"| {mode} | {_fmt_sci(th['abs_dE_max'])} | "
            f"{_fmt_sci(th['force_max_abs_max'])} | {th['cosine_min']:.2e} |"
        )
    lines.append("")

    lines.append("## uma vs devices=1 (graph-parallel gate)")
    lines.append("")
    lines.append(
        "| Path | ngpu | devices | |dE| vs d1 | max |ΔF| vs d1 | cosine vs d1 | gate |"
    )
    lines.append(
        "|------|------|---------|------------|----------------|--------------|------|"
    )
    for r in rows:
        if r.get("key") not in ("uma_double", "uma_mixed"):
            continue
        gate = r.get("parity_gate") or {}
        gate_s = "—"
        if gate.get("applicable"):
            gate_s = "PASS" if gate.get("passed") else "FAIL"
        de_d1 = "—" if int(r.get("ngpu", 1)) == 1 else _fmt_sci(r.get("abs_dE_vs_uma_d1"))
        lines.append(
            f"| {r.get('path')} | {r.get('ngpu')} | {r.get('uma_devices', '—')} | "
            f"{de_d1} | {_fmt_sci(r.get('force_max_abs_vs_uma_d1'))} | "
            f"{_fmt_e(r.get('cosine_vs_uma_d1'), 6) if r.get('cosine_vs_uma_d1') is not None else '—'} | "
            f"{gate_s} |"
        )
    lines.append("")

    lines.append("## Path × ngpu table (vs ASE)")
    lines.append("")
    lines.append(
        "| Path | ngpu | Energy (eV) | ms/eval | |dE| vs ASE@ngpu1 | "
        "Force MAE | Force RMSE | max abs dF_i | Cosine |"
    )
    lines.append(
        "|------|------|-------------|---------|------------------|"
        "----------|-----------|--------------|--------|"
    )
    for r in rows:
        is_ase_ref = r.get("path") == "ASE FP64" and int(r.get("ngpu", -1)) == 1
        de = "—" if is_ase_ref else _fmt_sci(r.get("abs_dE_vs_ase_ngpu1"))
        mae = "—" if is_ase_ref else _fmt_sci(r.get("force_mae"))
        rmse = "—" if is_ase_ref else _fmt_sci(r.get("force_rmse"))
        mx = "—" if is_ase_ref else _fmt_sci(r.get("force_max_abs"))
        cos = "—" if is_ase_ref else (
            _fmt_e(r.get("cosine"), 6) if r.get("cosine") is not None else "—"
        )
        lines.append(
            f"| {r.get('path')} | {r.get('ngpu')} | {_fmt_e(r.get('energy_eV'))} | "
            f"{_fmt_ms(r.get('ms_per_eval'))} | {de} | {mae} | {rmse} | {mx} | {cos} |"
        )
    lines.append("")

    lines.append("## Timing (ms/eval)")
    lines.append("")
    header = "| Path | " + " | ".join(f"ngpu{n}" for n in ngpus) + " |"
    sep = "|------|" + "|".join(["------"] * len(ngpus)) + "|"
    lines.append(header)
    lines.append(sep)
    for sp in paths:
        cells = []
        for n in ngpus:
            hit = next(
                (r for r in rows if r.get("path") == sp and int(r.get("ngpu")) == n),
                None,
            )
            cells.append(_fmt_ms(hit.get("ms_per_eval") if hit else None))
        lines.append(f"| {sp} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Energy |dE| vs ASE@ngpu1 (eV)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for sp in paths:
        cells = []
        for n in ngpus:
            hit = next(
                (r for r in rows if r.get("path") == sp and int(r.get("ngpu")) == n),
                None,
            )
            if hit is None:
                cells.append("—")
            elif sp == "ASE FP64" and n == 1:
                cells.append("—")
            else:
                cells.append(_fmt_sci(hit.get("abs_dE_vs_ase_ngpu1")))
        lines.append(f"| {sp} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Kokkos multi-GPU is **same-node** via `-k on g N`; SLURM uses "
        "`--ntasks=1` (no `srun -n N` / mpirun across GPUs)."
    )
    lines.append(
        "- uma/kk graph-parallel: `pair_style uma/kk precision <mode> devices N` "
        "shards UMA inference across GPUs (engine GP runtime)."
    )
    lines.append("- ASE and FairChem FC use FairChem `workers=N` in a single process.")
    lines.append(f"- Machine-readable merge: `{RESULTS_PARENT.name}/SUMMARY.json`.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    summary_path = Path(argv[0]) if argv else DEFAULT_SUMMARY
    summary = ensure_summary(summary_path)
    write_markdown(summary, OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
