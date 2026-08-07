#!/usr/bin/env python3
"""Merge results/ngpu{1,2,4}/parity.json → results/SUMMARY.json + SUMMARY.md.

Tables: path × ngpu → E, ms, |dE| vs ASE@ngpu1, force metrics,
and uma vs devices=1 parity gates (graph-parallel campaign).

Environment:
  RESULTS_PARENT  base dir containing ngpu1/ ngpu2/ ngpu4/ (default: results/)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from parity_gates import PARITY_THRESHOLDS, check_gate, summarize_gates

EX = Path(__file__).resolve().parent
RESULTS_PARENT = Path(
    os.environ.get("RESULTS_PARENT", str(EX / "results"))
).expanduser().resolve()
NGPUS = (1, 2, 4)

PATH_ORDER = (
    "ASE FairChem FP64",
    "ASE FP64",
    "FairChem LAMMPS fix external",
    "FairChem FC",
    "uma/kk precision double",
    "uma/kk double",
    "uma/kk precision mixed",
    "uma/kk mixed",
)


def _short_path(path: str) -> str:
    p = path.lower()
    if "ase" in p:
        return "ASE FP64"
    if "fairchem" in p or "fix external" in p:
        return "FairChem FC"
    if "double" in p:
        return "uma/kk double"
    if "mixed" in p:
        return "uma/kk mixed"
    return path


def _load_rows(parity: dict) -> list[dict]:
    if isinstance(parity.get("rows"), list):
        return list(parity["rows"])
    systems = parity.get("systems") or {}
    if isinstance(systems, dict) and systems:
        # Prefer nacl6 / first system
        for key in ("nacl_n6_rattle", "nacl6", "nacl_6"):
            if key in systems and isinstance(systems[key].get("rows"), list):
                return list(systems[key]["rows"])
        first = next(iter(systems.values()))
        if isinstance(first, dict) and isinstance(first.get("rows"), list):
            return list(first["rows"])
    return []


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


def _path_sort_key(short: str) -> tuple[int, str]:
    order = ["ASE FP64", "FairChem FC", "uma/kk double", "uma/kk mixed"]
    try:
        return (order.index(short), short)
    except ValueError:
        return (len(order), short)


def main() -> int:
    per_ngpu: dict[int, dict] = {}
    missing: list[str] = []
    for n in NGPUS:
        p = RESULTS_PARENT / f"ngpu{n}" / "parity.json"
        if not p.is_file():
            missing.append(str(p))
            continue
        per_ngpu[n] = json.loads(p.read_text(encoding="utf-8"))

    if not per_ngpu:
        print("ERROR: no ngpu*/parity.json found under", RESULTS_PARENT, file=sys.stderr)
        for m in missing:
            print(f"  missing: {m}", file=sys.stderr)
        return 1

    # Index: short_path -> ngpu -> row
    indexed: dict[str, dict[int, dict]] = {}
    meta: dict = {"ngpus_present": sorted(per_ngpu), "missing": missing}

    for n, parity in per_ngpu.items():
        rows = _load_rows(parity)
        meta.setdefault("sources", {})[str(n)] = {
            "gpu": parity.get("gpu"),
            "checkpoint": parity.get("checkpoint"),
            "natoms": parity.get("natoms"),
            "ngpus": parity.get("ngpus", n),
        }
        for r in rows:
            sp = _short_path(str(r.get("path", "")))
            indexed.setdefault(sp, {})[n] = r

    # Reference: ASE @ ngpu1 energy
    ase_ref_e = None
    ase_ngpu1 = indexed.get("ASE FP64", {}).get(1)
    if ase_ngpu1 is not None:
        ase_ref_e = ase_ngpu1.get("energy_eV")

    table_rows: list[dict] = []
    for sp in sorted(indexed.keys(), key=_path_sort_key):
        for n in NGPUS:
            r = indexed[sp].get(n)
            if r is None:
                continue
            e = r.get("energy_eV")
            de_vs_ase_ngpu1 = None
            if ase_ref_e is not None and e is not None:
                de_vs_ase_ngpu1 = abs(float(e) - float(ase_ref_e))
            # Prefer in-file abs_dE when present and n==1 ASE self, else use vs ASE@1
            entry = {
                "path": sp,
                "key": r.get("key"),
                "ngpu": n,
                "uma_devices": r.get("uma_devices", parity.get("uma_devices", n)),
                "energy_eV": e,
                "ms_per_eval": r.get("ms_per_eval"),
                "abs_dE_vs_ase_ngpu1": de_vs_ase_ngpu1,
                "abs_dE_vs_ase_f64": r.get("abs_dE_vs_ase_f64"),
                "force_mae": r.get("force_mae"),
                "force_rmse": r.get("force_rmse"),
                "force_max_abs": r.get("force_max_abs"),
                "force_max_norm_per_atom": r.get("force_max_norm_per_atom"),
                "cosine": r.get("cosine"),
                "abs_dE_vs_uma_d1": r.get("abs_dE_vs_uma_d1"),
                "force_max_abs_vs_uma_d1": r.get("force_max_abs_vs_uma_d1"),
                "cosine_vs_uma_d1": r.get("cosine_vs_uma_d1"),
                "parity_gate": r.get("parity_gate"),
            }
            table_rows.append(entry)

    # Aggregate parity gates for uma paths @ ngpu>1
    gate_list = [
        r["parity_gate"]
        for r in table_rows
        if r.get("parity_gate") and r.get("ngpu", 1) > 1
        and r.get("key") in ("uma_double", "uma_mixed")
    ]
    gates_summary = summarize_gates(gate_list)

    summary = {
        "title": "NaCl 6×6×6 multi-GPU parity summary",
        "natoms": 1728,
        "structure": (
            "src/ML-UMA/examples/delta_parity/structures/nacl6_rattle_fixed.extxyz"
        ),
        "reference": "ASE FairChem FP64 @ ngpu=1",
        "uma_d1_reference": "uma/kk devices=1 @ ngpu=1 (same precision)",
        "ase_ref_energy_eV_ngpu1": ase_ref_e,
        "kokkos_launch": "lmp -k on g ${NGPUS} -sf kk (ntasks=1, no MPI multi-GPU)",
        "uma_pair_style": "pair_style uma/kk precision <mode> devices ${UMA_DEVICES}",
        "fairchem_workers": "workers=${NGPUS} in one process",
        "precision": "ASE + uma/kk double = FP64",
        "parity_thresholds": PARITY_THRESHOLDS,
        "parity_gates_summary": gates_summary,
        "results_parent": str(RESULTS_PARENT),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "rows": table_rows,
    }

    RESULTS_PARENT.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_PARENT / "SUMMARY.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")

    # Markdown
    lines: list[str] = []
    lines.append("# NaCl 6×6×6 multi-GPU parity summary")
    lines.append("")
    lines.append(
        "Fixed geometry `nacl6_rattle_fixed.extxyz` (1728 atoms, δ=0.1 Å seed=0). "
        "uma/kk uses Kokkos same-node multi-GPU (`lmp -k on g N -sf kk`, "
        "`pair_style uma/kk ... devices N`, `--ntasks=1`). "
        "ASE/FC use FairChem `workers=N` in one process. "
        "ASE + uma/kk double are FP64."
    )
    lines.append("")
    lines.append(f"- **Reference energy:** ASE FP64 @ ngpu1 = `{_fmt_e(ase_ref_e)}` eV")
    lines.append(f"- **uma d1 gate reference:** devices=1 @ ngpu1")
    lines.append(f"- **ngpus present:** {sorted(per_ngpu)}")
    gs = gates_summary
    if gs.get("n_checked"):
        lines.append(
            f"- **parity gates (uma vs d1):** {gs['n_passed']}/{gs['n_checked']} passed "
            f"(all_passed={gs['all_passed']})"
        )
    if missing:
        lines.append(f"- **missing:** {', '.join(missing)}")
    lines.append("")
    lines.append("### Thresholds (uma vs devices=1)")
    lines.append("")
    lines.append("| Mode | |ΔE| max | max |ΔF| | cosine min |")
    lines.append("|------|----------|------------|------------|")
    for mode, th in PARITY_THRESHOLDS.items():
        lines.append(
            f"| {mode} | {_fmt_sci(th['abs_dE_max'])} | "
            f"{_fmt_sci(th['force_max_abs_max'])} | {th['cosine_min']:.2e} |"
        )
    lines.append("")
    lines.append(
        "| Path | ngpu | devices | Energy (eV) | ms/eval | |dE| vs ASE@ngpu1 | "
        "|dE| vs uma@d1 | max |ΔF| vs d1 | cosine vs d1 | gate |"
    )
    lines.append(
        "|------|------|---------|-------------|---------|-----------------------|"
        "---------------|----------------|--------------|------|"
    )
    for r in table_rows:
        is_ase_ref = r["path"] == "ASE FP64" and r["ngpu"] == 1
        de = "—" if is_ase_ref else _fmt_sci(r.get("abs_dE_vs_ase_ngpu1"))
        de_d1 = "—" if r.get("ngpu") == 1 and r.get("key") in ("uma_double", "uma_mixed") else _fmt_sci(r.get("abs_dE_vs_uma_d1"))
        mx_d1 = _fmt_sci(r.get("force_max_abs_vs_uma_d1"))
        cos_d1 = _fmt_e(r.get("cosine_vs_uma_d1"), 6) if r.get("cosine_vs_uma_d1") is not None else "—"
        gate = r.get("parity_gate") or {}
        gate_s = "—"
        if gate.get("applicable"):
            gate_s = "PASS" if gate.get("passed") else "FAIL"
        lines.append(
            f"| {r['path']} | {r['ngpu']} | {r.get('uma_devices', '—')} | "
            f"{_fmt_e(r.get('energy_eV'))} | {_fmt_ms(r.get('ms_per_eval'))} | {de} | "
            f"{de_d1} | {mx_d1} | {cos_d1} | {gate_s} |"
        )
    lines.append("")
    lines.append("## Legacy force table (vs ASE)")
    lines.append("")
    lines.append(
        "| Path | ngpu | Energy (eV) | ms/eval | |dE| vs ASE@ngpu1 (eV) | "
        "Force MAE | Force RMSE | max abs dF_i | max norm dF_atom | Cosine |"
    )
    lines.append(
        "|------|------|-------------|---------|-----------------------|"
        "----------|-----------|--------------|------------------|--------|"
    )
    for r in table_rows:
        is_ase_ref = r["path"] == "ASE FP64" and r["ngpu"] == 1
        de = "—" if is_ase_ref else _fmt_sci(r.get("abs_dE_vs_ase_ngpu1"))
        mae = "—" if is_ase_ref else _fmt_sci(r.get("force_mae"))
        rmse = "—" if is_ase_ref else _fmt_sci(r.get("force_rmse"))
        mx = "—" if is_ase_ref else _fmt_sci(r.get("force_max_abs"))
        mn = "—" if is_ase_ref else _fmt_sci(r.get("force_max_norm_per_atom"))
        cos = "—" if is_ase_ref else (
            _fmt_e(r.get("cosine"), 6) if r.get("cosine") is not None else "—"
        )
        lines.append(
            f"| {r['path']} | {r['ngpu']} | {_fmt_e(r.get('energy_eV'))} | "
            f"{_fmt_ms(r.get('ms_per_eval'))} | {de} | {mae} | {rmse} | "
            f"{mx} | {mn} | {cos} |"
        )
    lines.append("")
    lines.append("## Timing matrix (ms/eval)")
    lines.append("")
    paths = sorted({r["path"] for r in table_rows}, key=_path_sort_key)
    lines.append("| Path | " + " | ".join(f"ngpu{n}" for n in NGPUS) + " |")
    lines.append("|------|" + "|".join(["------"] * len(NGPUS)) + "|")
    for sp in paths:
        cells = []
        for n in NGPUS:
            hit = next((r for r in table_rows if r["path"] == sp and r["ngpu"] == n), None)
            cells.append(_fmt_ms(hit.get("ms_per_eval") if hit else None))
        lines.append(f"| {sp} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Machine-readable: `SUMMARY.json`.")
    lines.append("")

    out_md = RESULTS_PARENT / "SUMMARY.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
