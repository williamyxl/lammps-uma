#!/usr/bin/env python3
"""Regenerate settings_docs/*.md from settings_tables.json.

Usage:
  python regenerate_settings_docs.py
  python regenerate_settings_docs.py --ingest-matrix   # merge matrix/*/parity.json probes

Keeps W7 (or latest DONE_*) as primary uma ufast cells; W8 force-race probes
are recorded under cell['probes'] and shown in notes when present.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

CAMP = Path(__file__).resolve().parent
DOCS = CAMP / "settings_docs"
TABLES = DOCS / "settings_tables.json"

META = {
    "gen": {
        "file": "SETTINGS_no_merge_no_fast.md",
        "title": "no merge, no fast (`general`, `merge_mole=False`)",
        "execution_mode": "general",
        "merge_mole": False,
        "artifact": "uma-s-1p2-omat-f64",
        "eref": "ASE `general`",
    },
    "gmerge": {
        "file": "SETTINGS_merge_no_fast.md",
        "title": "merge, no fast (`general`, `merge_mole=True`)",
        "execution_mode": "general",
        "merge_mole": True,
        "artifact": "uma-s-1p2-omat-f64-merge",
        "eref": "ASE `general`+`merge_mole`",
    },
    "ufast_nomole": {
        "file": "SETTINGS_fast_no_merge.md",
        "title": "fast, no merge (`umas_fast_pytorch`, `merge_mole=False`)",
        "execution_mode": "umas_fast_pytorch",
        "merge_mole": False,
        "artifact": "n/a (illegal)",
        "eref": "n/a",
        "illegal_banner": True,
    },
    "ufast": {
        "file": "SETTINGS_fast_and_merge.md",
        "title": "fast and merge (`umas_fast_pytorch`, `merge_mole=True`)",
        "execution_mode": "umas_fast_pytorch",
        "merge_mole": True,
        "artifact": "uma-s-1p2-omat-f64-fast",
        "eref": "ASE `umas_fast_pytorch`+`merge_mole`",
    },
}

SYSTEMS = {
    "nacl6": ("NaCl6 (1728 atoms)", False),
    "water888": ("water888 (648 atoms, NVT)", True),
}


def _fmt(v: Any, kind: str = "g") -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if kind == "E":
        return f"{float(v):.4f}"
    if kind == "ms":
        return f"{float(v):.4f}"
    if kind == "sci":
        return f"{float(v):.3e}"
    return str(v)


def _notes(cell: dict) -> str:
    parts: list[str] = []
    if cell.get("job"):
        parts.append(str(cell["job"]))
    if cell.get("note"):
        parts.append(str(cell["note"]))
    for p in cell.get("probes") or []:
        label = p.get("label", "probe")
        st = p.get("status", "")
        ms = p.get("ms")
        job = p.get("job")
        frag = f"{label}"
        if st:
            frag += f" {st}"
        if ms is not None:
            frag += f" ms={ms}"
        if job:
            frag += f" ({job})"
        if p.get("note"):
            frag += f": {p['note']}"
        parts.append(frag)
    return " ".join(parts)


def _row(path: str, ngpu: int, cell: dict) -> str:
    st = cell.get("status") or "—"
    if st.startswith("SKIP"):
        return (
            f"| {path} | {ngpu} | {st} | — | — | — | — | — | — | {_notes(cell)} |"
        )
    return (
        f"| {path} | {ngpu} | {st} | {_fmt(cell.get('E'), 'E')} | "
        f"{_fmt(cell.get('dE'), 'sci')} | {_fmt(cell.get('mae'), 'sci')} | "
        f"{_fmt(cell.get('max_abs'), 'sci')} | {_fmt(cell.get('max_norm_atom'), 'sci')} | "
        f"{_fmt(cell.get('ms'), 'ms')} | {_notes(cell)} |"
    )


def _timing_row(path: str, cells: dict) -> str:
    vals = []
    for ng in ("1", "2", "4"):
        c = cells.get(ng) or {}
        st = c.get("status") or ""
        if st.startswith("SKIP") or c.get("ms") is None:
            vals.append(st if st.startswith("SKIP") else "—")
        else:
            vals.append(_fmt(c.get("ms"), "ms"))
    return f"| {path} | {vals[0]} | {vals[1]} | {vals[2]} |"


def render_doc(tag: str, block: dict, stamp: str) -> str:
    meta = META[tag]
    lines: list[str] = []
    lines.append(f"# Settings: {meta['title']}")
    lines.append("")
    lines.append(f"**Stamp:** {stamp}  ")
    lines.append(
        f"**FairChem:** `execution_mode={meta['execution_mode']}`, "
        f"`merge_mole={meta['merge_mole']}`  "
    )
    lines.append(f"**uma artifact:** `{meta['artifact']}`  ")
    lines.append(f"**E/F reference:** {meta['eref']}  ")
    lines.append("")
    lines.append(
        "Paths: **ASE** FairChem FP64 · **FC** LAMMPS (FairChem fix external) · "
        "**uma** `uma/kk` precision double."
    )
    lines.append("GPUs: 1 / 2 / 4 (ASE/FC `workers=N`; uma `devices N`, 1 MPI).")
    lines.append("")
    if meta.get("illegal_banner"):
        lines.append(
            "> **All cells SKIP — illegal FairChem combination.** "
            "`umas_fast_pytorch` requires `merge_mole=True`. No measurements."
        )
        lines.append("")
    lines.append("### Metrics")
    lines.append("")
    lines.append("| Suite | Timing | Force columns |")
    lines.append("|-------|--------|---------------|")
    lines.append(
        "| NaCl6 | `ms_per_eval_python` (SP) | vs reference forces; "
        "`max‖ΔF‖_atom` = max per-atom ‖ΔF‖ |"
    )
    lines.append("| water888 | `nvt_pair_ms_per_step` (NVT Pair) | same |")
    lines.append("")
    lines.append(
        "`|ΔE|` and forces are vs the settings reference above "
        "(not vs a different settings row)."
    )
    lines.append("")

    for sys_key, (sys_title, _) in SYSTEMS.items():
        sys_block = block.get(sys_key) or {}
        lines.append(f"## {sys_title}")
        lines.append("")
        lines.append(
            "| path | ngpu | status | E (eV) | \\|ΔE\\| (eV) | force MAE | "
            "force max\\|Δ\\| | max‖ΔF‖/atom | time (ms) | notes |"
        )
        lines.append(
            "|------|-----:|--------|-------:|-----------:|----------:"
            "|---------------:|-------------:|----------:|-------|"
        )
        for path in ("ase", "fc", "uma"):
            path_block = sys_block.get(path) or {}
            for ng in (1, 2, 4):
                cell = path_block.get(str(ng)) or {
                    "status": "PENDING",
                    "note": "missing from settings_tables.json",
                }
                lines.append(_row(path, ng, cell))
        lines.append("")
        short = "NaCl6" if sys_key == "nacl6" else "water888"
        lines.append(f"### Timing summary ({short})")
        lines.append("")
        lines.append("| path | @1 | @2 | @4 |")
        lines.append("|------|---:|---:|---:|")
        for path in ("ase", "fc", "uma"):
            lines.append(_timing_row(path, sys_block.get(path) or {}))
        lines.append("")

    lines.append("## Legend")
    lines.append("")
    lines.append("- `SKIP_ILLEGAL` — FairChem rejects this settings combo")
    lines.append(
        "- `SKIP_KNOWN_CRASH` — FC+`merge_mole`+FP64 crashes in FairChem MOLE "
        "merge (not fixed here)"
    )
    lines.append("- `REUSED` / `REUSED_Tier0` — locked campaign baseline")
    lines.append(
        "- `INVALID_FORCE` probe — timing recorded but E/F not trusted "
        "(e.g. W8 NCCL stream race); primary row stays prior valid gate"
    )
    lines.append(
        "- Force self-parity for ASE vs its own oracle is ~1e-16 (numerical noise)"
    )
    lines.append("")
    lines.append("See also [`MATRIX.md`](../MATRIX.md), [`GLOSSARY.md`](../GLOSSARY.md).")
    lines.append("")
    return "\n".join(lines)


def render_readme(stamp: str) -> str:
    return "\n".join(
        [
            "# Settings documents (E / force / timing)",
            "",
            f"**Stamp:** {stamp}",
            "",
            "| Document | `execution_mode` | `merge_mole` |",
            "|----------|------------------|--------------|",
            "| [SETTINGS_no_merge_no_fast.md](SETTINGS_no_merge_no_fast.md) | `general` | False |",
            "| [SETTINGS_merge_no_fast.md](SETTINGS_merge_no_fast.md) | `general` | True |",
            "| [SETTINGS_fast_no_merge.md](SETTINGS_fast_no_merge.md) | `umas_fast_pytorch` | False (**illegal**) |",
            "| [SETTINGS_fast_and_merge.md](SETTINGS_fast_and_merge.md) | `umas_fast_pytorch` | True |",
            "",
            "Each doc: ASE · FC · uma/kk × NaCl6 · water888 × @1/@2/@4 — energy, "
            "per-atom force parity, timing.",
            "",
            "Regenerate: `python ../regenerate_settings_docs.py` "
            "(optionally `--ingest-matrix`).",
            "",
        ]
    )


def _append_probe(cell: dict, probe: dict) -> None:
    probes = cell.setdefault("probes", [])
    # de-dupe by job
    job = probe.get("job")
    if job and any(p.get("job") == job for p in probes):
        for i, p in enumerate(probes):
            if p.get("job") == job:
                probes[i] = {**p, **probe}
                return
    probes.append(probe)


def ingest_matrix(tables: dict) -> int:
    """Merge known matrix/*/parity.json into tables; W8 → probe only."""
    matrix = CAMP / "matrix"
    if not matrix.is_dir():
        return 0
    n = 0
    # dirname → (tag, system, path, ngpu, mode)
    # mode: primary | probe_invalid_force
    patterns = [
        (
            re.compile(r"^nacl6_ase_gmerge_ngpu(\d+)$"),
            "gmerge",
            "nacl6",
            "ase",
            "primary",
        ),
        (
            re.compile(r"^nacl6_ase_ufast_ngpu(\d+)$"),
            "ufast",
            "nacl6",
            "ase",
            "primary",
        ),
        (
            re.compile(r"^nacl6_uma_double_gen_ngpu(\d+)$"),
            "gen",
            "nacl6",
            "uma",
            "primary",
        ),
        (
            re.compile(r"^nacl6_uma_double_gmerge_ngpu(\d+)$"),
            "gmerge",
            "nacl6",
            "uma",
            "primary",
        ),
        (
            re.compile(r"^nacl6_uma_double_ufast_ngpu(\d+)$"),
            "ufast",
            "nacl6",
            "uma",
            "primary",
        ),
        (
            re.compile(r"^nacl6_uma_ufast_w8(?:fix)?_ngpu(\d+)$"),
            "ufast",
            "nacl6",
            "uma",
            "w8_auto",
        ),
        (
            re.compile(r"^water888_uma_ufast_w8(?:fix)?_ngpu(\d+)$"),
            "ufast",
            "water888",
            "uma",
            "w8_auto",
        ),
    ]
    for d in sorted(matrix.iterdir()):
        if not d.is_dir():
            continue
        parity_path = d / "parity.json"
        if not parity_path.is_file():
            continue
        matched = None
        for rx, tag, system, path, mode in patterns:
            m = rx.match(d.name)
            if m:
                matched = (tag, system, path, int(m.group(1)), mode)
                break
        if not matched:
            continue
        tag, system, path, ngpu, mode = matched
        j = json.loads(parity_path.read_text())
        rows = j.get("rows") or []
        if not rows:
            continue
        r = rows[0]
        ms = r.get("ms_per_eval_python")
        if system == "water888" and r.get("nvt_pair_ms_per_step") is not None:
            ms = r.get("nvt_pair_ms_per_step")
        job = str(r.get("slurm_job_id") or "")
        cell = (
            tables.setdefault(tag, {})
            .setdefault(system, {})
            .setdefault(path, {})
            .setdefault(str(ngpu), {})
        )
        if mode == "w8_auto":
            # Promote only when forces look sane vs merge ASE scale; else probe.
            absmax = None
            fz = d / "forces.npz"
            if fz.is_file():
                try:
                    import numpy as np

                    z = np.load(fz)
                    key = next(
                        (
                            k
                            for k in z.files
                            if k.startswith("forces_") or k == "forces"
                        ),
                        None,
                    )
                    if key:
                        absmax = float(np.nanmax(np.abs(z[key])))
                except Exception as e:  # noqa: BLE001
                    absmax = None
                    _append_probe(
                        cell,
                        {
                            "label": "W8",
                            "status": "INVALID_FORCE",
                            "ms": ms,
                            "job": job,
                            "E": r.get("energy_eV"),
                            "note": f"npz check failed: {e}",
                        },
                    )
                    n += 1
                    continue
            force_bad = absmax is None or absmax > 1.0e3
            if absmax is None:
                # incomplete cell — skip until forces.npz lands
                continue
            if force_bad:
                note = "NCCL dedicated-stream race; forces garbage — do not promote"
                note = f"forces absmax={absmax:.3e}; {note}"
                _append_probe(
                    cell,
                    {
                        "label": "W8",
                        "status": "INVALID_FORCE",
                        "ms": ms,
                        "job": job,
                        "E": r.get("energy_eV"),
                        "note": note,
                    },
                )
                n += 1
                continue
            # Valid W8: promote primary row
            mode = "primary_w8"
        # primary: fill missing timing / refine DONE cells without clobbering
        # richer force stats already present
        if ms is not None:
            cell["ms"] = ms
        if r.get("energy_eV") is not None:
            cell["E"] = r["energy_eV"]
        if r.get("abs_dE_vs_ase_f64") is not None:
            cell["dE"] = r["abs_dE_vs_ase_f64"]
        for src, dst in (
            ("force_mae", "mae"),
            ("force_rmse", "rmse"),
            ("force_max_abs", "max_abs"),
            ("force_max_norm_per_atom", "max_norm_atom"),
        ):
            if r.get(src) is not None and (
                mode == "primary_w8"
                or cell.get(dst) is None
                or cell.get(dst) == "—"
            ):
                cell[dst] = r[src]
        if job:
            cell["job"] = job
        if mode == "primary_w8":
            cell["status"] = "DONE_W8"
            cell["note"] = "W8 stream-ordered NCCL"
        elif not cell.get("status") or cell["status"] in ("PENDING",):
            cell["status"] = "DONE"
        if system == "nacl6":
            cell.setdefault("metric", "ms_per_eval_python")
        else:
            cell.setdefault("metric", "nvt_pair_ms_per_step")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ingest-matrix",
        action="store_true",
        help="Merge matrix/*/parity.json into settings_tables.json before render",
    )
    args = ap.parse_args()
    tables = json.loads(TABLES.read_text())
    stamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
    if args.ingest_matrix:
        n = ingest_matrix(tables)
        tables["stamp"] = stamp
        TABLES.write_text(json.dumps(tables, indent=2) + "\n")
        print(f"ingested {n} matrix cells → {TABLES}")
    else:
        tables["stamp"] = stamp
        TABLES.write_text(json.dumps(tables, indent=2) + "\n")

    for tag, meta in META.items():
        block = tables.get(tag)
        if block is None:
            raise SystemExit(f"missing tag {tag} in {TABLES}")
        out = DOCS / meta["file"]
        out.write_text(render_doc(tag, block, stamp))
        print(f"wrote {out.name}")
    (DOCS / "README.md").write_text(render_readme(stamp))
    print(f"wrote README.md  stamp={stamp}")


if __name__ == "__main__":
    main()
