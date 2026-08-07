#!/usr/bin/env python3
"""Write Markdown + Cursor canvas reports from ``results/parity_table.json``.

Usage:
  python write_reports.py
  python write_reports.py /path/to/parity_table.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
RESULTS = OUT / "results"
DEFAULT_JSON = RESULTS / "parity_table.json"

# IDE-managed canvas directory for this workspace
CANVAS_DIR = Path(
    "/u/xyan11/.cursor/projects/work-nvme-bfzx-xyan11-workdir-lammps-uma/canvases"
)

SYSTEM_ORDER = (
    "nacl_n3_rattle",
    "al_fcc_rattle",
    "si_diamond_rattle",
    "nacl_n4_rattle",
    "nacl_n5_rattle",
    "nacl_n6_rattle",
    "al_fcc_n5_rattle",
    "si_diamond_n4_rattle",
    "nacl_n7_rattle",
    "si_diamond_n7_rattle",
    "nacl_n8_rattle",
    "al_fcc_n8_rattle",
    "si_diamond_n8_rattle",
)


def _fmt_e(x: float | None, prec: int = 10) -> str:
    if x is None:
        return "—"
    return f"{x:.{prec}f}"


def _fmt_sci(x: float | None) -> str:
    if x is None:
        return "—"
    if abs(x) == 0.0:
        return "0"
    return f"{x:.3e}"


def _fmt_ms(x: float | None) -> str:
    if x is None:
        return "—"
    if x >= 1000:
        return f"{x / 1000:.2f} s"
    return f"{x:.1f}"


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


def ordered_systems(report: dict) -> list[tuple[str, dict]]:
    systems = report.get("systems") or {}
    keys = [k for k in SYSTEM_ORDER if k in systems]
    keys += [k for k in systems if k not in keys]
    return [(k, systems[k]) for k in keys]


def write_markdown(report: dict, path: Path) -> None:
    pert = report.get("perturbation") or {}
    lines: list[str] = []
    lines.append("# UMA Delta parity report")
    lines.append("")
    lines.append(
        "Single-point **total energy** + **per-atom forces** + **timing** "
        "on rattled crystals (no lattice scale)."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- **GPU:** `{report.get('gpu', '?')}`")
    lines.append(f"- **Reference:** {report.get('reference', 'ASE FairChem FP64')}")
    lines.append(f"- **Checkpoint:** `{report.get('checkpoint', '')}`")
    lines.append(f"- **LMP_FC:** `{report.get('lmp_fc', '')}`")
    lines.append(f"- **LMP_UMA:** `{report.get('lmp_uma', '')}`")
    arts = report.get("artifacts") or {}
    if arts:
        lines.append(f"- **Artifact double:** `{arts.get('double', '')}`")
        lines.append(f"- **Artifact mixed:** `{arts.get('mixed', '')}`")
    lines.append(
        f"- **Perturbation:** {pert.get('mode', 'uniform_box')} "
        f"δ={pert.get('delta_A', 0.1)} Å, seed={pert.get('seed', 0)} "
        "(atomic displacement only)"
    )
    lines.append("")
    lines.append(
        "Force columns are **errors vs ASE**, not reference magnitudes: "
        "`max abs dF_i` = max component-wise |F−F_ASE|; "
        "`max norm dF_atom` = max per-atom ‖F−F_ASE‖."
    )
    lines.append("")
    lines.append(
        "Timing: ASE / FairChem = mean wall of repeated evals; "
        "uma/kk = LAMMPS Pair section / NVE steps after warmup."
    )
    lines.append("")

    for key, block in ordered_systems(report):
        title = block.get("system", key)
        natoms = block.get("natoms", "?")
        rows = block.get("rows") or []
        f_ref = None
        for r in rows:
            if r.get("path", "").startswith("ASE"):
                f_ref = r.get("f_ref_max_abs")
                break
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            f"{natoms} atoms"
            + (f"; ASE |F|_max = {_fmt_sci(f_ref)} eV/Å." if f_ref is not None else ".")
        )
        lines.append("")
        lines.append(
            "| Path | Energy (eV) | ms/eval | abs dE vs ASE (eV) | "
            "Force MAE | Force RMSE | max abs dF_i | max norm dF_atom | Cosine |"
        )
        lines.append(
            "|------|-------------|---------|-------------------|"
            "----------|-----------|--------------|------------------|--------|"
        )
        for r in rows:
            is_ase = str(r.get("path", "")).startswith("ASE")
            de = "—" if is_ase else _fmt_sci(r.get("abs_dE_vs_ase_f64"))
            mae = "—" if is_ase else _fmt_sci(r.get("force_mae"))
            rmse = "—" if is_ase else _fmt_sci(r.get("force_rmse"))
            mx = "—" if is_ase else _fmt_sci(r.get("force_max_abs"))
            mn = "—" if is_ase else _fmt_sci(r.get("force_max_norm_per_atom"))
            cos = "—" if is_ase else _fmt_e(r.get("cosine"), 6)
            lines.append(
                f"| {_short_path(r.get('path', ''))} | {_fmt_e(r.get('energy_eV'))} | "
                f"{_fmt_ms(r.get('ms_per_eval'))} | {de} | {mae} | {rmse} | {mx} | {mn} | {cos} |"
            )
        lines.append("")
        # Highlight uma double relative force error
        for r in rows:
            if "double" in str(r.get("path", "")).lower() and f_ref:
                rel = (r.get("force_max_abs") or 0) / f_ref if f_ref else None
                lines.append(
                    f"Relative force error (uma/kk double): "
                    f"|ΔF|_max / |F|_ref_max = {_fmt_sci(rel)} "
                    f"(max |ΔF_i| = {_fmt_sci(r.get('force_max_abs'))} eV/Å)."
                )
                lines.append("")
                break

    lines.append("## Notes")
    lines.append("")
    lines.append("- Energy is **total** (scalar). Forces are **per-atom**.")
    lines.append(
        "- Per-atom force arrays: `results/<sys>/per_atom_forces.npz` "
        "(`energy_*_eV`, `forces_*`)."
    )
    lines.append("- Machine-readable summary: `results/parity_table.json`.")
    if report.get("note"):
        lines.append(f"- {report['note']}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def write_canvas(report: dict, path: Path) -> None:
    """Emit a self-contained `.canvas.tsx` with embedded report data."""
    # Flatten rows for tables / charts
    systems = []
    timing_points = []
    de_points = []
    for key, block in ordered_systems(report):
        label = {
            "nacl_n3_rattle": "NaCl 3×3×3",
            "al_fcc_rattle": "Al 3×3×3",
            "si_diamond_rattle": "Si 3×3×3",
            "nacl_n4_rattle": "NaCl 4×4×4",
            "nacl_n5_rattle": "NaCl 5×5×5",
            "nacl_n6_rattle": "NaCl 6×6×6",
            "al_fcc_n5_rattle": "Al 5×5×5",
            "si_diamond_n4_rattle": "Si 4×4×4",
            "nacl_n7_rattle": "NaCl 7×7×7",
            "si_diamond_n7_rattle": "Si 7×7×7",
            "nacl_n8_rattle": "NaCl 8×8×8",
            "al_fcc_n8_rattle": "Al 8×8×8",
            "si_diamond_n8_rattle": "Si 8×8×8",
        }.get(key, key)
        rows_out = []
        for r in block.get("rows") or []:
            sp = _short_path(r.get("path", ""))
            rows_out.append(
                {
                    "path": sp,
                    "energy_eV": r.get("energy_eV"),
                    "ms_per_eval": r.get("ms_per_eval"),
                    "abs_dE": r.get("abs_dE_vs_ase_f64"),
                    "force_mae": r.get("force_mae"),
                    "force_rmse": r.get("force_rmse"),
                    "force_max_abs": r.get("force_max_abs"),
                    "force_max_norm": r.get("force_max_norm_per_atom"),
                    "cosine": r.get("cosine"),
                    "f_ref_max_abs": r.get("f_ref_max_abs"),
                }
            )
            if r.get("ms_per_eval") is not None:
                timing_points.append(
                    {"system": label, "series": sp, "ms": float(r["ms_per_eval"])}
                )
            if not sp.startswith("ASE") and r.get("abs_dE_vs_ase_f64") is not None:
                de_points.append(
                    {
                        "system": label,
                        "series": sp,
                        "dE": float(r["abs_dE_vs_ase_f64"]),
                    }
                )
        systems.append(
            {
                "key": key,
                "label": label,
                "title": block.get("system", key),
                "natoms": block.get("natoms"),
                "rows": rows_out,
            }
        )

    data = {
        "gpu": report.get("gpu"),
        "reference": report.get("reference"),
        "checkpoint": report.get("checkpoint"),
        "lmp_fc": report.get("lmp_fc"),
        "lmp_uma": report.get("lmp_uma"),
        "perturbation": report.get("perturbation"),
        "systems": systems,
        "timing_points": timing_points,
        "de_points": de_points,
    }
    data_json = json.dumps(data, indent=2)

    # Build bar-chart series from timing: one series per path
    path_names = []
    for s in systems:
        for r in s["rows"]:
            if r["path"] not in path_names:
                path_names.append(r["path"])

    tsx = f'''/* Auto-generated by write_reports.py — do not edit by hand. */
import {{
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Stack,
  Stat,
  Table,
  Text,
}} from "cursor/canvas";

const DATA = {data_json} as const;

function fmtE(x: number | null | undefined, p = 6): string {{
  if (x === null || x === undefined) return "—";
  return x.toFixed(p);
}}

function fmtSci(x: number | null | undefined): string {{
  if (x === null || x === undefined) return "—";
  if (x === 0) return "0";
  return x.toExponential(3);
}}

function fmtMs(x: number | null | undefined): string {{
  if (x === null || x === undefined) return "—";
  if (x >= 1000) return `${{(x / 1000).toFixed(2)}} s`;
  return `${{x.toFixed(1)}}`;
}}

function timingChartData() {{
  const systems = DATA.systems.map((s) => s.label);
  const series = {json.dumps(path_names)}.map((name: string) => ({{
    name,
    data: DATA.systems.map((s) => {{
      const row = s.rows.find((r) => r.path === name);
      return row?.ms_per_eval ?? 0;
    }}),
  }}));
  return {{ categories: systems, series }};
}}

function maxAbsDE(): number {{
  let m = 0;
  for (const s of DATA.systems) {{
    for (const r of s.rows) {{
      if (r.path.startsWith("ASE")) continue;
      if (r.abs_dE != null && r.abs_dE > m) m = r.abs_dE;
    }}
  }}
  return m;
}}

function bestUmaDoubleDE(): string {{
  const vals: number[] = [];
  for (const s of DATA.systems) {{
    const r = s.rows.find((x) => x.path.includes("double"));
    if (r?.abs_dE != null) vals.push(r.abs_dE);
  }}
  if (!vals.length) return "—";
  return fmtSci(Math.max(...vals));
}}

export default function UmaDeltaParity() {{
  const timing = timingChartData();
  const nSys = DATA.systems.length;
  const pert = DATA.perturbation;

  return (
    <Stack gap={{24}}>
      <Stack gap={{8}}>
        <H1>UMA Delta parity</H1>
        <Text tone="secondary">
          Total energy + per-atom forces + timing · rattled crystals · no lattice scale
        </Text>
        <Text tone="tertiary" size="small">
          GPU: {{DATA.gpu ?? "?"}} · ref: {{DATA.reference ?? "ASE FairChem FP64"}} ·
          rattle δ={{pert?.delta_A ?? 0.1}} Å seed={{pert?.seed ?? 0}}
        </Text>
      </Stack>

      <Grid columns={{4}} gap={{16}}>
        <Stat value={{String(nSys)}} label="Systems" />
        <Stat value="4" label="Paths / system" />
        <Stat value={{bestUmaDoubleDE()}} label="uma double max |ΔE| (eV)" tone="success" />
        <Stat value={{fmtSci(maxAbsDE())}} label="Worst |ΔE| any path (eV)" />
      </Grid>

      <Callout tone="info">
        Force metrics are errors vs ASE FP64. Energy is total (scalar) only; forces are per-atom.
        FairChem fix-external builds the cell in FP32 inside lammps_fc.
      </Callout>

      <Stack gap={{8}}>
        <H2>Timing (ms / eval)</H2>
        <Text tone="secondary" size="small">
          ASE/FairChem: mean wall of repeated evals. uma/kk: LAMMPS Pair / NVE steps.
        </Text>
        <BarChart
          categories={{timing.categories}}
          series={{timing.series}}
          height={{280}}
          valueSuffix=" ms"
        />
      </Stack>

      <Divider />

      {{DATA.systems.map((sys) => (
        <Stack key={{sys.key}} gap={{12}}>
          <Stack gap={{4}}>
            <H2>{{sys.label}}</H2>
            <Text tone="secondary" size="small">
              {{sys.title}} · {{sys.natoms}} atoms · ASE |F|_max ={{" "}}
              {{fmtSci(sys.rows.find((r) => r.path.startsWith("ASE"))?.f_ref_max_abs)}} eV/Å
            </Text>
          </Stack>
          <Card>
            <CardHeader trailing={{<Text size="small">{{sys.natoms}} atoms</Text>}}>
              Energy / force parity
            </CardHeader>
            <CardBody>
              <Table
                stickyHeader
                striped
                headers={{[
                  "Path",
                  "E (eV)",
                  "ms/eval",
                  "|ΔE|",
                  "F MAE",
                  "F RMSE",
                  "max |ΔF|",
                  "max ‖ΔF‖",
                  "cos",
                ]}}
                columnAlign={{[
                  "left",
                  "right",
                  "right",
                  "right",
                  "right",
                  "right",
                  "right",
                  "right",
                  "right",
                ]}}
                rows={{sys.rows.map((r) => {{
                  const ase = r.path.startsWith("ASE");
                  return [
                    r.path,
                    fmtE(r.energy_eV, 6),
                    fmtMs(r.ms_per_eval),
                    ase ? "—" : fmtSci(r.abs_dE),
                    ase ? "—" : fmtSci(r.force_mae),
                    ase ? "—" : fmtSci(r.force_rmse),
                    ase ? "—" : fmtSci(r.force_max_abs),
                    ase ? "—" : fmtSci(r.force_max_norm),
                    ase ? "—" : fmtE(r.cosine, 6),
                  ];
                }})}}
              />
            </CardBody>
          </Card>
        </Stack>
      ))}}

      <Stack gap={{4}}>
        <H3>Artifacts</H3>
        <Text size="small" tone="tertiary">
          LMP_FC: {{DATA.lmp_fc ?? "—"}}
        </Text>
        <Text size="small" tone="tertiary">
          LMP_UMA: {{DATA.lmp_uma ?? "—"}}
        </Text>
        <Text size="small" tone="tertiary">
          Checkpoint: {{DATA.checkpoint ?? "—"}}
        </Text>
      </Stack>
    </Stack>
  );
}}
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tsx, encoding="utf-8")
    print(f"wrote {path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    json_path = Path(args[0]) if args else DEFAULT_JSON
    if not json_path.is_file():
        raise SystemExit(f"missing report JSON: {json_path}")

    report = json.loads(json_path.read_text())
    if not (report.get("systems") or {}):
        raise SystemExit(f"no systems in {json_path}")

    md_path = RESULTS / "PARITY_REPORT.md"
    canvas_repo = RESULTS / "uma-delta-parity.canvas.tsx"
    canvas_ide = CANVAS_DIR / "uma-delta-parity.canvas.tsx"

    write_markdown(report, md_path)
    write_canvas(report, canvas_repo)
    # Mirror into IDE canvas dir when writable
    try:
        write_canvas(report, canvas_ide)
    except OSError as exc:
        print(f"warning: could not write IDE canvas ({exc})", flush=True)

    print(
        f"reports ready:\n  md={md_path}\n  canvas={canvas_repo}\n  ide={canvas_ide}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
