#!/usr/bin/env python3
"""Emit a small NaCl extxyz + a seeded metadata.json for the non-AC virial artifact.

The virial FD test needs a PLAIN (non-activation-checkpointed) traced artifact and a
small system (7 single-points). This writes:
  <OUT>/nacl.extxyz         : N=2 supercell (64 atoms), a=5.64, rattle 0.05, seed 0
  <OUT>/metadata.json       : seeded from a base artifact, metadata_version=2
matching the geometry virial_fd_check.py builds (build_nacl(n=2)).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase6_make_gp_inputs import build_nacl  # noqa: E402


def main():
    out = Path(os.environ["OUT"]); out.mkdir(parents=True, exist_ok=True)
    base_meta = Path(os.environ["BASE_META"])
    n = int(os.environ.get("N", "2"))
    syms, pos, cell = build_nacl(n)
    natoms = len(syms)

    # extxyz
    lat = " ".join(f"{cell[i,j]:.10f}" for i in range(3) for j in range(3))
    lines = [str(natoms),
             f'Lattice="{lat}" Properties=species:S:1:pos:R:3 pbc="T T T"']
    for s, p in zip(syms, pos):
        lines.append(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}")
    (out / "nacl.extxyz").write_text("\n".join(lines) + "\n")

    # seed metadata (plain, non-AC): copy base, drop AC/GP fields, stamp v2.
    meta = json.loads(base_meta.read_text())
    for k in ("num_blocks", "num_chunk_modules", "num_edgedeg",
              "edgedeg_chunk_module", "edge_pad_cap", "edge_pad_atom",
              "edge_ac_chunk", "gp", "gp_node_offset"):
        meta.pop(k, None)
    meta["metadata_version"] = 2
    meta["export_format"] = "w15_plain_traced"
    meta["world"] = 1
    meta["rank"] = 0
    inf = dict(meta.get("inference_settings") or {})
    inf["base_precision_dtype"] = "float64"
    inf["activation_checkpointing"] = False
    meta["inference_settings"] = inf
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"wrote {out}/nacl.extxyz ({natoms} atoms) + metadata.json (v2, plain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
