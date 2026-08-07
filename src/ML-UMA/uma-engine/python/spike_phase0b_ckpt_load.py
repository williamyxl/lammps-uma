#!/usr/bin/env python3
"""Phase 0b spike: prove FairChem ckpt is not LibTorch-loadable; traced artifact is.

No Ray. Run:
  python spike_phase0b_ckpt_load.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = Path("/work/nvme/bfzx/xyan11/workdir/uma-cache/uma-s-1p2.pt")
ART = ROOT / "artifacts" / "uma-s-1p2-omat-f64" / "model_traced.pt"


def main() -> int:
    out: dict = {"ckpt": str(CKPT), "artifact": str(ART)}
    try:
        torch.jit.load(str(CKPT), map_location="cpu")
        out["jit_load_ckpt"] = "UNEXPECTED_OK"
    except Exception as e:
        out["jit_load_ckpt"] = f"FAIL_EXPECTED: {type(e).__name__}: {e}"

    if not ART.is_file():
        out["jit_load_artifact"] = "MISSING"
        print(json.dumps(out, indent=2))
        return 2
    try:
        m = torch.jit.load(str(ART), map_location="cpu")
        out["jit_load_artifact"] = "OK"
        out["artifact_type"] = type(m).__name__
    except Exception as e:
        out["jit_load_artifact"] = f"FAIL: {e}"
        print(json.dumps(out, indent=2))
        return 1

    out["conclusion"] = (
        "Phase 0b: C++/LibTorch multi-GPU cannot start from Hydra ckpt; "
        "need non-opaque export or C++ MP+state_dict. Do not pivot to Ray."
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
