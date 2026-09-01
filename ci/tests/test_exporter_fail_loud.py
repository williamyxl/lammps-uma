#!/usr/bin/env python3
"""Tier 1 (R3/G4): exporters fail LOUD when a correctness-critical patch is missing.

Pure Python (no torch/fairchem/XPU). The wigner-chunk fix (N>=10 FP64 cliff) and the
xpu-device patch are correctness-critical; a silently-skipped patch yields an
artifact with wrong forces. This test asserts, by static inspection, that BOTH
exporters (`export_blocks_xpu.py` = P5'.5, `export_shards_xpu.py` = R3/G4) `raise`
on that failure rather than `WARN`+continue, and honor the same
`UMA_ALLOW_MISSING_PATCHES` escape hatch.

This is the regression guard for the audit finding that P5'.5 was closed on only
2 of the (then) exporter sites — `export_shards_xpu.py` was still fail-open.
"""
import re
import sys
from pathlib import Path

PY = Path(__file__).resolve().parents[2] / "src" / "ML-UMA" / "uma-engine" / "python"


def _wigner_block(src_text: str) -> str:
    """Return the code region around the wigner-chunk patch apply."""
    m = re.search(r"apply_xpu_prepare_wigner_chunking", src_text)
    assert m, "wigner-chunk apply site not found"
    start = src_text.rfind("try:", 0, m.start())
    # take the try/except that contains the apply call (~30 lines is plenty)
    return src_text[start:m.start() + 1200]


def _check_exporter(fname: str):
    text = (PY / fname).read_text()
    block = _wigner_block(text)
    assert "raise RuntimeError" in block, \
        f"{fname}: wigner-chunk failure must `raise`, not WARN+continue"
    assert "UMA_ALLOW_MISSING_PATCHES" in text, \
        f"{fname}: must honor the UMA_ALLOW_MISSING_PATCHES escape hatch"
    # The escape must gate the raise (permissive only when explicitly set).
    assert re.search(r"_allow_missing|UMA_ALLOW_MISSING_PATCHES", block), \
        f"{fname}: the wigner raise must be gated by the allow-missing flag"


def test_export_blocks_fail_loud():
    _check_exporter("export_blocks_xpu.py")


def test_export_shards_fail_loud():
    # R3/G4: this was the fail-open sibling.
    _check_exporter("export_shards_xpu.py")


def test_both_exporters_parse():
    import ast
    for f in ("export_blocks_xpu.py", "export_shards_xpu.py"):
        ast.parse((PY / f).read_text())


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    n_fail = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception:
            n_fail += 1; print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{len(tests)-n_fail}/{len(tests)} passed")
    sys.exit(1 if n_fail else 0)
