"""pytest fixtures/markers for the UMA Tier-1 tests.

Makes scripts/ and the exporter python/ importable, and auto-skips tests marked
needs_torch / needs_fairchem when those packages are absent (login base env), so
`pytest ci/tests` is green everywhere and the full set runs under the fxpu env.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "scripts", REPO / "src" / "ML-UMA" / "uma-engine" / "python"):
    if p.is_dir():
        sys.path.insert(0, str(p))


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def pytest_collection_modifyitems(config, items):
    skip_torch = pytest.mark.skip(reason="torch not available (run under fxpu env)")
    skip_fc = pytest.mark.skip(reason="fairchem-core not available (run under fxpu env)")
    have_torch = _have("torch")
    have_fc = _have("fairchem")
    for item in items:
        if "needs_torch" in item.keywords and not have_torch:
            item.add_marker(skip_torch)
        if "needs_fairchem" in item.keywords and not have_fc:
            item.add_marker(skip_fc)
