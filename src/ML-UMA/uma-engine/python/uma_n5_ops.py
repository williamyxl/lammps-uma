"""Step 22 (N5.3). Export-time stand-in for the five fused ops.

Schemas are read from include/uma/n5_ops.h so the traced name matches the
engine. The eager body returns its input. The exporter calls nothing here
when UMA_NITRO_FUSED is empty, so the stock graph is unchanged. A later block
stage replaces the body with the stock math it fuses.
"""
from __future__ import annotations

from pathlib import Path

from torch.library import Library, impl

_HEADER = Path(__file__).resolve().parents[1] / "include" / "uma" / "n5_ops.h"
_LIB: Library | None = None


def schemas() -> dict[str, str]:
    found: dict[str, str] = {}
    for line in _HEADER.read_text().splitlines():
        if "_SCHEMA" not in line or '"' not in line:
            continue
        name = line.split("UMA_N5_", 1)[1].split("_SCHEMA", 1)[0].lower()
        found[name] = line.split('"', 2)[1]
    if set(found) != {"f1", "f2", "f3", "f4", "f5"}:
        raise RuntimeError(f"uma_n5 schemas missing in {_HEADER}: {sorted(found)}")
    return found


def register() -> Library:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = Library("uma_n5", "DEF")
    for name, schema in schemas().items():
        lib.define(schema)

        def _eager(x, _name=name):
            return x

        impl(lib, name, "CompositeExplicitAutograd")(_eager)
    _LIB = lib
    return lib


def _load_on_import() -> None:
    # Tracing imports this module only when a flag is set. Register then.
    register()


_load_on_import()
