"""
Local UMA checkpoint paths under uma-cache/ (default: sibling of repo root).
"""

from __future__ import annotations

import os
from pathlib import Path

from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name

# Basenames in uma-cache/ (see /mnt/d/workdir/uma-cache/).
UMA_CACHE_CHECKPOINTS: dict[str, str] = {
    "uma-s-1p2": "uma-s-1p2.pt",
    "uma-m-1p1": "uma-m-1p1.pt",
    "uma-s-1p1": "uma-s-1p1.pt",
}


def uma_cache_dir() -> Path:
    """Directory containing flat *.pt checkpoints."""
    if env := os.environ.get("UMA_CACHE_DIR"):
        return Path(env).expanduser().resolve()
    # default: /mnt/d/workdir/uma-cache
    return Path("/mnt/d/workdir/uma-cache")


def checkpoint_path_for_model(model_name: str) -> Path:
    if model_name not in UMA_CACHE_CHECKPOINTS:
        raise KeyError(
            f"No uma-cache mapping for {model_name!r}. "
            f"Known: {sorted(UMA_CACHE_CHECKPOINTS)}"
        )
    return uma_cache_dir() / UMA_CACHE_CHECKPOINTS[model_name]


def resolve_checkpoint(model_name: str, explicit: str | Path | None = None) -> str:
    """
    Resolve checkpoint path: explicit CLI arg > uma-cache/{model}.pt > HuggingFace.
    """
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return str(path)

    if model_name in UMA_CACHE_CHECKPOINTS:
        cached = checkpoint_path_for_model(model_name)
        if cached.is_file():
            return str(cached)

    return pretrained_checkpoint_path_from_name(model_name)
