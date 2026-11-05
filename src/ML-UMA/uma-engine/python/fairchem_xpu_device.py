"""Allow FairChem ``MLIPPredictUnit(device="xpu")`` without a sibling hen/ tree.

Stock fairchem-core asserts ``device in ["cpu", "cuda"]`` and seeds CUDA only.
This is device enablement only — same UMA math, no architecture change.
"""
from __future__ import annotations

import logging
import random

_DEVICE_PATCHED = False
_log = logging.getLogger(__name__)


def patch_fairchem_xpu_device() -> None:
    """Accept device='xpu' on MLIPPredictUnit and seed torch.xpu when present."""
    global _DEVICE_PATCHED
    if _DEVICE_PATCHED:
        return

    import numpy as np
    import torch
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

    _orig_setup = MLIPPredictUnit._setup_device

    def _setup_device(self, device: str) -> None:  # type: ignore[no-untyped-def]
        key = str(device).strip().lower()
        if key == "xpu" or key.startswith("xpu:"):
            if not hasattr(torch, "xpu") or not torch.xpu.is_available():
                raise RuntimeError(f"requested {device!r} but torch.xpu is unavailable")
            self.device = "xpu"
            return
        return _orig_setup(self, device)

    def _set_seed(self, seed: int) -> None:  # type: ignore[no-untyped-def]
        logging.debug("Setting random seed to %s", seed)
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)

    MLIPPredictUnit._setup_device = _setup_device  # type: ignore[method-assign]
    MLIPPredictUnit.set_seed = _set_seed  # type: ignore[method-assign]
    _DEVICE_PATCHED = True
    _log.info("MLIPPredictUnit XPU device patch applied (no hen/)")
