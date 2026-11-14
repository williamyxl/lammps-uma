"""Trace a one-op module on XPU and save it. Compute-node only."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

import uma_sycl_ops


class SiluAffine(nn.Module):
    def forward(self, x, w, b):
        return torch.ops.uma_sycl.silu_affine(x, w, b)


def inputs(n: int, device: torch.device):
    idx = torch.arange(n, dtype=torch.float64, device=device)
    x = torch.sin(idx * 0.01)
    w = torch.cos(idx * 0.02) * 0.5 + 0.1
    b = torch.sin(idx * 0.007) * 2.0
    return x, w, b


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: trace_s2.py OUT.pt", file=sys.stderr)
        return 2
    if not torch.xpu.is_available():
        print("S2_TRACE_FAIL no XPU", file=sys.stderr)
        return 1
    uma_sycl_ops.ensure()
    device = torch.device("xpu")
    model = SiluAffine().to(device)
    example = inputs(128, device)
    traced = torch.jit.trace(model, example)
    graph = str(traced.graph)
    print(graph)
    if "uma_sycl::silu_affine" not in graph:
        print("S2_TRACE_FAIL op not in graph", file=sys.stderr)
        return 1
    # A baked example length would reject a different n.
    other = inputs(1000, device)
    y_other = traced(*other)
    y_ref = other[0] * other[1] + other[2]
    y_ref = y_ref * torch.sigmoid(y_ref)
    err = (y_other - y_ref).abs().max().item()
    if err > 1e-12:
        print(f"S2_TRACE_FAIL second shape err {err}", file=sys.stderr)
        return 1
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    print(f"S2_TRACE_OK {out} schema={uma_sycl_ops.schema()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
