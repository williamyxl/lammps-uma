#!/usr/bin/env python3
"""Step 23 (N5.4). Fused gate versus stock GateActivation.

Times two forwards plus one backward (the three checkpoint passes). The
scaffold path is torch autograd on the stock math and is not timed. F2 has
no GEMM, so there is no vendor-library GEMM variant.
"""
from __future__ import annotations

import os
import statistics
import time

import torch

from fairchem.core.models.uma.nn.activation import GateActivation

LIB = os.environ["F2_LIB"]
torch.ops.load_library(LIB)

VARIANTS = (
    (0, "wg128"),
    (1, "wg64"),
    (2, "wg32"),
    (3, "wg128_sg16"),
    (4, "wg128_sg32"),
    (5, "wg128_local"),
    (6, "wg64_local"),
    (7, "wg32_local"),
)
REL_BAR = 1e-12


def sync():
    torch.xpu.synchronize()


def rel(a, b):
    diff = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-30)
    return diff, diff / scale


def stock_fn(module):
    def run(g, x):
        return module(g, x)

    return run


def fused_fn(expand, variant):
    def run(g, x):
        return torch.ops.uma_n5f2.gate(g, x, expand, variant)

    return run


def three(run, g, x):
    with torch.no_grad():
        run(g, x)
    g2 = g.detach().requires_grad_(True)
    x2 = x.detach().requires_grad_(True)
    y = run(g2, x2)
    y.sum().backward()


def time_pair(stock, fused, g, x, reps=30):
    for _ in range(2):
        three(stock, g, x)
        three(fused, g, x)
    sync()
    left, right = [], []
    for i in range(reps):
        order = ((stock, left), (fused, right))
        if i % 2:
            order = ((fused, right), (stock, left))
        for fn, bucket in order:
            sync()
            t0 = time.perf_counter()
            three(fn, g, x)
            sync()
            bucket.append((time.perf_counter() - t0) * 1e3)
    med_s = statistics.median(left)
    med_f = statistics.median(right)
    mad_s = statistics.median(abs(v - med_s) for v in left)
    mad_f = statistics.median(abs(v - med_f) for v in right)
    return med_s, mad_s, med_f, mad_f


def check_pair(name, stock, fused, g, x):
    with torch.no_grad():
        ys, yf = stock(g, x), fused(g, x)
    y_abs, y_rel = rel(ys, yf)
    gs = g.detach().requires_grad_(True)
    xs = x.detach().requires_grad_(True)
    gf = g.detach().requires_grad_(True)
    xf = x.detach().requires_grad_(True)
    stock(gs, xs).sum().backward()
    fused(gf, xf).sum().backward()
    gx_abs, gx_rel = rel(xs.grad, xf.grad)
    gg_abs, gg_rel = rel(gs.grad, gf.grad)
    ok = y_rel <= REL_BAR and gx_rel <= REL_BAR and gg_rel <= REL_BAR
    print(
        f"N54_CHECK {name} y_abs={y_abs:.3e} y_rel={y_rel:.3e} "
        f"gx_abs={gx_abs:.3e} gx_rel={gx_rel:.3e} "
        f"gg_abs={gg_abs:.3e} gg_rel={gg_rel:.3e} ok={int(ok)}",
        flush=True,
    )
    return ok


def main():
    print("N54_ENTER", flush=True)
    print("N54_NOTE no GEMM in the gate; no vendor GEMM variant", flush=True)
    dev = torch.device("xpu")
    module = GateActivation(lmax=2, mmax=2, num_channels=128, m_prime=True).to(dev).double()
    expand = module.expand_index.to(dev)
    print("N54_EXPAND " + ",".join(str(int(v)) for v in expand.cpu()), flush=True)
    stock = stock_fn(module)
    g = torch.randn(65536, 256, dtype=torch.float64, device=dev)
    x = torch.randn(65536, 9, 128, dtype=torch.float64, device=dev)
    ok = True
    for vid, name in VARIANTS:
        try:
            ok = check_pair(name, stock, fused_fn(expand, vid), g, x) and ok
        except Exception as exc:
            ok = False
            print(f"N54_CHECK {name} FAIL {type(exc).__name__}: {exc}", flush=True)

    gg, gx = torch.ops.uma_n5f2.gate_scaffold(g, x, expand)
    gs = g.detach().requires_grad_(True)
    xs = x.detach().requires_grad_(True)
    stock(gs, xs).sum().backward()
    s_abs, s_rel = rel(gs.grad, gg)
    x_abs, x_rel = rel(xs.grad, gx)
    scaffold_ok = s_rel <= REL_BAR and x_rel <= REL_BAR
    print(
        f"N54_SCAFFOLD gg_abs={s_abs:.3e} gg_rel={s_rel:.3e} "
        f"gx_abs={x_abs:.3e} gx_rel={x_rel:.3e} ok={int(scaffold_ok)}",
        flush=True,
    )
    ok = ok and scaffold_ok

    gs = torch.randn(2, 4, dtype=torch.float64, device=dev, requires_grad=True)
    xs = torch.randn(2, 9, 2, dtype=torch.float64, device=dev, requires_grad=True)
    expand_small = expand.clone()
    try:
        grad_ok = torch.autograd.gradcheck(
            lambda a, b: torch.ops.uma_n5f2.gate(a, b, expand_small, 0),
            (gs, xs),
            eps=1e-6,
            atol=1e-8,
            rtol=1e-6,
        )
    except Exception as exc:
        grad_ok = False
        print(f"N54_GRADCHECK FAIL {type(exc).__name__}: {exc}", flush=True)
    else:
        print(f"N54_GRADCHECK ok={int(bool(grad_ok))}", flush=True)
    ok = ok and bool(grad_ok)

    for vid, name in VARIANTS:
        try:
            med_s, mad_s, med_f, mad_f = time_pair(stock, fused_fn(expand, vid), g, x)
        except Exception as exc:
            print(f"N54_ROW {name} FAIL {type(exc).__name__}: {exc}", flush=True)
            ok = False
            continue
        gain = med_s - med_f
        print(
            f"N54_ROW {name} stock_ms={med_s:.4f} mad={mad_s:.4f} "
            f"fused_ms={med_f:.4f} mad={mad_f:.4f} gain_ms={gain:.4f}",
            flush=True,
        )
    if ok:
        print("N54_PASS", flush=True)
        return
    print("N54_FAIL", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
