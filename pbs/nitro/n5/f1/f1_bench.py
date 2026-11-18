#!/usr/bin/env python3
"""Step 26 (N5.7). Fused LayerNorm + SiLU versus stock F.layer_norm then silu.

Times two forwards plus one backward (the three checkpoint passes). The
scaffold path is torch autograd on the stock math and is not timed. F1 has
no GEMM, so there is no vendor-library GEMM variant. Shape is the RadialMLP
hidden width: E=65536, C=128.
"""
from __future__ import annotations

import os
import statistics
import time

import torch
import torch.nn.functional as F

LIB = os.environ["F1_LIB"]
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
EPS = 1e-5
E = 65536
C = 128


def sync():
    torch.xpu.synchronize()


def rel(a, b):
    diff = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-30)
    return diff, diff / scale


def stock_fn(weight, bias):
    def run(x):
        return F.silu(F.layer_norm(x, (x.shape[-1],), weight, bias, EPS))

    return run


def fused_fn(weight, bias, variant):
    def run(x):
        return torch.ops.uma_n5f1.ln_silu(x, weight, bias, EPS, variant)

    return run


def three(run, x):
    with torch.no_grad():
        run(x)
    x2 = x.detach().requires_grad_(True)
    y = run(x2)
    y.sum().backward()


def time_pair(stock, fused, x, reps=30):
    for _ in range(2):
        three(stock, x)
        three(fused, x)
    sync()
    left, right = [], []
    for i in range(reps):
        order = ((stock, left), (fused, right))
        if i % 2:
            order = ((fused, right), (stock, left))
        for fn, bucket in order:
            sync()
            t0 = time.perf_counter()
            three(fn, x)
            sync()
            bucket.append((time.perf_counter() - t0) * 1e3)
    med_s = statistics.median(left)
    med_f = statistics.median(right)
    mad_s = statistics.median(abs(v - med_s) for v in left)
    mad_f = statistics.median(abs(v - med_f) for v in right)
    return med_s, mad_s, med_f, mad_f


def check_pair(name, stock, fused, x):
    with torch.no_grad():
        ys, yf = stock(x), fused(x)
    y_abs, y_rel = rel(ys, yf)
    xs = x.detach().requires_grad_(True)
    xf = x.detach().requires_grad_(True)
    stock(xs).sum().backward()
    fused(xf).sum().backward()
    gx_abs, gx_rel = rel(xs.grad, xf.grad)
    ok = y_rel <= REL_BAR and gx_rel <= REL_BAR
    print(
        f"N57_CHECK {name} y_abs={y_abs:.3e} y_rel={y_rel:.3e} "
        f"gx_abs={gx_abs:.3e} gx_rel={gx_rel:.3e} ok={int(ok)}",
        flush=True,
    )
    return ok


def main():
    print("N57_ENTER", flush=True)
    print("N57_NOTE no GEMM in LN+SiLU; no vendor GEMM variant", flush=True)
    print(f"N57_SHAPE E={E} C={C} eps={EPS}", flush=True)
    dev = torch.device("xpu")
    weight = torch.randn(C, dtype=torch.float64, device=dev)
    bias = torch.randn(C, dtype=torch.float64, device=dev)
    x = torch.randn(E, C, dtype=torch.float64, device=dev)
    stock = stock_fn(weight, bias)
    ok = True
    for vid, name in VARIANTS:
        try:
            ok = check_pair(name, stock, fused_fn(weight, bias, vid), x) and ok
        except Exception as exc:
            ok = False
            print(f"N57_CHECK {name} FAIL {type(exc).__name__}: {exc}", flush=True)

    gx_sc = torch.ops.uma_n5f1.ln_silu_scaffold(x, weight, bias, EPS)
    xs = x.detach().requires_grad_(True)
    stock(xs).sum().backward()
    s_abs, s_rel = rel(xs.grad, gx_sc)
    scaffold_ok = s_rel <= REL_BAR
    print(
        f"N57_SCAFFOLD gx_abs={s_abs:.3e} gx_rel={s_rel:.3e} ok={int(scaffold_ok)}",
        flush=True,
    )
    ok = ok and scaffold_ok

    w_s = torch.randn(4, dtype=torch.float64, device=dev)
    b_s = torch.randn(4, dtype=torch.float64, device=dev)
    xs = torch.randn(2, 4, dtype=torch.float64, device=dev, requires_grad=True)
    try:
        grad_ok = torch.autograd.gradcheck(
            lambda a: torch.ops.uma_n5f1.ln_silu(a, w_s, b_s, EPS, 0),
            (xs,),
            eps=1e-6,
            atol=1e-8,
            rtol=1e-6,
        )
    except Exception as exc:
        grad_ok = False
        print(f"N57_GRADCHECK FAIL {type(exc).__name__}: {exc}", flush=True)
    else:
        print(f"N57_GRADCHECK ok={int(bool(grad_ok))}", flush=True)
    ok = ok and bool(grad_ok)

    for vid, name in VARIANTS:
        try:
            med_s, mad_s, med_f, mad_f = time_pair(stock, fused_fn(weight, bias, vid), x)
        except Exception as exc:
            print(f"N57_ROW {name} FAIL {type(exc).__name__}: {exc}", flush=True)
            ok = False
            continue
        gain = med_s - med_f
        print(
            f"N57_ROW {name} stock_ms={med_s:.4f} mad={mad_s:.4f} "
            f"fused_ms={med_f:.4f} mad={mad_f:.4f} gain_ms={gain:.4f}",
            flush=True,
        )
    if ok:
        print("N57_PASS", flush=True)
        return
    print("N57_FAIL", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
