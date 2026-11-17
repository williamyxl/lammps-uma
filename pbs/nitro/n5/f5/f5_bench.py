#!/usr/bin/env python3
"""Step 25 (N5.6). Fused inverse rotation and scatter versus stock index_add_.

N2-b was not adopted, so the scatter is the same atomic add as stock.
conv2 GEMMs stay on the vendor library. One row includes a 384-wide vendor
linear. A split of the edges checks that two chunks add up to one scatter.
"""
from __future__ import annotations

import os
import statistics
import time

import torch

LIB = os.environ["F5_LIB"]
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
N_NODES = 1830


def sync():
    torch.xpu.synchronize()


def rel(a, b):
    diff = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-30)
    return diff, diff / scale


def stock(x, w, edge):
    y = torch.zeros(N_NODES, w.shape[1], x.shape[2], dtype=x.dtype, device=x.device)
    y.index_add_(0, edge[1], torch.bmm(w, x))
    return y


def fused(x, w, edge, variant):
    return torch.ops.uma_n5f5.scatter(x, w, edge, N_NODES, 0, variant)


def three(run, x, w, edge):
    with torch.no_grad():
        run(x, w, edge)
    x2 = x.detach().requires_grad_(True)
    w2 = w.detach().requires_grad_(True)
    run(x2, w2, edge).sum().backward()


def time_pair(stock_run, fused_run, x, w, edge, reps=30):
    for _ in range(2):
        three(stock_run, x, w, edge)
        three(fused_run, x, w, edge)
    sync()
    left, right = [], []
    for i in range(reps):
        order = ((stock_run, left), (fused_run, right))
        if i % 2:
            order = ((fused_run, right), (stock_run, left))
        for fn, bucket in order:
            sync()
            t0 = time.perf_counter()
            three(fn, x, w, edge)
            sync()
            bucket.append((time.perf_counter() - t0) * 1e3)
    med_s = statistics.median(left)
    med_f = statistics.median(right)
    mad_s = statistics.median(abs(v - med_s) for v in left)
    mad_f = statistics.median(abs(v - med_f) for v in right)
    return med_s, mad_s, med_f, mad_f


def main():
    print("N56_ENTER", flush=True)
    print("N56_NOTE N2-b was not adopted; scatter is atomic index_add_. conv2 GEMM stays vendor", flush=True)
    dev = torch.device("xpu")
    torch.manual_seed(0)
    e, m, c, ell = 65536, 9, 128, 9
    x = torch.randn(e, m, c, dtype=torch.float64, device=dev)
    w = torch.randn(e, ell, m, dtype=torch.float64, device=dev)
    src = torch.randint(0, 21952, (e,), dtype=torch.int64, device=dev)
    dst = torch.randint(0, N_NODES, (e,), dtype=torch.int64, device=dev)
    edge = torch.stack((src, dst))
    ok = True
    for vid, name in VARIANTS:
        try:
            with torch.no_grad():
                ys, yf = stock(x, w, edge), fused(x, w, edge, vid)
            y_abs, y_rel = rel(ys, yf)
            xs = x.detach().requires_grad_(True)
            ws = w.detach().requires_grad_(True)
            xf = x.detach().requires_grad_(True)
            wf = w.detach().requires_grad_(True)
            stock(xs, ws, edge).sum().backward()
            fused(xf, wf, edge, vid).sum().backward()
            gx_abs, gx_rel = rel(xs.grad, xf.grad)
            gw_abs, gw_rel = rel(ws.grad, wf.grad)
            good = y_rel <= REL_BAR and gx_rel <= REL_BAR and gw_rel <= REL_BAR
            print(
                f"N56_CHECK {name} y_rel={y_rel:.3e} gx_rel={gx_rel:.3e} gw_rel={gw_rel:.3e} "
                f"y_abs={y_abs:.3e} ok={int(good)}",
                flush=True,
            )
            ok = ok and good
        except Exception as exc:
            ok = False
            print(f"N56_CHECK {name} FAIL {type(exc).__name__}: {exc}", flush=True)

    with torch.no_grad():
        mid = e // 2
        left = fused(x[:mid], w[:mid], edge[:, :mid], 0)
        right = fused(x[mid:], w[mid:], edge[:, mid:], 0)
        whole = fused(x, w, edge, 0)
        b_abs, b_rel = rel(left + right, whole)
    bound_ok = b_rel <= REL_BAR
    print(f"N56_BOUNDARY y_rel={b_rel:.3e} y_abs={b_abs:.3e} ok={int(bound_ok)}", flush=True)
    ok = ok and bound_ok

    gx_s, gw_s = torch.ops.uma_n5f5.scatter_scaffold(x, w, edge, N_NODES, 0)
    xs = x.detach().requires_grad_(True)
    ws = w.detach().requires_grad_(True)
    stock(xs, ws, edge).sum().backward()
    s_abs, s_rel = rel(xs.grad, gx_s)
    w_abs, w_rel = rel(ws.grad, gw_s)
    scaffold_ok = s_rel <= REL_BAR and w_rel <= REL_BAR
    print(
        f"N56_SCAFFOLD gx_rel={s_rel:.3e} gw_rel={w_rel:.3e} "
        f"gx_abs={s_abs:.3e} gw_abs={w_abs:.3e} ok={int(scaffold_ok)}",
        flush=True,
    )
    ok = ok and scaffold_ok

    xs = torch.randn(4, 3, 2, dtype=torch.float64, device=dev, requires_grad=True)
    ws = torch.randn(4, 3, 3, dtype=torch.float64, device=dev, requires_grad=True)
    es = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 1]], dtype=torch.int64, device=dev)
    try:
        grad_ok = torch.autograd.gradcheck(
            lambda a, b: torch.ops.uma_n5f5.scatter(a, b, es, 2, 0, 0),
            (xs, ws),
            eps=1e-6,
            atol=1e-7,
            rtol=1e-5,
        )
    except Exception as exc:
        grad_ok = False
        print(f"N56_GRADCHECK FAIL {type(exc).__name__}: {exc}", flush=True)
    else:
        print(f"N56_GRADCHECK ok={int(bool(grad_ok))}", flush=True)
    ok = ok and bool(grad_ok)

    for vid, name in VARIANTS:
        try:
            med_s, mad_s, med_f, mad_f = time_pair(
                stock, lambda a, b, e_, _v=vid: fused(a, b, e_, _v), x, w, edge
            )
        except Exception as exc:
            ok = False
            print(f"N56_ROW {name} FAIL {type(exc).__name__}: {exc}", flush=True)
            continue
        print(
            f"N56_ROW {name} stock_ms={med_s:.4f} mad={mad_s:.4f} "
            f"fused_ms={med_f:.4f} mad={mad_f:.4f} gain_ms={med_s - med_f:.4f}",
            flush=True,
        )

    weight = torch.randn(384, 384, dtype=torch.float64, device=dev)
    bias = torch.randn(384, dtype=torch.float64, device=dev)

    def with_gemm(scatter):
        def run(a, b, e_):
            flat = a[:, :3, :].reshape(a.shape[0], -1)
            torch.nn.functional.linear(flat, weight, bias)
            return scatter(a, b, e_)

        return run

    try:
        med_s, mad_s, med_f, mad_f = time_pair(
            with_gemm(stock), with_gemm(lambda a, b, e_: fused(a, b, e_, 0)), x, w, edge
        )
        print(
            f"N56_ROW vendor_fc_m0 stock_ms={med_s:.4f} mad={mad_s:.4f} "
            f"fused_ms={med_f:.4f} mad={mad_f:.4f} gain_ms={med_s - med_f:.4f}",
            flush=True,
        )
    except Exception as exc:
        ok = False
        print(f"N56_ROW vendor_fc_m0 FAIL {type(exc).__name__}: {exc}", flush=True)

    if ok:
        print("N56_PASS", flush=True)
        return
    print("N56_FAIL", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
