#!/usr/bin/env python3
"""Step 24 (N5.5). Fused gather+rotate versus stock bmm.

fc_m0 is a vendor linear and is timed in one extra row. The block GEMMs are
the same kind of vendor call, so they are not rewritten. The fused kernel is
the memory-bound prologue.
"""
from __future__ import annotations

import os
import statistics
import time

import torch

LIB = os.environ["F4_LIB"]
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


def stock(x, edge, wigner):
    cat = torch.cat((x[edge[0]], x[edge[1]]), dim=2)
    return torch.bmm(wigner, cat)


def fused(x, edge, wigner, variant):
    return torch.ops.uma_n5f4.permute(x, edge, wigner, variant)


def three(run, x, edge, wigner):
    with torch.no_grad():
        run(x, edge, wigner)
    x2 = x.detach().requires_grad_(True)
    w2 = wigner.detach().requires_grad_(True)
    run(x2, edge, w2).sum().backward()


def time_pair(stock_run, fused_run, x, edge, wigner, reps=30):
    for _ in range(2):
        three(stock_run, x, edge, wigner)
        three(fused_run, x, edge, wigner)
    sync()
    left, right = [], []
    for i in range(reps):
        order = ((stock_run, left), (fused_run, right))
        if i % 2:
            order = ((fused_run, right), (stock_run, left))
        for fn, bucket in order:
            sync()
            t0 = time.perf_counter()
            three(fn, x, edge, wigner)
            sync()
            bucket.append((time.perf_counter() - t0) * 1e3)
    med_s = statistics.median(left)
    med_f = statistics.median(right)
    mad_s = statistics.median(abs(v - med_s) for v in left)
    mad_f = statistics.median(abs(v - med_f) for v in right)
    return med_s, mad_s, med_f, mad_f


def main():
    print("N55_ENTER", flush=True)
    print("N55_NOTE fc_m0 and block GEMMs stay on the vendor library", flush=True)
    dev = torch.device("xpu")
    torch.manual_seed(0)
    n, e, ell, c = 21952, 65536, 9, 128
    x = torch.randn(n, ell, c, dtype=torch.float64, device=dev)
    edge = torch.randint(0, n, (2, e), dtype=torch.int64, device=dev)
    wigner = torch.randn(e, ell, ell, dtype=torch.float64, device=dev)
    ok = True
    for vid, name in VARIANTS:
        try:
            with torch.no_grad():
                ys = stock(x, edge, wigner)
                yf = fused(x, edge, wigner, vid)
            y_abs, y_rel = rel(ys, yf)
            xs = x.detach().requires_grad_(True)
            ws = wigner.detach().requires_grad_(True)
            xf = x.detach().requires_grad_(True)
            wf = wigner.detach().requires_grad_(True)
            stock(xs, edge, ws).sum().backward()
            fused(xf, edge, wf, vid).sum().backward()
            gx_abs, gx_rel = rel(xs.grad, xf.grad)
            gw_abs, gw_rel = rel(ws.grad, wf.grad)
            good = y_rel <= REL_BAR and gx_rel <= REL_BAR and gw_rel <= REL_BAR
            print(
                f"N55_CHECK {name} y_rel={y_rel:.3e} gx_rel={gx_rel:.3e} "
                f"gw_rel={gw_rel:.3e} gx_abs={gx_abs:.3e} gw_abs={gw_abs:.3e} ok={int(good)}",
                flush=True,
            )
            ok = ok and good
        except Exception as exc:
            ok = False
            print(f"N55_CHECK {name} FAIL {type(exc).__name__}: {exc}", flush=True)

    gx_s, gw_s = torch.ops.uma_n5f4.permute_scaffold(x, edge, wigner)
    xs = x.detach().requires_grad_(True)
    ws = wigner.detach().requires_grad_(True)
    stock(xs, edge, ws).sum().backward()
    s_abs, s_rel = rel(xs.grad, gx_s)
    w_abs, w_rel = rel(ws.grad, gw_s)
    scaffold_ok = s_rel <= REL_BAR and w_rel <= REL_BAR
    print(
        f"N55_SCAFFOLD gx_rel={s_rel:.3e} gw_rel={w_rel:.3e} "
        f"gx_abs={s_abs:.3e} gw_abs={w_abs:.3e} ok={int(scaffold_ok)}",
        flush=True,
    )
    ok = ok and scaffold_ok

    xs = torch.randn(6, 3, 2, dtype=torch.float64, device=dev, requires_grad=True)
    es = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64, device=dev)
    ws = torch.randn(3, 3, 3, dtype=torch.float64, device=dev, requires_grad=True)
    try:
        grad_ok = torch.autograd.gradcheck(
            lambda a, b: torch.ops.uma_n5f4.permute(a, es, b, 0),
            (xs, ws),
            eps=1e-6,
            atol=1e-8,
            rtol=1e-6,
        )
    except Exception as exc:
        grad_ok = False
        print(f"N55_GRADCHECK FAIL {type(exc).__name__}: {exc}", flush=True)
    else:
        print(f"N55_GRADCHECK ok={int(bool(grad_ok))}", flush=True)
    ok = ok and bool(grad_ok)

    for vid, name in VARIANTS:
        try:
            med_s, mad_s, med_f, mad_f = time_pair(
                lambda a, e_, w: stock(a, e_, w),
                lambda a, e_, w, _v=vid: fused(a, e_, w, _v),
                x, edge, wigner,
            )
        except Exception as exc:
            ok = False
            print(f"N55_ROW {name} FAIL {type(exc).__name__}: {exc}", flush=True)
            continue
        print(
            f"N55_ROW {name} stock_ms={med_s:.4f} mad={mad_s:.4f} "
            f"fused_ms={med_f:.4f} mad={mad_f:.4f} gain_ms={med_s - med_f:.4f}",
            flush=True,
        )

    # Vendor fc_m0, 768 -> 640, on the m=0 slice of the rotated message.
    weight = torch.randn(640, 768, dtype=torch.float64, device=dev)
    bias = torch.randn(640, dtype=torch.float64, device=dev)

    def with_gemm(permute):
        def run(a, e_, w):
            y = permute(a, e_, w)
            flat = y[:, :3, :].reshape(y.shape[0], -1)
            return torch.nn.functional.linear(flat, weight, bias)

        return run

    try:
        med_s, mad_s, med_f, mad_f = time_pair(
            with_gemm(stock), with_gemm(lambda a, e_, w: fused(a, e_, w, 0)), x, edge, wigner
        )
        print(
            f"N55_ROW vendor_fc_m0 stock_ms={med_s:.4f} mad={mad_s:.4f} "
            f"fused_ms={med_f:.4f} mad={mad_f:.4f} gain_ms={med_s - med_f:.4f}",
            flush=True,
        )
    except Exception as exc:
        ok = False
        print(f"N55_ROW vendor_fc_m0 FAIL {type(exc).__name__}: {exc}", flush=True)

    if ok:
        print("N55_PASS", flush=True)
        return
    print("N55_FAIL", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
