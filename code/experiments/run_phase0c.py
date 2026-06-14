# -*- coding: utf-8 -*-
"""Phase 0c: after the B-round review (see 05_Phase0b_结果判读.md).

  C1  Leduc redo with the REAL O1 fix: advantage normalization AND
      utility scaling to [-1,1] (stability condition eta <= tau*c/L^2).
      Periodic anchor (the new default). Judge: stair-wise descent,
      clearly below the raw control, < 0.3 (stretch 0.1) at 10k iters.
  C2  overall linear last-iterate rate vs anchor period K on Kuhn
      (empirical constants for the linear-convergence main theorem).

Usage:  python run_phase0c.py [--quick]
"""
import argparse
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from games import build_kuhn, build_leduc
from exploitability import nash_conv
from algorithms import run_cfr, run_acfr

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def save_csv(name, rows, fields):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  saved {path}")


def eval_factory(game):
    def ev(sigma_last, sigma_avg):
        m = {}
        m["nc_last"], _ = nash_conv(game, sigma_last)
        if sigma_avg is not None:
            m["nc_avg"], _ = nash_conv(game, sigma_avg)
        return m
    return ev


def c1_leduc(leduc, quick):
    print(f"== C1: Leduc with utility scaling (max|u|={leduc.max_abs_u}) ==")
    ev = eval_factory(leduc)
    iters = 1000 if quick else 10000
    ee = 25 if quick else 100
    rows = {}
    t0 = time.time()

    _, _, log = run_cfr(leduc, iters, plus=True, eval_every=ee, eval_fn=ev)
    rows["cfrplus"] = log
    print(f"  cfrplus done ({time.time()-t0:.0f}s)")

    # raw control = the B4 failure mode, for contrast
    _, log, _ = run_acfr(leduc, iters, eta=0.2, tau=0.2,
                         anchor_mode="periodic", K=200,
                         normalize=False, eval_every=ee, eval_fn=ev)
    rows["raw_K200"] = log
    print(f"  raw control done ({time.time()-t0:.0f}s)")

    configs = [
        ("norm_eta0.1_K200", dict(eta=0.1, K=200)),
        ("norm_eta0.2_K200", dict(eta=0.2, K=200)),
        ("norm_eta0.2_K500", dict(eta=0.2, K=500)),
        ("norm_eta0.5_K200", dict(eta=0.5, K=200)),
        ("norm_eta0.2_K200_alt", dict(eta=0.2, K=200, alternating=True)),
    ]
    for name, kw in configs:
        _, log, _ = run_acfr(leduc, iters, tau=0.2,
                             anchor_mode="periodic",
                             normalize=True, scale_utilities=True,
                             eval_every=ee, eval_fn=ev, **kw)
        rows[name] = log
        print(f"  {name}: final = {log[-1]['nc_last']:.4f} "
              f"({time.time()-t0:.0f}s)")

    out = []
    for name, log in rows.items():
        for m in log:
            out.append({"alg": name, "iter": m["iter"],
                        "nc_last": m["nc_last"],
                        "nc_avg": m.get("nc_avg", "")})
    save_csv("c1_leduc.csv", out, ["alg", "iter", "nc_last", "nc_avg"])
    plt.figure(figsize=(8, 5))
    for name, log in rows.items():
        it = [m["iter"] for m in log]
        plt.semilogy(it, [max(m["nc_last"], 1e-12) for m in log],
                     label=f"{name} (last)")
    it = [m["iter"] for m in rows["cfrplus"]]
    plt.semilogy(it, [max(m["nc_avg"], 1e-12) for m in rows["cfrplus"]],
                 "--", label="cfrplus (avg)")
    plt.xlabel("iteration"); plt.ylabel("NashConv (original units)")
    plt.title("C1 Leduc: scaled+normalized A-CFR, periodic anchor")
    plt.legend(fontsize=7); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figC1_leduc.png"), dpi=150)
    plt.close()


def c2_rate_vs_K(kuhn, quick):
    print("== C2: overall linear rate vs anchor period K (Kuhn) ==")
    ev = eval_factory(kuhn)
    iters = 8000 if quick else 40000
    out, rows = [], {}
    for K in (25, 50, 100, 200):
        _, log, _ = run_acfr(kuhn, iters, eta=0.2, tau=0.2,
                             anchor_mode="periodic", K=K,
                             eval_every=max(10, iters // 400), eval_fn=ev)
        rows[f"K{K}"] = log
        # fit overall slope on the descending part (NashConv in [1e-10,1e-2])
        pts = [(m["iter"], m["nc_last"]) for m in log
               if 1e-10 < m["nc_last"] < 1e-2]
        slope = float("nan")
        if len(pts) > 10:
            xs = np.array([p[0] for p in pts], dtype=float)
            ys = np.log(np.array([p[1] for p in pts]))
            A = np.vstack([xs, np.ones(len(xs))]).T
            slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
        out.append({"K": K, "overall_slope": slope,
                    "final_nc": log[-1]["nc_last"]})
        print(f"  K={K}: overall slope {slope:.2e}/iter, "
              f"final {log[-1]['nc_last']:.2e}")
    save_csv("c2_rate_vs_K.csv", out, ["K", "overall_slope", "final_nc"])
    plt.figure(figsize=(8, 5))
    for name, log in rows.items():
        it = [m["iter"] for m in log]
        plt.semilogy(it, [max(m["nc_last"], 1e-13) for m in log], label=name)
    plt.xlabel("iteration"); plt.ylabel("NashConv (last iterate)")
    plt.title("C2 Kuhn: linear last-iterate rate vs anchor period "
              "(eta=0.2, tau=0.2)")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figC2_rate_vs_K.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    kuhn = build_kuhn()
    leduc = build_leduc()
    c2_rate_vs_K(kuhn, args.quick)
    c1_leduc(leduc, args.quick)
    print(f"\nALL DONE in {time.time()-t0:.0f}s. Results in "
          f"{os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
