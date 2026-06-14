# -*- coding: utf-8 -*-
"""Phase 0d: Leduc outer-rate diagnosis (see 06_Phase0c_结果判读.md).

Grid over (tau, eta) with the anchor period set by the C2 rule
K = ceil(7 / ln(1+eta*tau))  (inner loop solved to ~3 decades), phase-aligned
evaluation, and a per-phase contraction factor table. Goal: separate
"phase budget / K misconfiguration" from "game conditioning (sharpness)".

Usage:  python run_phase0d.py [--quick]
Runtime: ~60-90 min full, ~8 min quick.
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from games import build_leduc
from exploitability import nash_conv
from algorithms import run_acfr

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


def per_phase_factor(log, K, nc_hi=1.0, nc_lo=1e-9):
    """Fit per-phase contraction factor on the descending segment."""
    pts = [(m["iter"], m["nc_last"]) for m in log
           if nc_lo < m["nc_last"] < nc_hi and m["iter"] > 2 * K]
    if len(pts) < 6:
        return float("nan")
    xs = np.array([p[0] / K for p in pts], dtype=float)   # phase index
    ys = np.log(np.array([p[1] for p in pts]))
    A = np.vstack([xs, np.ones(len(xs))]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(np.exp(slope))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    iters = 1500 if args.quick else 15000
    leduc = build_leduc()
    ev = lambda s, _a: {"nc_last": nash_conv(leduc, s)[0]}

    grid = [(0.05, 1.0), (0.1, 0.5), (0.1, 1.0),
            (0.2, 0.5), (0.2, 1.0), (0.5, 0.5)]
    rows, table = {}, []
    t0 = time.time()
    for tau, eta in grid:
        K = max(10, math.ceil(7.0 / math.log(1.0 + eta * tau)))
        name = f"tau{tau}_eta{eta}_K{K}"
        _, log, _ = run_acfr(leduc, iters, eta=eta, tau=tau,
                             anchor_mode="periodic", K=K,
                             normalize=True, scale_utilities=True,
                             eval_every=K, eval_fn=ev)
        rows[name] = log
        fac = per_phase_factor(log, K)
        n_phases = iters // K
        table.append({"tau": tau, "eta": eta, "K": K,
                      "phases": n_phases,
                      "per_phase_factor": fac,
                      "final_nc": log[-1]["nc_last"]})
        print(f"  {name}: phases={n_phases}, factor={fac:.4f}, "
              f"final={log[-1]['nc_last']:.4f} ({time.time()-t0:.0f}s)")

    save_csv("d1_leduc_grid.csv", table,
             ["tau", "eta", "K", "phases", "per_phase_factor", "final_nc"])
    out = []
    for name, log in rows.items():
        for m in log:
            out.append({"alg": name, "iter": m["iter"],
                        "nc_last": m["nc_last"]})
    save_csv("d1_leduc_curves.csv", out, ["alg", "iter", "nc_last"])

    plt.figure(figsize=(8, 5))
    for name, log in rows.items():
        it = [m["iter"] for m in log]
        plt.semilogy(it, [max(m["nc_last"], 1e-12) for m in log],
                     label=name)
    plt.xlabel("iteration"); plt.ylabel("NashConv (last iterate)")
    plt.title("D1 Leduc: (tau, eta) grid, K by the C2 rule")
    plt.legend(fontsize=7); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figD1_leduc_grid.png"), dpi=150)
    plt.close()
    print(f"\nALL DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
