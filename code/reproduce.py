# -*- coding: utf-8 -*-
"""Phase 0: validate the two core theoretical predictions of A-CFR.

  (a) fixed anchor  -> last-iterate linear convergence to the regularized
      equilibrium; NashConv floor of order O(tau).
  (b) moving anchor -> NashConv of the LAST iterate -> 0 without any
      temperature annealing (tau stays constant).
  (c) baselines CFR / CFR+ : average converges, last iterate oscillates.

Usage:  python run_phase0.py [--quick]
Outputs: ../results/*.csv, ../results/*.png
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from games import build_kuhn, build_leduc
from exploitability import nash_conv, profile_value
from algorithms import (run_cfr, run_acfr, uniform_strategy,
                        strategy_l1_distance)

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


# ----------------------------------------------------------- sanity checks

def sanity(quick):
    print("== sanity checks ==")
    kuhn = build_kuhn()
    n_iset = len(kuhn.infosets)
    assert n_iset == 12, f"Kuhn should have 12 infosets, got {n_iset}"
    iters = 2000 if quick else 5000
    _, avg, _ = run_cfr(kuhn, iters, plus=False, eval_fn=None)
    val = profile_value(kuhn, avg)
    nc, _ = nash_conv(kuhn, avg)
    print(f"  Kuhn CFR({iters}): avg value={val:+.4f} (theory -1/18={-1/18:+.4f}), "
          f"NashConv={nc:.4f}")
    assert abs(val - (-1.0 / 18.0)) < 0.02, "Kuhn game value check failed"
    assert nc < 0.05, "Kuhn CFR convergence check failed"
    leduc = build_leduc()
    print(f"  Leduc built: {len(leduc.infosets)} infosets "
          f"(expected order of ~900)")
    nc_u, _ = nash_conv(leduc, uniform_strategy(leduc))
    print(f"  Leduc uniform NashConv = {nc_u:.4f} (must be > 0)")
    assert nc_u > 0
    print("  all sanity checks passed\n")
    return kuhn, leduc


# ----------------------------------------------------------- experiments

def e1_kuhn_main(kuhn, iters):
    """Main comparison on Kuhn: last-iterate behaviour."""
    print("== E1: Kuhn last-iterate comparison ==")
    ev = eval_factory(kuhn)
    rows = {}

    _, _, log = run_cfr(kuhn, iters, plus=False, eval_every=10, eval_fn=ev)
    rows["cfr"] = log
    _, _, log = run_cfr(kuhn, iters, plus=True, eval_every=10, eval_fn=ev)
    rows["cfrplus"] = log
    _, log, _ = run_acfr(kuhn, iters, eta=0.2, tau=0.2,
                         anchor_mode="fixed", eval_every=10, eval_fn=ev)
    rows["acfr_fixed"] = log
    _, log, _ = run_acfr(kuhn, iters, eta=0.2, tau=0.2,
                         anchor_mode="periodic", K=100,
                         eval_every=10, eval_fn=ev)
    rows["acfr_K100"] = log
    _, log, _ = run_acfr(kuhn, iters, eta=0.2, tau=0.2,
                         anchor_mode="ema", K=100,
                         eval_every=10, eval_fn=ev)
    rows["acfr_ema"] = log

    out = []
    for name, log in rows.items():
        for m in log:
            out.append({"alg": name, "iter": m["iter"],
                        "nc_last": m["nc_last"],
                        "nc_avg": m.get("nc_avg", "")})
    save_csv("e1_kuhn_main.csv", out,
             ["alg", "iter", "nc_last", "nc_avg"])

    plt.figure(figsize=(8, 5))
    for name, log in rows.items():
        it = [m["iter"] for m in log]
        plt.semilogy(it, [max(m["nc_last"], 1e-12) for m in log],
                     label=f"{name} (last)")
    for name in ("cfr", "cfrplus"):
        it = [m["iter"] for m in rows[name]]
        plt.semilogy(it, [max(m["nc_avg"], 1e-12) for m in rows[name]],
                     "--", label=f"{name} (avg)")
    plt.xlabel("iteration"); plt.ylabel("NashConv")
    plt.title("Kuhn: last-iterate NashConv (A-CFR) vs CFR/CFR+")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig1_kuhn_lastiter.png"), dpi=150)
    plt.close()
    print("  fig1 saved")


def e2_linear_rate(kuhn, iters):
    """Fixed anchor: distance to regularized equilibrium should decay
    linearly (straight line on semilog)."""
    print("== E2: fixed-anchor linear rate ==")
    eta, tau = 0.5, 1.0
    sigma_star, _, snaps = run_acfr(kuhn, iters, eta=eta, tau=tau,
                                    anchor_mode="fixed",
                                    eval_fn=None, snapshot_every=max(1, iters // 600))
    ts, ds = [], []
    for t, s in snaps:
        ts.append(t)
        ds.append(strategy_l1_distance(s, sigma_star))
    # fit slope on the middle of the run (avoid proxy-saturation tail)
    lo, hi = int(len(ts) * 0.05), int(len(ts) * 0.6)
    xs = np.array(ts[lo:hi], dtype=float)
    ys = np.array([max(d, 1e-300) for d in ds[lo:hi]])
    mask = ys > 1e-12
    slope = float("nan")
    if mask.sum() > 10:
        A = np.vstack([xs[mask], np.ones(mask.sum())]).T
        slope, _ = np.linalg.lstsq(A, np.log(ys[mask]), rcond=None)[0]
    pred = -np.log(1 + eta * tau)  # naive per-iter contraction exponent bound
    print(f"  fitted log-slope = {slope:.5f} per iter "
          f"(naive bound exponent -log(1+eta*tau) = {pred:.5f})")
    save_csv("e2_linear_rate.csv",
             [{"iter": t, "l1_dist": d} for t, d in zip(ts, ds)],
             ["iter", "l1_dist"])
    plt.figure(figsize=(7, 5))
    plt.semilogy(ts, [max(d, 1e-16) for d in ds])
    plt.xlabel("iteration"); plt.ylabel("L1 distance to sigma* (proxy)")
    plt.title(f"Kuhn, fixed anchor (eta={eta}, tau={tau}): "
              f"linear convergence, slope={slope:.4f}")
    plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig2_linear_rate.png"), dpi=150)
    plt.close()
    print("  fig2 saved")


def e3_tau_floor(kuhn, iters):
    """Fixed anchor: NashConv floor should scale ~ O(tau)."""
    print("== E3: gap floor vs tau ==")
    ev = eval_factory(kuhn)
    out = []
    floors = {}
    for eta in (0.1, 0.5):
        for tau in (0.05, 0.1, 0.2, 0.5, 1.0):
            _, log, _ = run_acfr(kuhn, iters, eta=eta, tau=tau,
                                 anchor_mode="fixed",
                                 eval_every=10, eval_fn=ev)
            floor = float(np.mean([m["nc_last"] for m in log[-10:]]))
            floors[(eta, tau)] = floor
            out.append({"eta": eta, "tau": tau, "floor": floor})
            print(f"  eta={eta} tau={tau}: floor={floor:.5f}")
    save_csv("e3_tau_floor.csv", out, ["eta", "tau", "floor"])
    plt.figure(figsize=(7, 5))
    for eta in (0.1, 0.5):
        taus = [0.05, 0.1, 0.2, 0.5, 1.0]
        plt.loglog(taus, [max(floors[(eta, t)], 1e-12) for t in taus],
                   "o-", label=f"eta={eta}")
    ref = np.array([0.05, 1.0])
    plt.loglog(ref, ref * floors[(0.5, 1.0)], "k--", alpha=0.5,
               label="slope 1 reference")
    plt.xlabel("tau"); plt.ylabel("NashConv floor (last iterate)")
    plt.title("Kuhn, fixed anchor: gap floor vs tau (predict O(tau))")
    plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig3_tau_floor.png"), dpi=150)
    plt.close()
    print("  fig3 saved")


def e4_anchor_schedule(kuhn, iters):
    """Moving anchor: K sweep + EMA; NashConv(last) should go to ~0."""
    print("== E4: anchor schedule sweep ==")
    ev = eval_factory(kuhn)
    rows = {}
    for K in (10, 50, 200):
        _, log, _ = run_acfr(kuhn, iters, eta=0.2, tau=0.2,
                             anchor_mode="periodic", K=K,
                             eval_every=10, eval_fn=ev)
        rows[f"periodic_K{K}"] = log
    for K in (50, 200):
        _, log, _ = run_acfr(kuhn, iters, eta=0.2, tau=0.2,
                             anchor_mode="ema", K=K,
                             eval_every=10, eval_fn=ev)
        rows[f"ema_1/{K}"] = log
    out = []
    for name, log in rows.items():
        for m in log:
            out.append({"alg": name, "iter": m["iter"],
                        "nc_last": m["nc_last"]})
        print(f"  {name}: final NashConv = {log[-1]['nc_last']:.6f}")
    save_csv("e4_anchor_schedule.csv", out, ["alg", "iter", "nc_last"])
    plt.figure(figsize=(8, 5))
    for name, log in rows.items():
        it = [m["iter"] for m in log]
        plt.semilogy(it, [max(m["nc_last"], 1e-12) for m in log], label=name)
    plt.xlabel("iteration"); plt.ylabel("NashConv (last iterate)")
    plt.title("Kuhn, moving anchor (eta=0.2, tau=0.2 fixed, no annealing)")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig4_anchor_schedule.png"), dpi=150)
    plt.close()
    print("  fig4 saved")


def e5_leduc(leduc, iters):
    """Leduc: A-CFR (best schedule from E4) vs CFR / CFR+."""
    print("== E5: Leduc ==")
    ev = eval_factory(leduc)
    rows = {}
    t0 = time.time()
    _, _, log = run_cfr(leduc, iters, plus=False, eval_every=25, eval_fn=ev)
    rows["cfr"] = log
    print(f"  cfr done ({time.time()-t0:.0f}s)")
    _, _, log = run_cfr(leduc, iters, plus=True, eval_every=25, eval_fn=ev)
    rows["cfrplus"] = log
    _, log, _ = run_acfr(leduc, iters, eta=0.2, tau=0.2,
                         anchor_mode="periodic", K=100,
                         eval_every=25, eval_fn=ev)
    rows["acfr_K100"] = log
    _, log, _ = run_acfr(leduc, iters, eta=0.2, tau=0.2,
                         anchor_mode="ema", K=100,
                         eval_every=25, eval_fn=ev)
    rows["acfr_ema"] = log
    out = []
    for name, log in rows.items():
        for m in log:
            out.append({"alg": name, "iter": m["iter"],
                        "nc_last": m["nc_last"],
                        "nc_avg": m.get("nc_avg", "")})
        print(f"  {name}: final last-iter NashConv = {log[-1]['nc_last']:.5f}")
    save_csv("e5_leduc.csv", out, ["alg", "iter", "nc_last", "nc_avg"])
    plt.figure(figsize=(8, 5))
    for name, log in rows.items():
        it = [m["iter"] for m in log]
        plt.semilogy(it, [max(m["nc_last"], 1e-12) for m in log],
                     label=f"{name} (last)")
    for name in ("cfr", "cfrplus"):
        it = [m["iter"] for m in rows[name]]
        plt.semilogy(it, [max(m["nc_avg"], 1e-12) for m in rows[name]],
                     "--", label=f"{name} (avg)")
    plt.xlabel("iteration"); plt.ylabel("NashConv")
    plt.title("Leduc: last-iterate NashConv")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig5_leduc.png"), dpi=150)
    plt.close()
    print("  fig5 saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke test with reduced iterations")
    args = ap.parse_args()
    q = args.quick
    t0 = time.time()
    kuhn, leduc = sanity(q)
    e1_kuhn_main(kuhn, 2000 if q else 20000)
    e2_linear_rate(kuhn, 3000 if q else 60000)
    e3_tau_floor(kuhn, 1000 if q else 10000)
    e4_anchor_schedule(kuhn, 2000 if q else 30000)
    e5_leduc(leduc, 300 if q else 3000)
    print(f"\nALL DONE in {time.time()-t0:.0f}s. "
          f"Results in {os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
