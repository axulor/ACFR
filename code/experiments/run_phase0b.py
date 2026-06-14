# -*- coding: utf-8 -*-
"""Phase 0b: follow-up experiments after the first results review
(see 04_Phase0_结果判读.md).

  B1  contraction-rate table: measured log-slope vs theory -ln(1+eta*tau)
  B2  gap floor vs tau down to tau=0.01 (nail the O(tau) slope)
  B3  adaptive anchor on Kuhn: drive last-iterate NashConv below 1e-3
  B4  Leduc: normalized A-CFR (the O1 fix) vs unnormalized vs CFR+

Usage:  python run_phase0b.py [--quick]
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
from algorithms import run_cfr, run_acfr, strategy_l1_distance

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


def b1_contraction(kuhn, quick):
    """Measured per-iter log-slope vs theoretical exponent."""
    print("== B1: contraction rate vs theory ==")
    configs = [(0.5, 1.0), (0.2, 0.5), (0.1, 0.2), (0.5, 0.5)]
    out = []
    plt.figure(figsize=(8, 5))
    for eta, tau in configs:
        iters = 400 if not quick else 200
        sigma_star, _, snaps = run_acfr(kuhn, iters, eta=eta, tau=tau,
                                        anchor_mode="fixed",
                                        snapshot_every=1)
        ts, ds = [], []
        for t, s in snaps:
            d = strategy_l1_distance(s, sigma_star)
            if d > 1e-12:
                ts.append(t); ds.append(d)
        # fit on early window: skip first 5 iters (transient), use next 40
        xs = np.array(ts[5:45], dtype=float)
        ys = np.log(np.array(ds[5:45]))
        slope = float("nan")
        if len(xs) > 10:
            A = np.vstack([xs, np.ones(len(xs))]).T
            slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
        theory = -np.log(1 + eta * tau)
        out.append({"eta": eta, "tau": tau,
                    "measured_slope": slope, "theory_bound": theory,
                    "ratio": slope / theory if theory else float("nan")})
        print(f"  eta={eta} tau={tau}: measured {slope:+.4f} | "
              f"bound {theory:+.4f} | ratio {slope/theory:.2f}")
        plt.semilogy(ts, ds, label=f"eta={eta}, tau={tau} "
                                   f"(slope {slope:.3f}/bound {theory:.3f})")
    save_csv("b1_contraction.csv", out,
             ["eta", "tau", "measured_slope", "theory_bound", "ratio"])
    plt.xlabel("iteration"); plt.ylabel("L1 distance to fixed point")
    plt.title("B1 Kuhn fixed anchor: linear convergence vs theory")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figB1_contraction.png"), dpi=150)
    plt.close()


def b2_tau_floor(kuhn, quick):
    print("== B2: gap floor down to tau=0.01 ==")
    eta = 0.5
    taus = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ev = eval_factory(kuhn)
    out, floors = [], {}
    for tau in taus:
        # ensure convergence: ~30/(eta*tau) iterations
        iters = int(min(60000, max(5000, 30.0 / (eta * tau))))
        if quick:
            iters = min(iters, 8000)
        _, log, _ = run_acfr(kuhn, iters, eta=eta, tau=tau,
                             anchor_mode="fixed", eval_every=max(10, iters // 100),
                             eval_fn=ev)
        floor = float(np.mean([m["nc_last"] for m in log[-5:]]))
        floors[tau] = floor
        out.append({"eta": eta, "tau": tau, "iters": iters, "floor": floor})
        print(f"  tau={tau}: floor={floor:.6f} ({iters} iters)")
    # log-log slope on the three smallest taus
    xs = np.log(np.array(taus[:3])); ys = np.log(np.array([floors[t] for t in taus[:3]]))
    A = np.vstack([xs, np.ones(len(xs))]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    print(f"  small-tau log-log slope = {slope:.3f} (predict -> 1)")
    save_csv("b2_tau_floor.csv", out, ["eta", "tau", "iters", "floor"])
    plt.figure(figsize=(7, 5))
    plt.loglog(taus, [floors[t] for t in taus], "o-", label="floor")
    ref = np.array([taus[0], taus[-1]])
    plt.loglog(ref, ref * floors[1.0], "k--", alpha=0.5, label="slope 1 ref")
    plt.xlabel("tau"); plt.ylabel("NashConv floor")
    plt.title(f"B2: floor vs tau (small-tau slope {slope:.2f})")
    plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figB2_tau_floor.png"), dpi=150)
    plt.close()


def b3_adaptive_anchor(kuhn, quick):
    print("== B3: adaptive anchor, target NashConv < 1e-3 ==")
    ev = eval_factory(kuhn)
    iters = 8000 if quick else 60000
    rows = {}
    for name, kw in [
        ("adaptive", dict(anchor_mode="adaptive", tol0=0.02)),
        ("adaptive_aggr", dict(anchor_mode="adaptive", tol0=0.05,
                               tol_decay=0.7)),
        ("periodic_K50", dict(anchor_mode="periodic", K=50)),
        ("ema_1/50", dict(anchor_mode="ema", K=50)),
    ]:
        _, log, _ = run_acfr(kuhn, iters, eta=0.2, tau=0.2,
                             eval_every=max(10, iters // 200), eval_fn=ev, **kw)
        rows[name] = log
        print(f"  {name}: final NashConv = {log[-1]['nc_last']:.2e}")
    out = []
    for name, log in rows.items():
        for m in log:
            out.append({"alg": name, "iter": m["iter"],
                        "nc_last": m["nc_last"]})
    save_csv("b3_adaptive_anchor.csv", out, ["alg", "iter", "nc_last"])
    plt.figure(figsize=(8, 5))
    for name, log in rows.items():
        it = [m["iter"] for m in log]
        plt.semilogy(it, [max(m["nc_last"], 1e-12) for m in log], label=name)
    plt.xlabel("iteration"); plt.ylabel("NashConv (last iterate)")
    plt.title("B3 Kuhn: adaptive anchor (eta=0.2, tau=0.2, no annealing)")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figB3_adaptive.png"), dpi=150)
    plt.close()


def b4_leduc_normalized(leduc, quick):
    print("== B4: Leduc, the O1 fix (normalized counterfactual values) ==")
    ev = eval_factory(leduc)
    iters = 400 if quick else 3000
    rows = {}
    t0 = time.time()
    _, _, log = run_cfr(leduc, iters, plus=True, eval_every=25, eval_fn=ev)
    rows["cfrplus"] = log
    print(f"  cfrplus done ({time.time()-t0:.0f}s)")
    # unnormalized control (the failure mode seen in E5)
    _, log, _ = run_acfr(leduc, iters, eta=0.2, tau=0.2,
                         anchor_mode="adaptive", tol0=0.02,
                         normalize=False, eval_every=25, eval_fn=ev)
    rows["acfr_raw"] = log
    print(f"  acfr_raw done ({time.time()-t0:.0f}s)")
    for eta in (0.1, 0.3, 1.0):
        _, log, _ = run_acfr(leduc, iters, eta=eta, tau=0.2,
                             anchor_mode="adaptive", tol0=0.02,
                             normalize=True, eval_every=25, eval_fn=ev)
        rows[f"acfr_norm_eta{eta}"] = log
        print(f"  acfr_norm eta={eta} done ({time.time()-t0:.0f}s), "
              f"final = {log[-1]['nc_last']:.4f}")
    out = []
    for name, log in rows.items():
        for m in log:
            out.append({"alg": name, "iter": m["iter"],
                        "nc_last": m["nc_last"],
                        "nc_avg": m.get("nc_avg", "")})
        print(f"  {name}: final last-iter NashConv = {log[-1]['nc_last']:.5f}")
    save_csv("b4_leduc_norm.csv", out, ["alg", "iter", "nc_last", "nc_avg"])
    plt.figure(figsize=(8, 5))
    for name, log in rows.items():
        it = [m["iter"] for m in log]
        plt.semilogy(it, [max(m["nc_last"], 1e-12) for m in log],
                     label=f"{name} (last)")
    it = [m["iter"] for m in rows["cfrplus"]]
    plt.semilogy(it, [max(m["nc_avg"], 1e-12) for m in rows["cfrplus"]],
                 "--", label="cfrplus (avg)")
    plt.xlabel("iteration"); plt.ylabel("NashConv")
    plt.title("B4 Leduc: normalized A-CFR vs raw vs CFR+")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figB4_leduc.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    q = args.quick
    t0 = time.time()
    kuhn = build_kuhn()
    leduc = build_leduc()
    b1_contraction(kuhn, q)
    b2_tau_floor(kuhn, q)
    b3_adaptive_anchor(kuhn, q)
    b4_leduc_normalized(leduc, q)
    print(f"\nALL DONE in {time.time()-t0:.0f}s. Results in "
          f"{os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
