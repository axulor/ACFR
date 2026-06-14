# -*- coding: utf-8 -*-
"""Phase 1d (see 10_项目总结 §五):

  D1  C1 redo with the right yardstick: squared L2 distance to a
      reference solution -> predict slope -1 vs batch size B
      (NashConv mixes in a sqrt and a deterministic offset; that was
      the apparent "slope -0.31" of P1c).
  D2  Leduc rematch with the FIXED adaptive-lambda signal
      (correction-magnitude EMA instead of TD self-consistency).

Usage:  python run_phase1d.py [--quick]
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
from algorithms import run_acfr
from sampling import run_sacfr

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


def d1_floor_vs_B(kuhn, quick):
    print("== D1: squared-distance floor vs B (slope -1 test) ==")
    # reference solution: long scheduled full-feedback run -> ~exact Nash
    ref, _, _ = run_acfr(kuhn, 4000 if quick else 40000, eta=0.2, tau=0.2,
                         anchor_mode="periodic", K=50,
                         superphase=5, eta_decay=0.5,
                         scale_utilities=True, normalize=True)

    def ev(s, _a):
        d2 = sum(float(((s[I] - ref[I]) ** 2).sum()) for I in s)
        return {"dist2": d2, "nc_last": nash_conv(kuhn, s)[0]}

    episodes = 40000 if quick else 240000
    out, floors = [], {}
    for B in (4, 16, 64, 256):
        _, log = run_sacfr(kuhn, episodes, eta=0.5, tau=0.1,
                           K_ep=2000, batch_size=B,
                           superphase=None, lam=1.0,
                           eval_every=max(2000, episodes // 100),
                           eval_fn=ev)
        floor = float(np.mean([m["dist2"] for m in log[-20:]]))
        nc = float(np.mean([m["nc_last"] for m in log[-20:]]))
        floors[B] = floor
        out.append({"B": B, "dist2_floor": floor, "nc_floor": nc})
        print(f"  B={B}: dist2 floor={floor:.5f}, NashConv floor={nc:.5f}")
    Bs = [4, 16, 64, 256]
    xs = np.log(np.array(Bs, dtype=float))
    ys = np.log(np.array([floors[b] for b in Bs]))
    A = np.vstack([xs, np.ones(len(Bs))]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    print(f"  dist2 log-log slope = {slope:.3f} (predict ~ -1)")
    save_csv("p1d_floor_vs_B.csv", out, ["B", "dist2_floor", "nc_floor"])
    plt.figure(figsize=(6, 5))
    plt.loglog(Bs, [floors[b] for b in Bs], "o-", label="dist^2 floor")
    ref_x = np.array([4.0, 256.0])
    plt.loglog(ref_x, floors[4] * (4.0 / ref_x), "k--", alpha=0.5,
               label="slope -1 ref")
    plt.xlabel("batch size B"); plt.ylabel("squared L2 distance floor")
    plt.title(f"D1 Kuhn: variance scaling (slope {slope:.2f})")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP1D1_floor_vs_B.png"), dpi=150)
    plt.close()


def d2_leduc_adaptive(leduc, quick):
    print("== D2: Leduc rematch, fixed adaptive-lambda signal ==")
    episodes = 100000 if quick else 1000000
    ee = max(5000, episodes // 100)
    ev = lambda s, _a: {"nc_last": nash_conv(leduc, s)[0]}
    rows = {}
    t0 = time.time()
    for name, lam, kappa in (("lam1.0", 1.0, None),
                             ("adaptive_k4", "adaptive", 4.0),
                             ("adaptive_k8", "adaptive", 8.0)):
        kw = dict(kappa=kappa) if kappa else {}
        _, log = run_sacfr(leduc, episodes, eta=0.5, tau=0.1,
                           K_ep=4000, batch_size=32,
                           superphase=4, eta_decay=0.5,
                           lam=lam, eval_every=ee, eval_fn=ev, **kw)
        rows[name] = log
        print(f"  {name}: last={log[-1]['nc_last']:.4f} "
              f"({time.time()-t0:.0f}s)")
    out = []
    for name, log in rows.items():
        for m in log:
            out.append({"alg": name, "episode": m["episode"],
                        "nc_last": m["nc_last"]})
    save_csv("p1d_leduc_adaptive.csv", out, ["alg", "episode", "nc_last"])
    plt.figure(figsize=(8, 5))
    for name, log in rows.items():
        it = [m["episode"] for m in log]
        plt.semilogy(it, [max(m["nc_last"], 1e-12) for m in log],
                     label=name)
    plt.xlabel("episodes"); plt.ylabel("NashConv (last iterate)")
    plt.title("D2 Leduc: fixed adaptive lambda vs lam=1")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP1D2_leduc.png"), dpi=150)
    plt.close()


def d3_outer_rate_isolation(leduc, quick):
    """O-c diagnostic (11_P2 doc §8(v)): with K = 3x the C2 rule the inner
    residual is ~1e-9 -- if the per-phase factor still scales ~1/tau, the
    tau-dependence belongs to the outer (EB) structure, not to inner
    inexactness."""
    import math
    print("== D3: outer-rate isolation (huge K, tau sweep) ==")
    iters = 3000 if quick else 20000
    ev = lambda s, _a: {"nc_last": nash_conv(leduc, s)[0]}
    out = []
    for tau in (0.1, 0.2, 0.5):
        eta = 0.5
        K = 3 * max(10, math.ceil(7.0 / math.log(1.0 + eta * tau)))
        _, log, _ = run_acfr(leduc, iters, eta=eta, tau=tau,
                             anchor_mode="periodic", K=K,
                             normalize=True, scale_utilities=True,
                             eval_every=K, eval_fn=ev)
        pts = [(m["iter"] / K, m["nc_last"]) for m in log
               if 1e-9 < m["nc_last"] < 1.0 and m["iter"] > 2 * K]
        fac = float("nan")
        if len(pts) > 5:
            xs = np.array([p[0] for p in pts])
            ys = np.log(np.array([p[1] for p in pts]))
            A = np.vstack([xs, np.ones(len(xs))]).T
            sl, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
            fac = float(np.exp(sl))
        out.append({"tau": tau, "K": K, "phases": iters // K,
                    "per_phase_factor": fac,
                    "final_nc": log[-1]["nc_last"]})
        print(f"  tau={tau}, K={K}: factor={fac:.4f}, "
              f"final={log[-1]['nc_last']:.4f}")
    save_csv("p1d_outer_isolation.csv", out,
             ["tau", "K", "phases", "per_phase_factor", "final_nc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--oc", action="store_true",
                    help="also run the D3 outer-rate isolation (slow)")
    args = ap.parse_args()
    t0 = time.time()
    kuhn = build_kuhn()
    leduc = build_leduc()
    d1_floor_vs_B(kuhn, args.quick)
    d2_leduc_adaptive(leduc, args.quick)
    if args.oc:
        d3_outer_rate_isolation(leduc, args.quick)
    print(f"\nALL DONE in {time.time()-t0:.0f}s. Results in "
          f"{os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
