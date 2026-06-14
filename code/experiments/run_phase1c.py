# -*- coding: utf-8 -*-
"""Phase 1c (see 09_Phase1b_结果判读.md):

  C1  Thm-3 variance scaling: steady-state noise floor vs batch size B
      (constant stepsize, schedule off). Predict log-log slope ~ -1.
  C2  the lambda spectrum's real battlefield: Leduc, model-free sampling.
      Predict: lam=0 hits a bias floor, lam=1 slow (variance),
      adaptive wins.

Usage:  python run_phase1c.py [--quick]
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
from sampling import run_sacfr, run_os_mccfr

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


def c1_floor_vs_B(kuhn, quick):
    print("== C1: noise floor vs batch size (Thm 3 scaling) ==")
    episodes = 40000 if quick else 240000
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    out, floors = [], {}
    for B in (4, 16, 64):
        _, log = run_sacfr(kuhn, episodes, eta=0.5, tau=0.1,
                           K_ep=2000, batch_size=B,
                           superphase=None,        # constant stepsize
                           lam=1.0,                # unbiased endpoint
                           eval_every=max(2000, episodes // 100),
                           eval_fn=ev)
        floor = float(np.mean([m["nc_last"] for m in log[-20:]]))
        floors[B] = floor
        out.append({"B": B, "floor": floor})
        print(f"  B={B}: floor={floor:.5f}")
    xs = np.log(np.array([4, 16, 64], dtype=float))
    ys = np.log(np.array([floors[b] for b in (4, 16, 64)]))
    A = np.vstack([xs, np.ones(3)]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    print(f"  log-log slope = {slope:.3f} (predict ~ -1)")
    save_csv("p1c_floor_vs_B.csv", out, ["B", "floor"])
    plt.figure(figsize=(6, 5))
    Bs = [4, 16, 64]
    plt.loglog(Bs, [floors[b] for b in Bs], "o-", label="measured floor")
    ref = np.array([4.0, 64.0])
    plt.loglog(ref, floors[4] * (4.0 / ref), "k--", alpha=0.5,
               label="slope -1 ref")
    plt.xlabel("batch size B"); plt.ylabel("steady-state NashConv")
    plt.title(f"C1 Kuhn: noise floor vs B (slope {slope:.2f})")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP1C1_floor_vs_B.png"), dpi=150)
    plt.close()


def c2_leduc_sampled(leduc, quick):
    print("== C2: Leduc model-free, lambda spectrum ==")
    episodes = 100000 if quick else 1000000
    ee = max(5000, episodes // 100)
    ev = lambda s, _a: {"nc_last": nash_conv(leduc, s)[0]}
    rows = {}
    t0 = time.time()
    _, avg, log = run_os_mccfr(leduc, episodes, eval_every=ee, eval_fn=ev)
    rows["osmccfr"] = log
    print(f"  os-mccfr: avg={log[-1]['nc_avg']:.4f} "
          f"last={log[-1]['nc_last']:.4f} ({time.time()-t0:.0f}s)")
    for lam in (0.0, 1.0, "adaptive"):
        _, log = run_sacfr(leduc, episodes, eta=0.5, tau=0.1,
                           K_ep=4000, batch_size=32,
                           superphase=4, eta_decay=0.5,
                           lam=lam, eval_every=ee, eval_fn=ev)
        rows[f"sacfr_lam{lam}"] = log
        print(f"  sacfr lam={lam}: last={log[-1]['nc_last']:.4f} "
              f"({time.time()-t0:.0f}s)")
    out = []
    for name, log in rows.items():
        for m in log:
            out.append({"alg": name, "episode": m["episode"],
                        "nc_last": m["nc_last"],
                        "nc_avg": m.get("nc_avg", "")})
    save_csv("p1c_leduc_sampled.csv", out,
             ["alg", "episode", "nc_last", "nc_avg"])
    plt.figure(figsize=(8, 5))
    for name, log in rows.items():
        it = [m["episode"] for m in log]
        plt.semilogy(it, [max(m["nc_last"], 1e-12) for m in log],
                     label=f"{name} (last)")
    it = [m["episode"] for m in rows["osmccfr"]]
    plt.semilogy(it, [max(m["nc_avg"], 1e-12) for m in rows["osmccfr"]],
                 "--", label="osmccfr (avg)")
    plt.xlabel("episodes"); plt.ylabel("NashConv (original units)")
    plt.title("C2 Leduc model-free: lambda spectrum vs OS-MCCFR")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP1C2_leduc.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    kuhn = build_kuhn()
    leduc = build_leduc()
    c1_floor_vs_B(kuhn, args.quick)
    c2_leduc_sampled(leduc, args.quick)
    print(f"\nALL DONE in {time.time()-t0:.0f}s. Results in "
          f"{os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
