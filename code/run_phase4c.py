# -*- coding: utf-8 -*-
"""Phase 4c: Leduc neural diagnosis (the recipe did NOT transfer).

P4a-A1 features: min 1.449@32k (vs tabular 1.394), degrades to ~2.4
after 120k (4th superphase, eta<0.03). Two separable problems:
  A) mid-run level is weak (never clearly below tabular) -- new; on
     Kuhn the neural version tracked tabular easily.
  B) late degradation under deep annealing -- familiar shape.

Instrumented 100k arms (diag on: qRMSE vs exact Q^sigma, distill KL,
interference), one hypothesis each:
  base    current recipe (diagnosis baseline)
  qplus   Q training x3 (q_steps 120, q_batch 1024) -- Q quality
  lrdown  lr,q_lr 1e-2 -> 3e-3 -- feature-net interference / step scale
  wide    width 128 -> 256 -- capacity

  python run_phase4c.py --exp base|qplus|lrdown|wide [--quick]
"""
import argparse
import csv
import os
import time

import numpy as np

from games import build_leduc
from exploitability import nash_conv
from neural_acfr import NeuralACFR

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

ARMS = {
    "base": dict(),
    "qplus": dict(q_steps=120, q_batch=1024),
    "lrdown": dict(lr=3e-3, q_lr=3e-3),
    "wide": dict(width=256),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="base", choices=list(ARMS))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    g = build_leduc()
    episodes = 20000 if args.quick else 100000
    ee = max(2000, episodes // 50)
    ev = lambda s, _a: {"nc_last": nash_conv(g, s)[0]}
    kw = dict(eta=0.5, tau=0.1, K_ep=2000, batch_size=16, superphase=4,
              eta_decay=0.5, lam=1.0, distill_steps=20, width=128,
              seed=0, encoding="features", lr_couple=True, q_anchor=True,
              q_replay=20000, q_batch=512, diag=True)
    kw.update(ARMS[args.exp])
    t0 = time.time()
    agent = NeuralACFR(g, **kw)
    log = agent.run(episodes, eval_every=ee, eval_fn=ev)
    nc = [m["nc_last"] for m in log]
    d = agent.diag_log
    n = len(d)
    win = lambda lo, hi, key: float(np.mean(
        [r[key] for r in d[int(n*lo):max(int(n*hi), int(n*lo)+1)]]))
    print(f"  {args.exp}: final={nc[-1]:.4f}, min={min(nc):.4f}, "
          f"qRMSE {win(0.05,0.15,'q_rmse'):.3f}->{win(0.85,1.0,'q_rmse'):.3f}, "
          f"kl_late={win(0.85,1.0,'kl_fit'):.5f}, "
          f"interf_late={win(0.85,1.0,'interference_l1'):.4f} "
          f"({time.time()-t0:.0f}s)")
    # dump curve + diag
    with open(os.path.join(RESULTS, f"p4c_{args.exp}_leduc.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alg", "episode", "nc_last"])
        w.writeheader()
        for m in log:
            w.writerow({"alg": args.exp, "episode": m["episode"],
                        "nc_last": m["nc_last"]})
    with open(os.path.join(RESULTS, f"p4c_{args.exp}_diag.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(d[0].keys()))
        w.writeheader()
        for r in d:
            w.writerow(r)
    print("  saved p4c csvs")


if __name__ == "__main__":
    main()
