# -*- coding: utf-8 -*-
"""Ad-hoc diagnostic (2026-06-13, Ubuntu): why does the P3f late rebound
return on torch 2.11 for some seeds?

Runs the full P3f onehot recipe at 400k with diag=True, dumping the
per-update error-decomposition meters (kl_fit / interference_l1 / q_rmse)
plus the NashConv(last) curve, so we can see WHICH component rises when a
seed rebounds. Compare a rebounding seed vs a well-behaved one.

Usage: python diag_rebound.py --seed 1 [--episodes 400000]
"""
import argparse
import csv
import os
import time

import numpy as np
from games import build_kuhn
from exploitability import nash_conv
from neural_acfr import NeuralACFR

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=400000)
    args = ap.parse_args()
    kuhn = build_kuhn()
    ee = max(2000, args.episodes // 200)
    nc_curve = []
    ev = lambda s, _a: nc_curve.append(None) or {"nc_last": nash_conv(kuhn, s)[0]}
    agent = NeuralACFR(kuhn, eta=0.5, tau=0.1, K_ep=2000, batch_size=16,
                       superphase=4, eta_decay=0.5, lam=1.0, distill_steps=20,
                       seed=args.seed, encoding="onehot", lr_couple=True,
                       q_anchor=True, q_replay=20000, q_batch=512, diag=True)
    t0 = time.time()
    log = agent.run(args.episodes, eval_every=ee, eval_fn=ev)
    print(f"  seed {args.seed} done in {time.time()-t0:.0f}s")

    nc_path = os.path.join(RESULTS, f"diag_rebound_nc_s{args.seed}.csv")
    with open(nc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "nc_last"])
        w.writeheader()
        for m in log:
            w.writerow({"episode": m["episode"], "nc_last": m["nc_last"]})

    diag_path = os.path.join(RESULTS, f"diag_rebound_meters_s{args.seed}.csv")
    fields = ["episode", "eta", "distill_steps", "kl_fit", "l1_fit",
              "interference_l1", "q_rmse"]
    with open(diag_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in agent.diag_log:
            w.writerow({k: d[k] for k in fields})
    nc = [m["nc_last"] for m in log]
    lq = float(np.mean(nc[3 * len(nc) // 4:]))
    print(f"  seed {args.seed}: final={nc[-1]:.4f} min={min(nc):.4f} "
          f"lastQ={lq:.4f}  -> {nc_path}, {diag_path}")


if __name__ == "__main__":
    main()
