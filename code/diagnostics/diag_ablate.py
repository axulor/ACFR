# -*- coding: utf-8 -*-
"""Ablation for the torch-2.11 late rebound (2026-06-13, Ubuntu).

Diagnosis from diag_rebound: in rebounding seeds, policy-side
interference is killed by lr_couple (->0), but q_rmse is pinned high
and never recovers; nc tracks the high q_rmse. Hypothesis: q_anchor
(bootstrapping Q targets from the frozen anchor snapshot) self-reinforces
a biased Q. This script re-runs a given seed with individual Q
stabilizers toggled off to see which restores q_rmse recovery / kills
the rebound.

Usage: python diag_ablate.py --seed 1 --variant no_qanchor|no_qreplay|no_qboth
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

# each variant is an override on top of the full P3f recipe (BASE).
BASE = dict(eta=0.5, tau=0.1, K_ep=2000, batch_size=16, superphase=4,
            eta_decay=0.5, lam=1.0, distill_steps=20, encoding="onehot",
            lr_couple=True, q_anchor=True, q_replay=20000, q_batch=512)
VARIANTS = {
    "no_qanchor": dict(q_anchor=False),
    "no_qreplay": dict(q_replay=0),
    "no_qboth":   dict(q_anchor=False, q_replay=0),
    "full":       dict(),
    "big_batch":  dict(batch_size=64),   # 4x batch averaging: kills target var?
    "lam05":      dict(lam=0.5),         # lean on Q baseline: lower estimator var?
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variant", default="no_qanchor", choices=list(VARIANTS))
    ap.add_argument("--episodes", type=int, default=400000)
    args = ap.parse_args()
    kuhn = build_kuhn()
    ee = max(2000, args.episodes // 200)
    cfg = dict(BASE); cfg.update(VARIANTS[args.variant])
    agent = NeuralACFR(kuhn, seed=args.seed, diag=True, **cfg)
    t0 = time.time()
    log = agent.run(args.episodes, eval_every=ee,
                    eval_fn=lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]})
    nc = [m["nc_last"] for m in log]
    lq = float(np.mean(nc[3 * len(nc) // 4:]))
    tag = f"s{args.seed}_{args.variant}"
    with open(os.path.join(RESULTS, f"diag_ablate_nc_{tag}.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "nc_last"]); w.writeheader()
        for m in log:
            w.writerow({"episode": m["episode"], "nc_last": m["nc_last"]})
    with open(os.path.join(RESULTS, f"diag_ablate_meters_{tag}.csv"), "w",
              newline="") as f:
        fl = ["episode", "eta", "kl_fit", "interference_l1", "q_rmse"]
        w = csv.DictWriter(f, fieldnames=fl); w.writeheader()
        for d in agent.diag_log:
            w.writerow({k: d[k] for k in fl})
    print(f"  {tag}: final={nc[-1]:.4f} min={min(nc):.4f} lastQ={lq:.4f} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
