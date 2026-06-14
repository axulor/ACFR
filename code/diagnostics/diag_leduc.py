# -*- coding: utf-8 -*-
"""Diagnose the Leduc neural last-iterate degradation on torch 2.11.

All 3 seeds of the load-bearing config (lam=0.5 + triple-freeze) reach
min ~1.15 (below tabular 1.394) then degrade to final ~2.0-2.8. The
Windows headline (0.702) does not reproduce. Hypotheses:
  freeze : the frozen eta phase lets the last-iterate random-walk under
           estimator variance with no annealing damping -> drift.
  anneal : continued annealing (eta_min=0) should damp the walk.
  bigb   : larger batch (B=64) cuts estimator variance -> smaller walk.
  anneal_bigb : both.
diag=True records q_rmse / interference / kl so we can see what rises at
the degradation onset.

Usage: python diag_leduc.py --arm freeze|anneal|bigb|anneal_bigb
                            [--seed 0] [--quick]
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

BASE = dict(eta=0.5, tau=0.1, K_ep=2000, batch_size=16, superphase=4,
            eta_decay=0.5, eta_min=0.0625, lam=0.5, distill_steps=20,
            width=128, encoding="features", lr_couple=True, q_anchor=True,
            q_replay=20000, q_batch=512)
ARMS = {
    "freeze":      dict(),                         # current load-bearing
    "anneal":      dict(eta_min=0.0),              # keep annealing
    "bigb":        dict(batch_size=64),            # 4x variance averaging
    "anneal_bigb": dict(eta_min=0.0, batch_size=64),
    # anneal_bigb + re-distill ALL seen infoset targets each cycle:
    # kills policy drift at non-batch infosets (still current-policy
    # targets, NO average-strategy net). REFUTED: stale targets -> worse.
    "abr":         dict(eta_min=0.0, batch_size=64, replay_targets=True),
    # variance-amplitude hypothesis: Leduc (depth 4, u=13) compounds
    # estimator variance per Prop 3.2; B=64 (enough for Kuhn depth 2)
    # is too small here -> push batch up (pure variance reduction).
    "anneal_b128": dict(eta_min=0.0, batch_size=128),
    "anneal_b256": dict(eta_min=0.0, batch_size=256),
    # eps_k inner-accuracy condition (Thm 2): anchor must move slowly
    # enough for the inner loop to converge first; noisier neural inner
    # loop on torch 2.11 may need larger K. (anneal + B64 base)
    "anneal_K4k":  dict(eta_min=0.0, batch_size=64, K_ep=4000),
    "anneal_K8k":  dict(eta_min=0.0, batch_size=64, K_ep=8000),
    "anneal_K16k": dict(eta_min=0.0, batch_size=64, K_ep=16000),
    # soft anchor (anchor_ema<1): anchor lags policy -> restoring inertia
    # to damp the residual init-dependent last-iterate drift. On best
    # base (anneal + B64 + K8000 sweet spot).
    "soft50": dict(eta_min=0.0, batch_size=64, K_ep=8000, anchor_ema=0.5),
    "soft25": dict(eta_min=0.0, batch_size=64, K_ep=8000, anchor_ema=0.25),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="anneal", choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    g = build_leduc()
    episodes = 20000 if args.quick else 400000
    ee = max(4000, episodes // 100)
    ev = lambda s, _a: {"nc_last": nash_conv(g, s)[0]}
    cfg = dict(BASE); cfg.update(ARMS[args.arm])
    agent = NeuralACFR(g, seed=args.seed, diag=True, **cfg)
    t0 = time.time()
    log = agent.run(episodes, eval_every=ee, eval_fn=ev)
    nc = [m["nc_last"] for m in log]
    lq = float(np.mean(nc[3 * len(nc) // 4:]))
    dt = time.time() - t0
    tag = f"{args.arm}_s{args.seed}"
    print(f"  {tag}: final={nc[-1]:.4f} min={min(nc):.4f} lastQ={lq:.4f} "
          f"({dt:.0f}s, {dt/episodes*1e3:.2f}ms/ep)")
    with open(os.path.join(RESULTS, f"diag_leduc_nc_{tag}.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "nc_last"]); w.writeheader()
        for m in log:
            w.writerow({"episode": m["episode"], "nc_last": m["nc_last"]})
    with open(os.path.join(RESULTS, f"diag_leduc_meters_{tag}.csv"), "w",
              newline="") as f:
        fl = ["episode", "eta", "kl_fit", "interference_l1", "q_rmse"]
        w = csv.DictWriter(f, fieldnames=fl); w.writeheader()
        for d in agent.diag_log:
            w.writerow({k: d[k] for k in fl})


if __name__ == "__main__":
    main()
