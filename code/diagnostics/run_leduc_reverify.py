# -*- coding: utf-8 -*-
"""Re-verify the load-bearing Leduc headline on Ubuntu / torch 2.11.

Doc 22 §3 (Windows): neural lam=0.5 + eta/K/lr triple-freeze @ eta_min
=0.0625, features, width 128, 400k, seed 0 -> min 0.645, final 0.702,
vs tabular sacfr 1.394 (2x) and OS-MCCFR last 1.779 (2.5x).

This script reproduces exactly that config and also runs the tabular
sacfr baseline, so the 2x margin can be re-checked head-to-head on this
stack. Run arms as separate processes for wall-clock parallelism.

Usage: python run_leduc_reverify.py --exp neural|tabular [--quick]
                                     [--seed 0] [--episodes N]
"""
import argparse
import csv
import os
import time

import numpy as np
from games import build_leduc
from exploitability import nash_conv
from sampling import run_sacfr
from neural_acfr import NeuralACFR

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# load-bearing neural recipe (doc 22 §3): lam=0.5 + triple-freeze
NEURAL = dict(eta=0.5, tau=0.1, K_ep=2000, batch_size=16, superphase=4,
              eta_decay=0.5, eta_min=0.0625, lam=0.5, distill_steps=20,
              width=128, encoding="features", lr_couple=True,
              q_anchor=True, q_replay=20000, q_batch=512)
# tabular sampled baseline (RECIPE from phase4a, lam=1)
TABULAR = dict(eta=0.5, tau=0.1, K_ep=2000, batch_size=16, superphase=4,
               eta_decay=0.5, lam=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="neural", choices=["neural", "tabular"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--episodes", type=int, default=0)
    args = ap.parse_args()
    g = build_leduc()
    episodes = args.episodes or (20000 if args.quick else 400000)
    ee = max(4000, episodes // 50)
    ev = lambda s, _a: {"nc_last": nash_conv(g, s)[0]}
    t0 = time.time()
    if args.exp == "tabular":
        _, log = run_sacfr(g, episodes, eval_every=ee, eval_fn=ev,
                           seed=args.seed, **TABULAR)
        alg = "tabular"
    else:
        agent = NeuralACFR(g, seed=args.seed, **NEURAL)
        log = agent.run(episodes, eval_every=ee, eval_fn=ev)
        alg = "neural_lam05_frz"
    nc = [m["nc_last"] for m in log]
    lq = float(np.mean(nc[3 * len(nc) // 4:]))
    dt = time.time() - t0
    print(f"  {alg} s{args.seed}: final={nc[-1]:.4f} min={min(nc):.4f} "
          f"lastQ={lq:.4f}  ({dt:.0f}s, {dt/episodes*1e3:.2f}ms/ep)")
    tag = f"{args.exp}_s{args.seed}" + ("_quick" if args.quick else "")
    with open(os.path.join(RESULTS, f"leduc_reverify_{tag}.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alg", "episode", "nc_last"])
        w.writeheader()
        for m in log:
            w.writerow({"alg": alg, "episode": m["episode"],
                        "nc_last": m["nc_last"]})


if __name__ == "__main__":
    main()
