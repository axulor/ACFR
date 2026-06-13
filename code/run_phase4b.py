# -*- coding: utf-8 -*-
"""Phase 4b (Windows prelim): Liar's Dice, recipe transferred verbatim.

Claim under test: the final recipe (3-way annealing + Q-anchor +
Q-replay + features) transfers to a new game with ZERO per-game tuning.
Game: Liar's Dice 2x1die, sides=5 (5120 infosets; sides=6 = OpenSpiel
standard reserved for the Ubuntu run).

Arms (400k episodes, seed 0): tabular sacfr / neural features /
OS-MCCFR (avg + last reference).

  python run_phase4b.py --exp tab|feat|os [--quick]
"""
import argparse
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from games import build_liars_dice
from exploitability import nash_conv
from sampling import run_sacfr, run_os_mccfr
from neural_acfr import NeuralACFR

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

RECIPE = dict(eta=0.5, tau=0.1, K_ep=2000, batch_size=16,
              superphase=4, eta_decay=0.5, lam=1.0)


def dump(name, curves):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alg", "episode", "nc_last"])
        w.writeheader()
        for alg, lg in curves.items():
            for m in lg:
                w.writerow({"alg": alg, "episode": m["episode"],
                            "nc_last": m["nc_last"]})
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="tab",
                    choices=["tab", "feat", "os", "merge"])
    ap.add_argument("--sides", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    g = build_liars_dice(args.sides)
    episodes = 20000 if args.quick else 400000
    ee = max(4000, episodes // 50)
    ev = lambda s, _a: {"nc_last": nash_conv(g, s)[0]}
    t0 = time.time()

    if args.exp == "merge":
        curves = {}
        for which in ("tab", "feat", "os"):
            p = os.path.join(RESULTS, f"p4b_{which}_liars{args.sides}.csv")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                for r in csv.DictReader(f):
                    curves.setdefault(r["alg"], []).append(
                        {"episode": int(r["episode"]),
                         "nc_last": float(r["nc_last"])})
        plt.figure(figsize=(8, 5))
        for alg, lg in curves.items():
            plt.semilogy([m["episode"] for m in lg],
                         [max(m["nc_last"], 1e-12) for m in lg],
                         label=alg)
        plt.xlabel("episodes"); plt.ylabel("NashConv")
        plt.title(f"P4b: Liar's Dice (sides={args.sides}), zero-tuning "
                  f"transfer")
        plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS,
                                 f"figP4b_liars{args.sides}.png"), dpi=150)
        plt.close()
        print("  merged figure written")
        return

    if args.exp == "tab":
        _, log = run_sacfr(g, episodes, eval_every=ee, eval_fn=ev,
                           seed=0, **RECIPE)
        curves = {"tabular": log}
    elif args.exp == "feat":
        # Leduc-winning neural config (P4c): lam=0.5 (depth-compounding
        # variance) + freeze eta/K/lr at 0.0625 (3-way coupled).
        kw = dict(RECIPE, lam=0.5, eta_min=0.0625)
        agent = NeuralACFR(g, **kw, distill_steps=20, width=128,
                           seed=0, encoding="features", lr_couple=True,
                           q_anchor=True, q_replay=20000, q_batch=512)
        log = agent.run(episodes, eval_every=ee, eval_fn=ev)
        curves = {"neural_features": log}
    else:
        _, _, log = run_os_mccfr(g, episodes, eval_every=ee, eval_fn=ev,
                                 seed=0)
        curves = {"osmccfr_avg": [{"episode": m["episode"],
                                   "nc_last": m["nc_avg"]} for m in log],
                  "osmccfr_last": [{"episode": m["episode"],
                                    "nc_last": m["nc_last"]} for m in log]}
    for alg, lg in curves.items():
        print(f"  {alg}: final={lg[-1]['nc_last']:.4f}, "
              f"min={min(m['nc_last'] for m in lg):.4f} "
              f"({time.time()-t0:.0f}s)")
    dump(f"p4b_{args.exp}_liars{args.sides}.csv", curves)


if __name__ == "__main__":
    main()
