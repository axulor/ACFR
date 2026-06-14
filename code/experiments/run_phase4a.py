# -*- coding: utf-8 -*-
"""Phase 4a: real feature encodings (Kuhn sanity + Leduc at small scale).

Final neural recipe from P3d/P3e: superphase annealing (never freeze) +
Q-anchor + Q-replay (fresh recomputed targets) + distill 20.

  A0       Kuhn 200k, 2 seeds: encoding=features vs onehot (both with
           the final recipe). Success: features tracks onehot/tabular --
           "generalizing encoding does not break the loop".
  A1_*     Leduc 400k, seed 0: tabular sampled baseline vs neural
           onehot vs neural features. Reference points (P1c, 1M eps):
           tabular sacfr 1.27, OS-MCCFR avg 0.27. Hope: feature net
           beats tabular sacfr via statistical sharing across the 936
           infosets (doc 10, open question #3).

Sub-experiments run as separate processes for wall-clock parallelism:
  python run_phase4a.py --exp a0 | a1_tab | a1_onehot | a1_feat [--quick]
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
from sampling import run_sacfr
from neural_acfr import NeuralACFR

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

RECIPE = dict(eta=0.5, tau=0.1, K_ep=2000, batch_size=16,
              superphase=4, eta_decay=0.5, lam=1.0)
QSTAB = dict(q_anchor=True, q_replay=20000, q_batch=512,
             lr_couple=True)   # P3f: full final recipe incl. lr anchor


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


def plot(name, curves, title):
    plt.figure(figsize=(8, 5))
    for alg, lg in curves.items():
        plt.semilogy([m["episode"] for m in lg],
                     [max(m["nc_last"], 1e-12) for m in lg], label=alg)
    plt.xlabel("episodes"); plt.ylabel("NashConv (last iterate)")
    plt.title(title); plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, name), dpi=150)
    plt.close()


def a0(quick):
    print("== P4a-A0: Kuhn, features vs onehot (final recipe) ==")
    kuhn = build_kuhn()
    episodes = 20000 if quick else 200000
    ee = max(2000, episodes // 100)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    curves = {}
    t0 = time.time()
    for enc in ("onehot", "features"):
        for sd in (0, 1):
            agent = NeuralACFR(kuhn, **RECIPE, **QSTAB, distill_steps=20,
                               seed=sd, encoding=enc)
            log = agent.run(episodes, eval_every=ee, eval_fn=ev)
            curves[f"{enc}_s{sd}"] = log
            print(f"  {enc} seed{sd}: final={log[-1]['nc_last']:.4f}, "
                  f"min={min(m['nc_last'] for m in log):.4f} "
                  f"({time.time()-t0:.0f}s)")
    dump("p4a_a0_kuhn.csv", curves)
    plot("figP4a_A0_kuhn.png", curves,
         "P4a-A0: Kuhn, generalizing features vs one-hot")


def a1(which, quick):
    leduc = build_leduc()
    episodes = 20000 if quick else 400000
    ee = max(4000, episodes // 50)
    ev = lambda s, _a: {"nc_last": nash_conv(leduc, s)[0]}
    t0 = time.time()
    curves = {}
    if which == "a1_tab":
        _, log = run_sacfr(leduc, episodes, eval_every=ee, eval_fn=ev,
                           seed=0, **RECIPE)
        curves["tabular"] = log
    else:
        enc = "onehot" if which == "a1_onehot" else "features"
        agent = NeuralACFR(leduc, **RECIPE, **QSTAB, distill_steps=20,
                           width=128, seed=0, encoding=enc)
        log = agent.run(episodes, eval_every=ee, eval_fn=ev)
        curves[f"neural_{enc}"] = log
    alg = list(curves)[0]
    print(f"  {alg}: final={curves[alg][-1]['nc_last']:.4f}, "
          f"min={min(m['nc_last'] for m in curves[alg]):.4f} "
          f"({time.time()-t0:.0f}s)")
    dump(f"p4a_{which}_leduc.csv", curves)


def merge_a1():
    """Combine the three a1 CSVs into one figure (run after all done)."""
    curves = {}
    for which in ("a1_tab", "a1_onehot", "a1_feat"):
        p = os.path.join(RESULTS, f"p4a_{which}_leduc.csv")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                curves.setdefault(r["alg"], []).append(
                    {"episode": int(r["episode"]),
                     "nc_last": float(r["nc_last"])})
    if curves:
        plot("figP4a_A1_leduc.png", curves,
             "P4a-A1: Leduc, neural (final recipe) vs tabular sampled")
        print("  merged figure written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="a0",
                    choices=["a0", "a1_tab", "a1_onehot", "a1_feat",
                             "merge"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.exp == "a0":
        a0(args.quick)
    elif args.exp == "merge":
        merge_a1()
    else:
        a1(args.exp, args.quick)


if __name__ == "__main__":
    main()
