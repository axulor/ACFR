# -*- coding: utf-8 -*-
"""Phase 3e: long-horizon confirmation of the final neural recipe.

P3d verdict: the P3b 'schedule pathology' was misattributed -- the
annealing schedule is fine; Q-drift variance inflation was the disease.
Final recipe: superphase annealing (naive, never freeze) + Q-anchor +
Q-replay.  At 200k it matches tabular (min 0.062 vs 0.068) and is still
descending.

This phase: 400k episodes, neural recipe x 2 seeds vs tabular x 1 seed.
Success: neural floor keeps stepping down across superphases (the
paper's central neural claim) and stays within ~2x of tabular.

Usage:  python run_phase3e.py [--quick]
"""
import argparse
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from games import build_kuhn
from exploitability import nash_conv
from sampling import run_sacfr
from neural_acfr import NeuralACFR

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    kuhn = build_kuhn()
    episodes = 40000 if args.quick else 400000
    ee = max(2000, episodes // 200)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    base = dict(eta=0.5, tau=0.1, K_ep=2000, batch_size=16,
                superphase=4, eta_decay=0.5, lam=1.0)
    out, curves = [], {}

    _, log = run_sacfr(kuhn, episodes, eval_every=ee, eval_fn=ev,
                       seed=0, **base)
    curves["tabular"] = log
    print(f"  tabular: final={log[-1]['nc_last']:.4f} "
          f"({time.time()-t0:.0f}s)")

    for sd in (0, 1):
        agent = NeuralACFR(kuhn, **base, distill_steps=20, seed=sd,
                           q_anchor=True, q_replay=20000, q_batch=512)
        log = agent.run(episodes, eval_every=ee, eval_fn=ev)
        curves[f"neural_qboth_s{sd}"] = log
        print(f"  neural_qboth seed{sd}: final={log[-1]['nc_last']:.4f}, "
              f"min={min(m['nc_last'] for m in log):.4f} "
              f"({time.time()-t0:.0f}s)")

    for name, lg in curves.items():
        for m in lg:
            out.append({"alg": name, "episode": m["episode"],
                        "nc_last": m["nc_last"]})
    path = os.path.join(RESULTS, "p3e_long.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alg", "episode", "nc_last"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"  saved {path}")

    plt.figure(figsize=(8, 5))
    for name, lg in curves.items():
        plt.semilogy([m["episode"] for m in lg],
                     [max(m["nc_last"], 1e-12) for m in lg], label=name)
    plt.xlabel("episodes"); plt.ylabel("NashConv (last iterate)")
    plt.title("P3e: final neural recipe vs tabular, 400k")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3e_long.png"), dpi=150)
    plt.close()
    print(f"\nALL DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
