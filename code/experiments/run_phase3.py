# -*- coding: utf-8 -*-
"""Phase 3: neural A-CFR (bridge version). Requires:  pip install torch

  N1  Kuhn: neural vs tabular sampled A-CFR, same schedule, lam=1.
      Success: curves roughly overlap -> the neural layer adds no
      pathology (bootstrapped non-reset training is stable).
  N2  distillation-budget ablation: distill_steps in {5, 20, 80}.
      Prop-4 evidence: floor should shrink with distillation accuracy
      (larger budget -> smaller delta), with NO error accumulation.
  N3  (--leduc) Leduc neural run, lam=1.

Usage:  python run_phase3.py [--quick] [--leduc]
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


def save_csv(name, rows, fields):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  saved {path}")


def n1_bridge(kuhn, quick):
    print("== N1: neural vs tabular (Kuhn, lam=1) ==")
    episodes = 30000 if quick else 200000
    ee = max(2000, episodes // 100)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    rows = {}
    t0 = time.time()
    _, log = run_sacfr(kuhn, episodes, eta=0.5, tau=0.1, K_ep=2000,
                       batch_size=16, superphase=4, eta_decay=0.5,
                       lam=1.0, eval_every=ee, eval_fn=ev)
    rows["tabular"] = log
    print(f"  tabular: {log[-1]['nc_last']:.4f} ({time.time()-t0:.0f}s)")
    agent = NeuralACFR(kuhn, eta=0.5, tau=0.1, K_ep=2000, batch_size=16,
                       superphase=4, eta_decay=0.5, lam=1.0,
                       distill_steps=20)
    log = agent.run(episodes, eval_every=ee, eval_fn=ev)
    rows["neural"] = log
    print(f"  neural : {log[-1]['nc_last']:.4f} ({time.time()-t0:.0f}s)")
    out = []
    for name, lg in rows.items():
        for m in lg:
            out.append({"alg": name, "episode": m["episode"],
                        "nc_last": m["nc_last"]})
    save_csv("p3_n1_bridge.csv", out, ["alg", "episode", "nc_last"])
    plt.figure(figsize=(7, 5))
    for name, lg in rows.items():
        plt.semilogy([m["episode"] for m in lg],
                     [max(m["nc_last"], 1e-12) for m in lg], label=name)
    plt.xlabel("episodes"); plt.ylabel("NashConv (last iterate)")
    plt.title("N1 Kuhn: neural bridge vs tabular A-CFR")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3_N1_bridge.png"), dpi=150)
    plt.close()


def n2_distill_budget(kuhn, quick):
    print("== N2: distillation budget ablation (Prop 4) ==")
    episodes = 30000 if quick else 150000
    ee = max(2000, episodes // 100)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    rows = {}
    for ds in (5, 20, 80):
        agent = NeuralACFR(kuhn, eta=0.5, tau=0.1, K_ep=2000,
                           batch_size=16, superphase=4, eta_decay=0.5,
                           lam=1.0, distill_steps=ds)
        log = agent.run(episodes, eval_every=ee, eval_fn=ev)
        rows[f"distill{ds}"] = log
        floor = float(np.mean([m["nc_last"] for m in log[-10:]]))
        print(f"  distill_steps={ds}: floor={floor:.4f}")
    out = []
    for name, lg in rows.items():
        for m in lg:
            out.append({"alg": name, "episode": m["episode"],
                        "nc_last": m["nc_last"]})
    save_csv("p3_n2_distill.csv", out, ["alg", "episode", "nc_last"])
    plt.figure(figsize=(7, 5))
    for name, lg in rows.items():
        plt.semilogy([m["episode"] for m in lg],
                     [max(m["nc_last"], 1e-12) for m in lg], label=name)
    plt.xlabel("episodes"); plt.ylabel("NashConv (last iterate)")
    plt.title("N2 Kuhn: floor vs distillation budget (Prop 4)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3_N2_distill.png"), dpi=150)
    plt.close()


def n3_leduc(leduc, quick):
    print("== N3: Leduc neural (lam=1) ==")
    episodes = 100000 if quick else 1000000
    ee = max(5000, episodes // 100)
    ev = lambda s, _a: {"nc_last": nash_conv(leduc, s)[0]}
    agent = NeuralACFR(leduc, eta=0.5, tau=0.1, K_ep=4000, batch_size=32,
                       superphase=4, eta_decay=0.5, lam=1.0,
                       distill_steps=20, q_steps=40)
    t0 = time.time()
    log = agent.run(episodes, eval_every=ee, eval_fn=ev)
    print(f"  final={log[-1]['nc_last']:.4f} ({time.time()-t0:.0f}s)")
    save_csv("p3_n3_leduc.csv",
             [{"episode": m["episode"], "nc_last": m["nc_last"]}
              for m in log], ["episode", "nc_last"])
    plt.figure(figsize=(7, 5))
    plt.semilogy([m["episode"] for m in log],
                 [max(m["nc_last"], 1e-12) for m in log])
    plt.xlabel("episodes"); plt.ylabel("NashConv")
    plt.title("N3 Leduc: neural A-CFR (bridge)")
    plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3_N3_leduc.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--leduc", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    kuhn = build_kuhn()
    n1_bridge(kuhn, args.quick)
    n2_distill_budget(kuhn, args.quick)
    if args.leduc:
        n3_leduc(build_leduc(), args.quick)
    print(f"\nALL DONE in {time.time()-t0:.0f}s. Results in "
          f"{os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
