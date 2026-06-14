# -*- coding: utf-8 -*-
"""Phase 3b: the Prop-4 schedule corollary experiments
(see 14_P3冒烟判读与进展再评估.md). Run AFTER run_phase3.py finishes
(different output filenames; no clobbering).

  B1  Kuhn 200k, lam=1: tabular vs neural-naive (the N1 pathology) vs
      neural-matched (grow_distill) vs neural-frozen (eta_min).
      Success: matched/frozen track tabular; naive rises after ~8k.
  B2  clean Prop-4 ordering: schedule OFF (superphase=None), floor vs
      distill budget {5,20,80}. Success: monotone floors, no drift over
      100k episodes (no error accumulation).

Usage:  python run_phase3b.py [--quick]
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


def save_csv(name, rows, fields):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  saved {path}")


def b1_schedules(kuhn, quick):
    print("== P3b-B1: schedule corollary (Kuhn) ==")
    episodes = 30000 if quick else 200000
    ee = max(2000, episodes // 100)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    base = dict(eta=0.5, tau=0.1, K_ep=2000, batch_size=16,
                superphase=4, eta_decay=0.5, lam=1.0)
    rows = {}
    t0 = time.time()
    _, log = run_sacfr(kuhn, episodes, eval_every=ee, eval_fn=ev, **base)
    rows["tabular"] = log
    print(f"  tabular: {log[-1]['nc_last']:.4f} ({time.time()-t0:.0f}s)")
    for name, kw in [
        ("neural_naive", dict(distill_steps=20)),
        ("neural_matched", dict(distill_steps=20, grow_distill=True)),
        ("neural_frozen", dict(distill_steps=20, eta_min=0.125)),
    ]:
        agent = NeuralACFR(kuhn, **base, **kw)
        log = agent.run(episodes, eval_every=ee, eval_fn=ev)
        rows[name] = log
        print(f"  {name}: {log[-1]['nc_last']:.4f} ({time.time()-t0:.0f}s)")
    out = []
    for name, lg in rows.items():
        for m in lg:
            out.append({"alg": name, "episode": m["episode"],
                        "nc_last": m["nc_last"]})
    save_csv("p3b_b1_schedules.csv", out, ["alg", "episode", "nc_last"])
    plt.figure(figsize=(8, 5))
    for name, lg in rows.items():
        plt.semilogy([m["episode"] for m in lg],
                     [max(m["nc_last"], 1e-12) for m in lg], label=name)
    plt.xlabel("episodes"); plt.ylabel("NashConv (last iterate)")
    plt.title("P3b-B1: Prop-4 schedule corollary (matched vs naive)")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3b_B1_schedules.png"), dpi=150)
    plt.close()


def b2_floor_ordering(kuhn, quick):
    print("== P3b-B2: distill-budget floors, schedule OFF ==")
    episodes = 30000 if quick else 100000
    ee = max(2000, episodes // 100)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    rows, out = {}, []
    for ds in (5, 20, 80):
        agent = NeuralACFR(kuhn, eta=0.5, tau=0.1, K_ep=2000,
                           batch_size=16, superphase=None, lam=1.0,
                           distill_steps=ds)
        log = agent.run(episodes, eval_every=ee, eval_fn=ev)
        rows[f"distill{ds}"] = log
        floor = float(np.mean([m["nc_last"] for m in log[-10:]]))
        drift = floor - float(np.mean([m["nc_last"]
                                       for m in log[len(log)//2:
                                                    len(log)//2+10]]))
        out.append({"distill_steps": ds, "floor": floor, "late_drift": drift})
        print(f"  distill={ds}: floor={floor:.4f}, late drift={drift:+.4f}")
    save_csv("p3b_b2_floors.csv", out,
             ["distill_steps", "floor", "late_drift"])
    plt.figure(figsize=(8, 5))
    for name, lg in rows.items():
        plt.semilogy([m["episode"] for m in lg],
                     [max(m["nc_last"], 1e-12) for m in lg], label=name)
    plt.xlabel("episodes"); plt.ylabel("NashConv (last iterate)")
    plt.title("P3b-B2: floors vs distillation budget (schedule off)")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3b_B2_floors.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    kuhn = build_kuhn()
    b1_schedules(kuhn, args.quick)
    b2_floor_ordering(kuhn, args.quick)
    print(f"\nALL DONE in {time.time()-t0:.0f}s. Results in "
          f"{os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
