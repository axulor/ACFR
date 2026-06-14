# -*- coding: utf-8 -*-
"""Phase 1b: revised sampling experiments (see 08_Phase1_结果判读.md).

Fixes over phase 1:
  - A-CFR gets the stochastic stepsize schedule (superphase: eta halves,
    K doubles) -> noise floor must DESCEND in a staircase, while CFR+
    stays at its noise floor. The quantitative scissors.
  - Sampled A-CFR uses minibatch-averaged advantages (variance/B) with
    a stronger geometric schedule.

Usage:  python run_phase1b.py [--quick]
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
from algorithms import run_cfr, run_acfr
from sampling import run_sacfr, run_os_mccfr

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


def p1a2_scissors(kuhn, quick):
    print("== P1A2: noise scissors with stepsize schedule ==")
    iters = 8000 if quick else 60000
    ev = lambda s, a: ({"nc_last": nash_conv(kuhn, s)[0]} if a is None else
                       {"nc_last": nash_conv(kuhn, s)[0],
                        "nc_avg": nash_conv(kuhn, a)[0]})
    rows = {}
    for ns in (0.05, 0.2):
        _, _, log = run_cfr(kuhn, iters, plus=True, noise_std=ns,
                            eval_every=max(10, iters // 200), eval_fn=ev)
        rows[f"cfrplus_n{ns}"] = log
        # constant-eta control (the phase-1 behaviour: flat floor)
        _, log, _ = run_acfr(kuhn, iters, eta=0.2, tau=0.2,
                             anchor_mode="periodic", K=50,
                             scale_utilities=True, normalize=True,
                             noise_std=ns,
                             eval_every=max(10, iters // 200), eval_fn=ev)
        rows[f"acfr_const_n{ns}"] = log
        # scheduled version: floor must descend
        _, log, _ = run_acfr(kuhn, iters, eta=0.4, tau=0.2,
                             anchor_mode="periodic", K=50,
                             superphase=5, eta_decay=0.5,
                             scale_utilities=True, normalize=True,
                             noise_std=ns,
                             eval_every=max(10, iters // 200), eval_fn=ev)
        rows[f"acfr_sched_n{ns}"] = log
        print(f"  noise={ns}: cfr+ {rows[f'cfrplus_n{ns}'][-1]['nc_last']:.4f}"
              f" | const {rows[f'acfr_const_n{ns}'][-1]['nc_last']:.4f}"
              f" | sched {rows[f'acfr_sched_n{ns}'][-1]['nc_last']:.4f}")
    out = []
    for name, log in rows.items():
        for m in log:
            out.append({"alg": name, "iter": m["iter"],
                        "nc_last": m["nc_last"]})
    save_csv("p1a2_scissors.csv", out, ["alg", "iter", "nc_last"])
    plt.figure(figsize=(8, 5))
    styles = {"cfrplus": "--", "acfr_const": ":", "acfr_sched": "-"}
    colors = {0.05: "C1", 0.2: "C3"}
    for ns in (0.05, 0.2):
        for kind, st in styles.items():
            log = rows[f"{kind}_n{ns}"]
            it = [m["iter"] for m in log]
            plt.semilogy(it, [max(m["nc_last"], 1e-12) for m in log], st,
                         color=colors[ns], label=f"{kind}, noise={ns}")
    plt.xlabel("iteration"); plt.ylabel("NashConv (last iterate)")
    plt.title("P1A2 Kuhn: scheduled A-CFR floor descends; CFR+ floor flat")
    plt.legend(fontsize=7); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP1A2_scissors.png"), dpi=150)
    plt.close()


def p1b2_sampled(kuhn, quick):
    print("== P1B2: batched sampled A-CFR, lambda spectrum ==")
    episodes = 40000 if quick else 400000
    ee = max(2000, episodes // 150)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    rows = {}
    t0 = time.time()
    _, avg, log = run_os_mccfr(kuhn, episodes, eval_every=ee, eval_fn=ev)
    rows["osmccfr"] = log
    print(f"  os-mccfr: avg={log[-1]['nc_avg']:.4f} ({time.time()-t0:.0f}s)")
    for lam in (0.0, 0.5, 1.0, "adaptive"):
        _, log = run_sacfr(kuhn, episodes, eta=0.5, tau=0.1,
                           K_ep=2000, batch_size=16,
                           superphase=4, eta_decay=0.5,
                           lam=lam, eval_every=ee, eval_fn=ev)
        rows[f"sacfr_lam{lam}"] = log
        print(f"  sacfr lam={lam}: last={log[-1]['nc_last']:.4f} "
              f"({time.time()-t0:.0f}s)")
    out = []
    for name, log in rows.items():
        for m in log:
            out.append({"alg": name, "episode": m["episode"],
                        "nc_last": m["nc_last"],
                        "nc_avg": m.get("nc_avg", "")})
    save_csv("p1b2_sampled.csv", out,
             ["alg", "episode", "nc_last", "nc_avg"])
    plt.figure(figsize=(8, 5))
    for name, log in rows.items():
        it = [m["episode"] for m in log]
        plt.semilogy(it, [max(m["nc_last"], 1e-12) for m in log],
                     label=f"{name} (last)")
    it = [m["episode"] for m in rows["osmccfr"]]
    plt.semilogy(it, [max(m["nc_avg"], 1e-12) for m in rows["osmccfr"]],
                 "--", label="osmccfr (avg)")
    plt.xlabel("episodes"); plt.ylabel("NashConv")
    plt.title("P1B2 Kuhn: batched sampled A-CFR vs OS-MCCFR")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP1B2_sampled.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    kuhn = build_kuhn()
    p1a2_scissors(kuhn, args.quick)
    p1b2_sampled(kuhn, args.quick)
    print(f"\nALL DONE in {time.time()-t0:.0f}s. Results in "
          f"{os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
