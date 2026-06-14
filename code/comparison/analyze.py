# -*- coding: utf-8 -*-
"""Analyze the comparison campaign: overlay exploitability-vs-episode curves
(ours last-iterate vs baselines avg-iterate), average over seeds, and print
summary metrics that surface where A-CFR (ours) has an advantage:
  - final exp, min exp (over training)
  - AUC (area under log-exp curve) = overall convergence quality
  - episodes-to-threshold (sample efficiency at a target exp)

Reads logs_ours/ours_{method}_{osgame}_s{seed}.csv and
logs_baseline/{algo}_{harnessgame}_s{seed}.csv. Produces figures in figs/.

Usage (any env with numpy+matplotlib):
  python analyze.py --games KuhnPoker,LeducPoker
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_OURS = os.path.join(HERE, "logs_ours")
LOG_BASE = os.path.join(HERE, "logs_baseline")
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

GAME_OS = {
    "KuhnPoker": "kuhn_poker",
    "LeducPoker": "leduc_poker",
    "LiarsDice5": "liars_dice(numdice=1,dice_sides=5)",
    "LiarsDice6": "liars_dice(numdice=1,dice_sides=6)",
}
OURS_LABELS = {"neural": "A-CFR neural onehot (ours)",
               "neural_feat": "A-CFR neural FEATURES (ours)",
               "sampling": "A-CFR sampled (ours)"}


def read_curve(path):
    eps, exps = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            eps.append(int(r["episode"]))
            exps.append(float(r["exp"]))
    return np.array(eps), np.array(exps)


def collect(game):
    """Return {label: list of (eps, exps) over seeds}."""
    series = defaultdict(list)
    osg = GAME_OS[game]
    for path in glob.glob(os.path.join(LOG_OURS, f"ours_*_{osg}_s*.csv")):
        base = os.path.basename(path)[:-4]  # strip .csv
        # method = between 'ours_' and '_{osg}_s{seed}'
        method = base[len("ours_"):].rsplit(f"_{osg}_s", 1)[0]
        series[OURS_LABELS.get(method, "A-CFR " + method + " (ours)")].append(
            read_curve(path))
    for path in glob.glob(os.path.join(LOG_BASE, f"*_{game}_s*.csv")):
        algo = os.path.basename(path).split("_" + game + "_")[0]
        series[algo + " (avg-iter)"].append(read_curve(path))
    return series


def interp_mean(curves, grid):
    """Mean exp over seeds on a common episode grid (log-interp)."""
    vals = []
    for eps, exps in curves:
        e = np.maximum(exps, 1e-12)
        vals.append(np.interp(grid, eps, e, left=e[0], right=e[-1]))
    vals = np.array(vals)
    return vals.mean(0), vals.std(0)


def summarize(game):
    series = collect(game)
    if not series:
        print(f"[{game}] no data yet")
        return
    # common grid = up to min of max-episodes across all
    maxep = min(max(c[0][-1] for c in cs) for cs in series.values())
    grid = np.linspace(maxep * 0.02, maxep, 200)
    n_seed = {k: len(v) for k, v in series.items()}
    print(f"\n===== {game} (episodes up to {int(maxep)}) =====")
    THRESH = [0.5, 0.3, 0.2, 0.1, 0.05]
    hdr = "  ".join(f"ep@{t}".rjust(8) for t in THRESH)
    print(f"{'method':<34} {'seeds':>5} {'final':>8} {'min':>8}  {hdr}")
    rows = {}
    eff = {}  # label -> {thresh: episodes-to-reach}
    for label, curves in sorted(series.items()):
        m, s = interp_mean(curves, grid)
        final, mn = m[-1], m.min()
        auc = float(np.trapz(np.log10(m), grid) / (grid[-1] - grid[0]))
        e = {}
        for t in THRESH:
            below = np.where(m <= t)[0]
            e[t] = grid[below[0]] if len(below) else float("inf")
        eff[label] = e
        rows[label] = (m, s, final, mn, auc, e.get(0.1, float("inf")))
        cells = "  ".join((f"{int(e[t]):>8}" if np.isfinite(e[t])
                           else f"{'--':>8}") for t in THRESH)
        print(f"{label:<34} {n_seed[label]:>5} {final:>8.4f} {mn:>8.4f}  "
              f"{cells}")
    # head-to-head: best OURS vs best BASELINE at each threshold
    ours = {k: v for k, v in eff.items() if "ours" in k}
    base = {k: v for k, v in eff.items() if "ours" not in k
            and "NFSP" not in k}  # NFSP too weak; exclude from 'best CFR'
    if ours and base:
        print("  -- sample-efficiency head-to-head (episodes to reach exp) --")
        for t in THRESH:
            bo = min((eff[k][t], k) for k in ours)
            bb = min((eff[k][t], k) for k in base)
            if np.isfinite(bo[0]) or np.isfinite(bb[0]):
                spd = (bb[0] / bo[0]) if (np.isfinite(bo[0])
                                          and bo[0] > 0) else float("nan")
                verdict = (f"ours {spd:.1f}x faster" if np.isfinite(spd)
                           and spd > 1 else
                           (f"baseline {1/spd:.1f}x faster"
                            if np.isfinite(spd) else "n/a"))
                print(f"     exp<={t}: ours {bo[0] if np.isfinite(bo[0]) else '--'}"
                      f" ({bo[1].split(' ')[1] if np.isfinite(bo[0]) else ''}) | "
                      f"best-CFR {bb[0] if np.isfinite(bb[0]) else '--'} "
                      f"({bb[1].split(' ')[0] if np.isfinite(bb[0]) else ''}) "
                      f"-> {verdict}")
    # plot
    plt.figure(figsize=(8, 5.5))
    for label, (m, s, *_rest) in rows.items():
        is_ours = "ours" in label
        plt.plot(grid, m, lw=2.4 if is_ours else 1.4,
                 ls="-" if is_ours else "--",
                 label=label, zorder=3 if is_ours else 2)
        plt.fill_between(grid, np.maximum(m - s, 1e-12), m + s, alpha=0.12)
    plt.yscale("log")
    plt.xlabel("episodes sampled")
    plt.ylabel("exploitability (NashConv/2)")
    plt.title(f"{game}: A-CFR (last-iterate) vs baselines (avg-iterate)")
    plt.legend(fontsize=7.5, ncol=1)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGS, f"compare_{game}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  fig -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="KuhnPoker,LeducPoker")
    args = ap.parse_args()
    for g in args.games.split(","):
        if g:
            summarize(g)


if __name__ == "__main__":
    main()
