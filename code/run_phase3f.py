# -*- coding: utf-8 -*-
"""Phase 3f: lr-coupled annealing (the last floor).

P3e negative result: even with Q stabilizers, all 400k runs rebound
after ~200k (mins 0.018-0.062, finals 0.16-0.44). Diagnosis: the
surviving floor is policy-side per-update SGD noise (interference
~0.01-0.02/update, measured in P3c-C0) which does not shrink as eta
anneals -- random-walk variance ~ sigma_w^2/(eta*tau) GROWS. The true
Prop-4 delta is update variance, not fit bias. Fix: lr *= eta_decay
whenever eta decays (noise ~ lr^2 ~ eta^2 => floor ~ eta keeps falling).

Arms (Kuhn 400k): lr-coupled onehot x2 seeds, lr-coupled features x2
seeds. Reference curves from p3e_long.csv (uncoupled onehot + tabular)
and p4a_a0_kuhn.csv (uncoupled features) are overlaid in the figure.
Success: no rebound (last-quarter mean <= min*1.5) and final <= 2x
tabular(0.018).

Usage:  python run_phase3f.py [--quick]
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
from neural_acfr import NeuralACFR

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    kuhn = build_kuhn()
    episodes = 40000 if args.quick else 400000
    ee = max(2000, episodes // 200)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    t0 = time.time()
    out, curves = [], {}
    for enc in ("onehot", "features"):
        for sd in (0, 1):
            agent = NeuralACFR(kuhn, eta=0.5, tau=0.1, K_ep=2000,
                               batch_size=16, superphase=4,
                               eta_decay=0.5, lam=1.0, distill_steps=20,
                               seed=sd, encoding=enc, lr_couple=True,
                               q_anchor=True, q_replay=20000,
                               q_batch=512)
            log = agent.run(episodes, eval_every=ee, eval_fn=ev)
            name = f"lrc_{enc}_s{sd}"
            curves[name] = log
            nc = [m["nc_last"] for m in log]
            lq = float(np.mean(nc[3 * len(nc) // 4:]))
            print(f"  {name}: final={nc[-1]:.4f}, min={min(nc):.4f}, "
                  f"lastQ={lq:.4f} ({time.time()-t0:.0f}s)")
            for m in log:
                out.append({"alg": name, "episode": m["episode"],
                            "nc_last": m["nc_last"]})
    path = os.path.join(RESULTS, "p3f_lrcouple.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alg", "episode", "nc_last"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"  saved {path}")

    # overlay references
    refs = {}
    for fname, want in (("p3e_long.csv", ("tabular", "neural_qboth_s0")),
                        ("p4a_a0_kuhn.csv", ("features_s0",))):
        p = os.path.join(RESULTS, fname)
        if os.path.exists(p):
            with open(p) as f:
                for r in csv.DictReader(f):
                    if r["alg"] in want:
                        refs.setdefault("ref_" + r["alg"], []).append(
                            (int(r["episode"]), float(r["nc_last"])))
    plt.figure(figsize=(8, 5))
    for name, pts in refs.items():
        pts.sort()
        plt.semilogy([p[0] for p in pts], [max(p[1], 1e-12) for p in pts],
                     "--", alpha=0.5, label=name)
    for name, lg in curves.items():
        plt.semilogy([m["episode"] for m in lg],
                     [max(m["nc_last"], 1e-12) for m in lg], lw=1.8,
                     label=name)
    plt.xlabel("episodes"); plt.ylabel("NashConv (last iterate)")
    plt.title("P3f: lr-coupled annealing kills the late rebound?")
    plt.legend(fontsize=7); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3f_lrcouple.png"), dpi=150)
    plt.close()
    print(f"\nALL DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
