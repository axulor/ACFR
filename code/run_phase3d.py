# -*- coding: utf-8 -*-
"""Phase 3d: Q-side drift fix (multi-seed ablation).

P3c-C0 diagnosis: policy-distillation residual is tiny (KL ~ 1e-4..1e-7,
shrinks with budget), but q_rmse GROWS over time (0.15 -> 0.7-1.1) in
3 of 4 runs. Mechanism: at lam=1, Q is a control variate -- Q error adds
no bias but inflates estimator variance ~ (V-Q)^2/xi, raising the noise
floor eta*sigma^2/(B*tau) by ~an order of magnitude. Plugging measured
qRMSE ~0.8 in reproduces the observed floors (~0.15 at eta=0.25).
The N1/P3b "schedule pathology" is therefore (at least partly) Q-drift
inflation, not policy distillation.

  D0/D1  at the worst drift point (eta=0.125 fixed, schedule off,
         ds=20, 40k eps, seeds {0,1,2}):
           base        (no stabilizer; reproduces drift)
           q_anchor    (bootstrap from anchor-phase Q snapshot)
           q_replay    (transition replay, fresh bootstrapped targets)
           q_both      (anchor + replay)
  D2     lam=0.5 cross-check (variance damping w/o fixing Q), seeds 3.
  D3     winner at the full 200k superphase schedule, frozen at
         eta*=0.0625 (C1 argmin), vs P3b tabular/naive reference.
         Success: tracks tabular (<= 2x).

Usage:  python run_phase3d.py [--quick]
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

SEEDS = (0, 1, 2)


def save_csv(name, rows, fields):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  saved {path}")


def wmean(seq, lo, hi):
    n = len(seq)
    a, b = int(n * lo), max(int(n * hi), int(n * lo) + 1)
    return float(np.mean(seq[a:b]))


def run_one(kuhn, episodes, ee, ev, seed, **kw):
    agent = NeuralACFR(kuhn, tau=0.1, K_ep=2000, batch_size=16,
                       seed=seed, diag=True, **kw)
    log = agent.run(episodes, eval_every=ee, eval_fn=ev)
    nc = [m["nc_last"] for m in log]
    qr = [d["q_rmse"] for d in agent.diag_log]
    return {
        "floor": wmean(nc, 0.75, 1.0),
        "drift": wmean(nc, 0.75, 1.0) - wmean(nc, 0.45, 0.55),
        "q_rmse_early": wmean(qr, 0.05, 0.2),
        "q_rmse_late": wmean(qr, 0.85, 1.0),
        "log": log, "qr": qr,
    }


def d1_stabilizers(kuhn, quick):
    print("== P3d-D1: Q stabilizers (eta=0.125, schedule off, 3 seeds) ==")
    episodes = 10000 if quick else 40000
    ee = max(1000, episodes // 50)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    variants = [
        ("base", dict()),
        ("q_anchor", dict(q_anchor=True)),
        ("q_replay", dict(q_replay=20000, q_batch=512)),
        ("q_both", dict(q_anchor=True, q_replay=20000, q_batch=512)),
        ("lam05", dict(lam=0.5)),
    ]
    rows, curves, qcurves = [], {}, {}
    for name, kw in variants:
        floors, drifts, qre, qrl = [], [], [], []
        for sd in SEEDS:
            t0 = time.time()
            r = run_one(kuhn, episodes, ee, ev, sd, eta=0.125,
                        superphase=None,
                        **dict(dict(lam=1.0, distill_steps=20), **kw))
            floors.append(r["floor"]); drifts.append(r["drift"])
            qre.append(r["q_rmse_early"]); qrl.append(r["q_rmse_late"])
            if sd == 0:
                curves[name] = r["log"]; qcurves[name] = r["qr"]
            print(f"  {name} seed{sd}: floor={r['floor']:.4f} "
                  f"drift={r['drift']:+.4f} qRMSE {r['q_rmse_early']:.3f}"
                  f"->{r['q_rmse_late']:.3f} ({time.time()-t0:.0f}s)")
        rows.append({"variant": name,
                     "floor_mean": float(np.mean(floors)),
                     "floor_std": float(np.std(floors)),
                     "drift_mean": float(np.mean(drifts)),
                     "q_rmse_early": float(np.mean(qre)),
                     "q_rmse_late": float(np.mean(qrl))})
        print(f"  -> {name}: floor={rows[-1]['floor_mean']:.4f}"
              f"+-{rows[-1]['floor_std']:.4f}, "
              f"drift={rows[-1]['drift_mean']:+.4f}, "
              f"qRMSE_late={rows[-1]['q_rmse_late']:.3f}")
    save_csv("p3d_d1_stabilizers.csv", rows,
             ["variant", "floor_mean", "floor_std", "drift_mean",
              "q_rmse_early", "q_rmse_late"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in curves:
        ax[0].semilogy([m["episode"] for m in curves[name]],
                       [max(m["nc_last"], 1e-12) for m in curves[name]],
                       label=name)
        ax[1].plot(np.linspace(0, episodes, len(qcurves[name])),
                   qcurves[name], label=name, alpha=0.8)
    ax[0].set_title("NashConv (seed 0)"); ax[0].set_xlabel("episodes")
    ax[1].set_title("q_rmse vs exact Q^sigma (seed 0)")
    ax[1].set_xlabel("episodes")
    for a in ax:
        a.legend(fontsize=8); a.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3d_D1_stabilizers.png"),
                dpi=150)
    plt.close()
    return rows


def d3_final(kuhn, quick, rows):
    # winner = lowest floor among true Q stabilizers (lam05 is a
    # cross-check, not a candidate fix: it changes the estimator)
    cand = [r for r in rows if r["variant"] in
            ("q_anchor", "q_replay", "q_both")]
    win = min(cand, key=lambda r: r["floor_mean"])["variant"]
    base = next(r for r in rows if r["variant"] == "base")
    print(f"== P3d-D3: winner={win} at full schedule, frozen eta*=0.0625 ==")
    if min(c["floor_mean"] for c in cand) > 0.9 * base["floor_mean"]:
        print("  WARNING: no stabilizer beat base by >10%; "
              "running winner anyway for the record.")
    episodes = 30000 if quick else 200000
    ee = max(2000, episodes // 100)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    kwmap = {"q_anchor": dict(q_anchor=True),
             "q_replay": dict(q_replay=20000, q_batch=512),
             "q_both": dict(q_anchor=True, q_replay=20000, q_batch=512)}
    out, curves = [], {}
    # three arms: frozen@eta* (anneal 0.5->0.0625 then freeze; NEW K
    # semantics: K freezes with eta), naive schedule (anneal forever),
    # const eta* (no schedule at all, K=2000 fixed) -- the last one
    # tests the path-dependence hypothesis from P3c-C3: passing through
    # the mid-eta drift region poisons the nets even if eta ends small.
    arms = [
        (f"{win}_frozen", dict(dict(eta=0.5, superphase=4, eta_decay=0.5,
                                    eta_min=0.0625), **kwmap[win])),
        (f"{win}_naive_sched", dict(dict(eta=0.5, superphase=4,
                                         eta_decay=0.5, eta_min=0.0),
                                    **kwmap[win])),
        (f"{win}_const_etastar", dict(dict(eta=0.0625, superphase=None),
                                      **kwmap[win])),
        ("none_const_etastar", dict(eta=0.0625, superphase=None)),
    ]
    for name, extra in arms:
        kw = dict(lam=1.0, distill_steps=20)
        kw.update(extra)
        t0 = time.time()
        agent = NeuralACFR(kuhn, tau=0.1, K_ep=2000, batch_size=16,
                           seed=0, **kw)
        log = agent.run(episodes, eval_every=ee, eval_fn=ev)
        curves[name] = log
        fl = float(np.mean([m["nc_last"] for m in log[-10:]]))
        print(f"  {name}: final={log[-1]['nc_last']:.4f} floor={fl:.4f} "
              f"({time.time()-t0:.0f}s)")
        for m in log:
            out.append({"alg": name, "episode": m["episode"],
                        "nc_last": m["nc_last"]})
    save_csv("p3d_d3_final.csv", out, ["alg", "episode", "nc_last"])
    ref = {}
    b1 = os.path.join(RESULTS, "p3b_b1_schedules.csv")
    if os.path.exists(b1):
        with open(b1) as f:
            for r in csv.DictReader(f):
                if r["alg"] in ("tabular", "neural_naive"):
                    ref.setdefault(r["alg"], []).append(
                        (int(r["episode"]), float(r["nc_last"])))
    plt.figure(figsize=(8, 5))
    for name, pts in ref.items():
        pts.sort()
        plt.semilogy([p[0] for p in pts], [max(p[1], 1e-12) for p in pts],
                     "--", alpha=0.6, label=name + " (P3b)")
    for name, lg in curves.items():
        plt.semilogy([m["episode"] for m in lg],
                     [max(m["nc_last"], 1e-12) for m in lg], lw=2,
                     label=name)
    plt.xlabel("episodes"); plt.ylabel("NashConv (last iterate)")
    plt.title("P3d-D3: Q-stabilized neural A-CFR vs P3b baselines")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3d_D3_final.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    kuhn = build_kuhn()
    rows = d1_stabilizers(kuhn, args.quick)
    d3_final(kuhn, args.quick, rows)
    print(f"\nALL DONE in {time.time()-t0:.0f}s. Results in "
          f"{os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
