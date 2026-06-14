# -*- coding: utf-8 -*-
"""Phase 3c: measure the REAL distillation error delta and fix the
neural schedule on principled grounds (see 16_P3b_结果判读与P3c设计.md).

P3b verdict: naive pathology reproduced; but frozen kept creeping up
(slow drift even with eta frozen / schedule off) and matched plateaued
at 0.33 (delta ~ 1/steps is wrong). Hypotheses:
  H1  delta_opt(steps) is sublinear / saturating
  H2  the slow drift lives on the policy-distillation side (replay or
      bigger budget kills it)
  H3  floor(eta) is U-shaped -> freeze at the interior optimum eta*

  C0  delta scaling: schedule off, eta=.25, distill in {5,20,80,320},
      diag on. Measures per-update KL fit residual (the actual delta),
      its time drift, interference at non-target infosets, exact-Q RMSE.
  C1  U-curve: schedule off, distill=20, eta in {1,.5,.25,.125,.0625}.
  C2  replay ablation at eta=.125: base20 vs replay20 vs base80.
  C3  final 200k schedule run: frozen@eta* (picked from C1) and
      replay+frozen@eta*, plotted against P3b's tabular/naive curves.

Usage:  python run_phase3c.py [--quick]
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


def save_csv(name, rows, fields):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  saved {path}")


def floor_of(log, frac=0.25):
    vals = [m["nc_last"] for m in log]
    k = max(1, int(len(vals) * frac))
    return float(np.mean(vals[-k:]))


def window_mean(seq, lo_frac, hi_frac):
    n = len(seq)
    lo, hi = int(n * lo_frac), max(int(n * hi_frac), int(n * lo_frac) + 1)
    return float(np.mean(seq[lo:hi]))


def c0_delta_scaling(kuhn, quick):
    print("== P3c-C0: delta scaling & drift (schedule off, eta=0.25) ==")
    episodes = 10000 if quick else 40000
    ee = max(1000, episodes // 50)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    out, diag_rows, curves = [], [], {}
    for ds in (5, 20, 80, 320):
        t0 = time.time()
        agent = NeuralACFR(kuhn, eta=0.25, tau=0.1, K_ep=2000,
                           batch_size=16, superphase=None, lam=1.0,
                           distill_steps=ds, diag=True)
        log = agent.run(episodes, eval_every=ee, eval_fn=ev)
        curves[ds] = log
        kl = [d["kl_fit"] for d in agent.diag_log]
        qr = [d["q_rmse"] for d in agent.diag_log]
        inter = [d["interference_l1"] for d in agent.diag_log]
        row = {
            "distill_steps": ds,
            "kl_early": window_mean(kl, 0.10, 0.25),
            "kl_mid": window_mean(kl, 0.45, 0.55),
            "kl_late": window_mean(kl, 0.85, 1.00),
            "interference_late": window_mean(inter, 0.85, 1.00),
            "q_rmse_late": window_mean(qr, 0.85, 1.00),
            "floor": floor_of(log),
        }
        row["kl_drift"] = row["kl_late"] - row["kl_mid"]
        out.append(row)
        for d in agent.diag_log:
            d2 = dict(d)
            d2["run_ds"] = ds
            diag_rows.append(d2)
        print(f"  ds={ds}: kl_late={row['kl_late']:.5f} "
              f"(drift {row['kl_drift']:+.5f}), interf={row['interference_late']:.5f}, "
              f"qRMSE={row['q_rmse_late']:.4f}, floor={row['floor']:.4f} "
              f"({time.time()-t0:.0f}s)")
    # scaling exponent alpha: kl_late ~ steps^-alpha (first three points
    # + full fit; saturation shows as flattening at large steps)
    dss = np.array([r["distill_steps"] for r in out], dtype=float)
    kls = np.array([r["kl_late"] for r in out], dtype=float)
    alpha_all = -np.polyfit(np.log(dss), np.log(np.maximum(kls, 1e-12)), 1)[0]
    alpha_pair = [float(-(np.log(kls[i + 1]) - np.log(kls[i]))
                        / (np.log(dss[i + 1]) - np.log(dss[i])))
                  for i in range(len(dss) - 1)]
    print(f"  alpha(all)={alpha_all:.3f}; pairwise alphas={['%.3f' % a for a in alpha_pair]}")
    save_csv("p3c_c0_delta.csv", out,
             ["distill_steps", "kl_early", "kl_mid", "kl_late", "kl_drift",
              "interference_late", "q_rmse_late", "floor"])
    save_csv("p3c_c0_diag_full.csv", diag_rows,
             ["run_ds", "episode", "eta", "distill_steps", "kl_fit",
              "l1_fit", "interference_l1", "q_rmse"])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for ds in curves:
        d = [r for r in diag_rows if r["run_ds"] == ds]
        ep = [r["episode"] for r in d]
        ax[0].semilogy(ep, [max(r["kl_fit"], 1e-12) for r in d],
                       label=f"ds={ds}", alpha=0.7)
        ax[1].semilogy([m["episode"] for m in curves[ds]],
                       [max(m["nc_last"], 1e-12) for m in curves[ds]],
                       label=f"ds={ds}")
    ax[0].set_title("per-update distill residual KL (delta)")
    ax[0].set_xlabel("episodes"); ax[0].legend(fontsize=8)
    ax[1].set_title("NashConv (last iterate)")
    ax[1].set_xlabel("episodes"); ax[1].legend(fontsize=8)
    ax[2].loglog(dss, kls, "o-", label="kl_late")
    ax[2].loglog(dss, kls[0] * (dss / dss[0]) ** (-1.0), "k--",
                 alpha=0.5, label="slope -1 (assumed by matched)")
    ax[2].set_title(f"delta vs steps (alpha={alpha_all:.2f})")
    ax[2].set_xlabel("distill steps"); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3c_C0_delta.png"), dpi=150)
    plt.close()
    return out


def c1_eta_ucurve(kuhn, quick):
    print("== P3c-C1: floor(eta) U-curve (schedule off, ds=20) ==")
    episodes = 15000 if quick else 60000
    ee = max(1000, episodes // 50)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    etas = (1.0, 0.5, 0.25, 0.125, 0.0625)
    out, curves = [], {}
    for eta in etas:
        t0 = time.time()
        agent = NeuralACFR(kuhn, eta=eta, tau=0.1, K_ep=2000,
                           batch_size=16, superphase=None, lam=1.0,
                           distill_steps=20)
        log = agent.run(episodes, eval_every=ee, eval_fn=ev)
        curves[eta] = log
        fl = floor_of(log)
        half = window_mean([m["nc_last"] for m in log], 0.45, 0.55)
        out.append({"eta": eta, "floor": fl, "drift": fl - half})
        print(f"  eta={eta}: floor={fl:.4f} (drift {fl-half:+.4f}) "
              f"({time.time()-t0:.0f}s)")
    save_csv("p3c_c1_ucurve.csv", out, ["eta", "floor", "drift"])
    eta_star = min(out, key=lambda r: r["floor"])["eta"]
    print(f"  eta* = {eta_star}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for eta in etas:
        ax[0].semilogy([m["episode"] for m in curves[eta]],
                       [max(m["nc_last"], 1e-12) for m in curves[eta]],
                       label=f"eta={eta}")
    ax[0].set_title("NashConv, fixed eta (schedule off)")
    ax[0].set_xlabel("episodes"); ax[0].legend(fontsize=8)
    ax[1].loglog([r["eta"] for r in out], [r["floor"] for r in out], "o-")
    ax[1].axvline(eta_star, color="r", ls=":", label=f"eta*={eta_star}")
    ax[1].set_title("floor vs eta (U-curve, Prop-4 corollary)")
    ax[1].set_xlabel("eta"); ax[1].set_ylabel("floor"); ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3c_C1_ucurve.png"), dpi=150)
    plt.close()
    return eta_star


def c2_replay(kuhn, quick):
    print("== P3c-C2: target-replay ablation (eta=0.125, schedule off) ==")
    episodes = 15000 if quick else 60000
    ee = max(1000, episodes // 50)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    variants = [
        ("base_ds20", dict(distill_steps=20)),
        ("replay_ds20", dict(distill_steps=20, replay_targets=True)),
        ("base_ds80", dict(distill_steps=80)),
        ("replay_ds80", dict(distill_steps=80, replay_targets=True)),
    ]
    out, curves = [], {}
    for name, kw in variants:
        t0 = time.time()
        agent = NeuralACFR(kuhn, eta=0.125, tau=0.1, K_ep=2000,
                           batch_size=16, superphase=None, lam=1.0,
                           diag=True, **kw)
        log = agent.run(episodes, eval_every=ee, eval_fn=ev)
        curves[name] = log
        kl = [d["kl_fit"] for d in agent.diag_log]
        fl = floor_of(log)
        half = window_mean([m["nc_last"] for m in log], 0.45, 0.55)
        out.append({"variant": name, "floor": fl, "drift": fl - half,
                    "kl_late": window_mean(kl, 0.85, 1.0),
                    "kl_drift": window_mean(kl, 0.85, 1.0)
                                - window_mean(kl, 0.45, 0.55)})
        print(f"  {name}: floor={fl:.4f} (drift {fl-half:+.4f}, "
              f"kl_late={out[-1]['kl_late']:.5f}) ({time.time()-t0:.0f}s)")
    save_csv("p3c_c2_replay.csv", out,
             ["variant", "floor", "drift", "kl_late", "kl_drift"])
    plt.figure(figsize=(8, 5))
    for name, lg in curves.items():
        plt.semilogy([m["episode"] for m in lg],
                     [max(m["nc_last"], 1e-12) for m in lg], label=name)
    plt.xlabel("episodes"); plt.ylabel("NashConv (last iterate)")
    plt.title("P3c-C2: target replay vs budget (eta=0.125 fixed)")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3c_C2_replay.png"), dpi=150)
    plt.close()
    return out


def c3_final(kuhn, quick, eta_star, replay_helps):
    print(f"== P3c-C3: final schedule run (freeze at eta*={eta_star}, "
          f"replay={'on' if replay_helps else 'off'} variant included) ==")
    episodes = 30000 if quick else 200000
    ee = max(2000, episodes // 100)
    ev = lambda s, _a: {"nc_last": nash_conv(kuhn, s)[0]}
    eta_star = min(eta_star, 0.5)
    base = dict(eta=0.5, tau=0.1, K_ep=2000, batch_size=16,
                superphase=4, eta_decay=0.5, lam=1.0)
    variants = [("frozen_star", dict(distill_steps=20, eta_min=eta_star))]
    if replay_helps:
        variants.append(("replay_frozen_star",
                         dict(distill_steps=20, eta_min=eta_star,
                              replay_targets=True)))
    curves, out = {}, []
    for name, kw in variants:
        t0 = time.time()
        agent = NeuralACFR(kuhn, **base, **kw)
        log = agent.run(episodes, eval_every=ee, eval_fn=ev)
        curves[name] = log
        print(f"  {name}: final={log[-1]['nc_last']:.4f}, "
              f"floor={floor_of(log, 0.1):.4f} ({time.time()-t0:.0f}s)")
        for m in log:
            out.append({"alg": name, "episode": m["episode"],
                        "nc_last": m["nc_last"]})
    save_csv("p3c_c3_final.csv", out, ["alg", "episode", "nc_last"])

    # overlay P3b reference curves (same seeds/config)
    ref = {}
    b1 = os.path.join(RESULTS, "p3b_b1_schedules.csv")
    if os.path.exists(b1):
        with open(b1) as f:
            for r in csv.DictReader(f):
                if r["alg"] in ("tabular", "neural_naive", "neural_matched"):
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
    plt.title("P3c-C3: principled freeze at eta* vs P3b baselines")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figP3c_C3_final.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    kuhn = build_kuhn()
    c0_delta_scaling(kuhn, args.quick)
    eta_star = c1_eta_ucurve(kuhn, args.quick)
    c2 = c2_replay(kuhn, args.quick)
    base = {r["variant"]: r for r in c2}
    replay_helps = (base["replay_ds20"]["floor"]
                    < 0.85 * base["base_ds20"]["floor"])
    c3_final(kuhn, args.quick, eta_star, replay_helps)
    print(f"\nALL DONE in {time.time()-t0:.0f}s. Results in "
          f"{os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
