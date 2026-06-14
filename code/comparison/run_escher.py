# -*- coding: utf-8 -*-
"""Run ESCHER (McAleer et al. 2023) baseline on an OpenSpiel game, logging
exploitability(=NashConv/2) vs episodes-sampled, same CSV format as the
other baselines/ours.

ESCHER is a standalone TF solver (baselines/ESCHER/ESCHER.py) with its own
interface (per-iteration). It evaluates the AVERAGE policy's nash_conv every
`check_every` iterations. Per iteration it samples:
    num_val_fn_traversals + 20 (value)  +  2 * num_traversals (regret, 2p)
trajectories; we count ALL of them as episodes (total game samples), the
honest common unit. ESCHER's conv = nash_conv (sum) -> divide by 2 to match
the exploitability metric used everywhere else.

Networks set to the 3x64 protocol (matching the DeepPDCFR baselines).

Usage (deeppdcfr env):
  python run_escher.py --game kuhn_poker --episodes 1000000 --seed 0
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ESCHER_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                          "baselines", "ESCHER")
sys.path.insert(0, ESCHER_DIR)
LOGDIR = os.path.join(HERE, "logs_baseline")
os.makedirs(LOGDIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="kuhn_poker")
    ap.add_argument("--harness_game", default="",
                    help="name used in the output csv (harness convention, "
                         "e.g. KuhnPoker); defaults from --game")
    ap.add_argument("--episodes", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_traversals", type=int, default=1000)
    ap.add_argument("--num_val_traversals", type=int, default=1000)
    ap.add_argument("--check_every", type=int, default=2)
    args = ap.parse_args()

    import random
    import numpy as _np
    import tensorflow as tf
    random.seed(args.seed); _np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    import pyspiel
    from ESCHER import ESCHERSolver

    game = pyspiel.load_game(args.game)
    ep_per_iter = args.num_val_traversals + 20 + 2 * args.num_traversals
    n_iter = max(1, args.episodes // ep_per_iter)
    print(f"ESCHER {args.game}: ep/iter={ep_per_iter}, n_iter={n_iter}, "
          f"check_every={args.check_every}, seed={args.seed}")

    solver = ESCHERSolver(
        game,
        policy_network_layers=(64, 64, 64),
        regret_network_layers=(64, 64, 64),
        value_network_layers=(64, 64, 64),
        num_iterations=n_iter,
        num_traversals=args.num_traversals,
        num_val_fn_traversals=args.num_val_traversals,
        learning_rate=1e-3,
        # train steps / batch matched to the DeepPDCFR protocol scale
        # (defaults are huge: 15000/5000/4048). policy net trained only at
        # exploitability checks.
        regret_network_train_steps=750,
        value_network_train_steps=1000,
        policy_network_train_steps=2000,
        batch_size_regret=2048,
        batch_size_value=2048,
        batch_size_average_policy=2048,
        check_exploitability_every=args.check_every,
        compute_exploitability=True,
        save_policy_weights=False,
        train_device="cpu", infer_device="cpu",
    )
    t0 = time.time()
    _r, _pl, convs, nodes = solver.solve()
    # convs[k] recorded at iteration i_k = k*check_every (i from 0). episodes
    # up to iteration i (after the i=0 warmup traverse(1) + (i+1) iters):
    rows = []
    for k, c in enumerate(convs):
        i = k * args.check_every
        episode = 1 + (i + 1) * ep_per_iter
        rows.append({"episode": int(episode), "exp": float(c) / 2.0})
    hg = args.harness_game or {
        "kuhn_poker": "KuhnPoker", "leduc_poker": "LeducPoker"}.get(
        args.game, args.game)
    path = os.path.join(LOGDIR, f"ESCHER_{hg}_s{args.seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "exp"]); w.writeheader()
        for r in rows:
            w.writerow(r)
    if rows:
        print(f"  ESCHER/{hg}/s{args.seed}: final={rows[-1]['exp']:.4f} "
              f"min={min(r['exp'] for r in rows):.4f} pts={len(rows)} "
              f"({time.time()-t0:.0f}s) -> {path}")


if __name__ == "__main__":
    main()
