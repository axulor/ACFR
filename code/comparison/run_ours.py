# -*- coding: utf-8 -*-
"""Run A-CFR (ours) on an OpenSpiel game, logging exploitability vs episodes
in the SAME metric/protocol as the DeepPDCFR baseline harness.

The baseline harness logs `exp = open_spiel ... exploitability(game, policy)`
= NashConv/2. We convert our last-iterate strategy sigma to an OpenSpiel
TabularPolicy and call the identical function, so curves are apples-to-apples
(same env, same game, same metric, x-axis = episodes/trajectories sampled).

We report the LAST ITERATE (A-CFR's whole point: no average-strategy network).
Methods:
  neural   : NeuralACFR with the locked theory-aligned recipe
             (continued annealing + soft anchor + B64 + Q-stabilizers).
  sampling : tabular sampled A-CFR (run_sacfr) -- last iterate.

Usage (deeppdcfr env):
  python run_ours.py --game leduc_poker --method neural --episodes 1000000 \
                     --seed 0 --eval_every 20000
Games: kuhn_poker, leduc_poker, liars_dice, liars_dice(dice_sides=N), ...
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # code/ for core modules

import pyspiel
from open_spiel.python import policy as os_policy
from open_spiel.python.algorithms import exploitability as os_exp

from os_adapter import build_openspiel, openspiel_feature_fn
from neural_acfr import NeuralACFR
from sampling import run_sacfr

LOGDIR = os.path.join(HERE, "logs_ours")
os.makedirs(LOGDIR, exist_ok=True)

# locked theory-aligned neural recipe (doc: anneal + soft anchor + B64).
# K_ep is the eps_k inner-accuracy knob; scales with game depth -- use a
# per-game default (Leduc-class deep games like a larger K).
NEURAL = dict(eta=0.5, tau=0.1, batch_size=64, superphase=4, eta_decay=0.5,
              eta_min=0.0, lam=0.5, distill_steps=20,
              lr_couple=True, q_anchor=True, q_replay=20000, q_batch=512,
              anchor_ema=0.5)
SAMPLING = dict(eta=0.5, tau=0.1, batch_size=64, superphase=4,
                eta_decay=0.5, lam=0.5)


def make_evaluator(game, g):
    """Returns eval_fn(sigma)-> exp (=NashConv/2, == baseline `exp`).
    Builds info_state -> legal-action-ints map from our adapter tree; our
    sigma[I] is ordered by legal_actions() so columns align."""
    info_legal = {}
    for n in _decision_nodes(g):
        info_legal[n.infoset] = n.meta["os_actions"]

    def to_tabular(sigma):
        tp = os_policy.TabularPolicy(game)
        for info_state, probs in sigma.items():
            if info_state not in tp.state_lookup:
                continue
            row = tp.state_lookup[info_state]
            arr = tp.action_probability_array[row]
            arr[:] = 0.0
            for k, a in enumerate(info_legal[info_state]):
                arr[a] = probs[k]
        return tp

    def evaluate(sigma):
        return float(os_exp.exploitability(game, to_tabular(sigma)))

    return evaluate


def _decision_nodes(g):
    from games import Terminal, Chance
    out = []
    seen = set()
    def rec(node):
        if isinstance(node, Terminal):
            return
        if isinstance(node, Chance):
            for _, c in node.children:
                rec(c)
            return
        if id(node) not in seen:
            seen.add(id(node))
            out.append(node)
        for c in node.children:
            rec(c)
    rec(g.root)
    return out


def k_for_game(game_name, n_iset):
    # eps_k condition: deeper/noisier games need slower anchor. Heuristic
    # by infoset count (Kuhn 12 -> 2000; Leduc 936 -> 8000; large -> 16000).
    if n_iset <= 100:
        return 2000
    if n_iset <= 2000:
        return 8000
    return 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="leduc_poker")
    ap.add_argument("--method", default="neural",
                    choices=["neural", "sampling"])
    ap.add_argument("--episodes", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=20000)
    ap.add_argument("--K", type=int, default=0, help="override anchor period")
    ap.add_argument("--encoding", default="onehot",
                    choices=["onehot", "features"],
                    help="neural input: onehot (tabular-like) or features "
                         "(OpenSpiel info_state_tensor, generalizes)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    game = pyspiel.load_game(args.game)
    g = build_openspiel(args.game,
                        with_features=(args.method == "neural"
                                       and args.encoding == "features"))
    n_iset = len(g.infosets)
    K = args.K or k_for_game(args.game, n_iset)
    evaluate = make_evaluator(game, g)
    print(f"{args.game}: {n_iset} isets, K={K}, method={args.method}, "
          f"episodes={args.episodes}, seed={args.seed}")

    rows = []
    t0 = time.time()

    if args.method == "neural":
        recipe = dict(NEURAL, encoding=args.encoding)
        if args.encoding == "features":
            recipe["feature_fn"] = openspiel_feature_fn(g)
        agent = NeuralACFR(g, K_ep=K, seed=args.seed, **recipe)
        ev = lambda s, _a: {"exp": evaluate(s)}
        log = agent.run(args.episodes, eval_every=args.eval_every, eval_fn=ev)
        for m in log:
            rows.append({"episode": m["episode"], "exp": m["exp"]})
    else:
        ev = lambda s, _a: {"exp": evaluate(s)}
        _, log = run_sacfr(g, args.episodes, eval_every=args.eval_every,
                           eval_fn=ev, seed=args.seed, K_ep=K, **SAMPLING)
        for m in log:
            rows.append({"episode": m["episode"], "exp": m["exp"]})

    exps = [r["exp"] for r in rows]
    lq = float(np.mean(exps[3 * len(exps) // 4:])) if exps else float("nan")
    print(f"  done: final exp={exps[-1]:.5f} min={min(exps):.5f} "
          f"lastQ={lq:.5f} ({time.time()-t0:.0f}s)")

    enc_sfx = "_feat" if (args.method == "neural"
                          and args.encoding == "features") else ""
    tag = args.tag or f"ours_{args.method}{enc_sfx}"
    path = os.path.join(LOGDIR, f"{tag}_{args.game}_s{args.seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "exp"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
