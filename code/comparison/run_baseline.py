# -*- coding: utf-8 -*-
"""Run ONE DeepPDCFR baseline on a game, parse its exploitability-vs-episode
curve from stdout, and save to logs_baseline/{algo}_{game}_s{seed}.csv in the
same (episode, exp) format used by run_ours.py. `exp` here is OpenSpiel
exploitability(=NashConv/2), identical metric to run_ours.

Algos: NFSP QPG RPG OSDeepCFR DREAM VRDeepDCFRPlus VRDeepPDCFRPlus
Games (harness names): KuhnPoker LeducPoker LiarsDice5 LiarsDice6
                       GoofSpielImp5 GoofSpielImp6 Battleship_22_3

Usage (deeppdcfr env):
  python run_baseline.py --algo DREAM --game LeducPoker --episodes 1000000 \
                         --seed 0 [--num_traversals 10000]
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(os.path.dirname(HERE)), "baselines",
                    "DeepPDCFR")
SCRIPTS = os.path.join(REPO, "scripts")
LOGDIR = os.path.join(HERE, "logs_baseline")
os.makedirs(LOGDIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--episodes", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_traversals", type=int, default=10000)
    ap.add_argument("--eval_every", type=int, default=30000)
    ap.add_argument("--extra", default="",
                    help="extra sacred overrides, comma-sep key=val "
                         "(e.g. advantage_buffer_size=20000,"
                         "ave_policy_buffer_size=20000)")
    ap.add_argument("--suffix", default="",
                    help="appended to output csv name (e.g. _mem20k)")
    args = ap.parse_args()

    cfg = os.path.join(REPO, "configs", f"{args.algo}.yaml")
    # algo families differ in param names: NFSP/QPG/RPG are RL (no
    # num_traversals; NFSP uses num_train_episodes), CFR-based use
    # num_episodes + num_traversals.
    cmd = [sys.executable, "run.py", "with", cfg,
           f"game_name={args.game}", f"seed={args.seed}",
           "save_log=False"]
    if args.algo == "NFSP":
        cmd += [f"num_train_episodes={args.episodes}",
                f"eval_every={args.eval_every}"]
    elif args.algo in ("QPG", "RPG"):
        cmd += [f"num_episodes={args.episodes}"]
    else:  # OSDeepCFR / DREAM / VRDeepDCFRPlus / VRDeepPDCFRPlus
        cmd += [f"num_episodes={args.episodes}",
                f"num_traversals={args.num_traversals}"]
    if args.extra:
        cmd += [kv for kv in args.extra.split(",") if kv]
    cmd += ["--force"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=SCRIPTS, capture_output=True, text=True,
                          env=env)
    text = proc.stdout + "\n" + proc.stderr
    # parse "| exp ... | <num>" and "| episode ... | <int>" pairs (in order)
    exps = [float(x) for x in re.findall(r"\|\s*exp\s*\|\s*([\d.eE+-]+)", text)]
    eps = [int(x) for x in re.findall(r"\|\s*episode\s*\|\s*(\d+)", text)]
    n = min(len(exps), len(eps))
    if n == 0:
        print(f"  !! {args.algo}/{args.game}/s{args.seed}: NO DATA parsed "
              f"(exit {proc.returncode}). Tail:")
        print("\n".join(text.strip().splitlines()[-15:]))
        sys.exit(1)
    rows = [{"episode": eps[i], "exp": exps[i]} for i in range(n)]
    path = os.path.join(
        LOGDIR, f"{args.algo}_{args.game}_s{args.seed}{args.suffix}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "exp"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    fin = rows[-1]["exp"]
    mn = min(r["exp"] for r in rows)
    print(f"  {args.algo}/{args.game}/s{args.seed}: final={fin:.4f} "
          f"min={mn:.4f} pts={n} ({time.time()-t0:.0f}s) -> {path}")


if __name__ == "__main__":
    main()
