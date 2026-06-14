# -*- coding: utf-8 -*-
"""Comparison campaign orchestrator (Ubuntu, 2026-06-13).

Goal: head-to-head A-CFR (ours, last-iterate) vs the DeepPDCFR baselines
(avg-iterate) on OpenSpiel games, identical exploitability metric and
episode budget. Runs a matrix of (method/algo x game x seed) as single-core
subprocesses with a concurrency cap; each job writes its own
(episode, exp) CSV (logs_ours / logs_baseline). Idempotent: skips jobs whose
CSV already exists, so the campaign can be extended/resumed.

Usage (deeppdcfr env):
  python campaign.py --episodes 1000000 --seeds 0,1 --workers 14 \
      --games KuhnPoker,LeducPoker --algos DREAM,OSDeepCFR,VRDeepPDCFRPlus \
      --ours neural,sampling
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_OURS = os.path.join(HERE, "logs_ours")
LOG_BASE = os.path.join(HERE, "logs_baseline")
RUNLOG = os.path.join(HERE, "logs_run")
os.makedirs(RUNLOG, exist_ok=True)

# harness game name -> OpenSpiel load string (for run_ours)
GAME_OS = {
    "KuhnPoker": "kuhn_poker",
    "LeducPoker": "leduc_poker",
    "LiarsDice5": "liars_dice(numdice=1,dice_sides=5)",
    "LiarsDice6": "liars_dice(numdice=1,dice_sides=6)",
}


def baseline_csv(algo, game, seed):
    return os.path.join(LOG_BASE, f"{algo}_{game}_s{seed}.csv")


def ours_csv(method, game, seed):
    return os.path.join(LOG_OURS, f"ours_{method}_{GAME_OS[game]}_s{seed}.csv")


def run_job(spec):
    kind, name, game, seed, episodes, eval_every = spec
    tag = f"{kind}:{name}/{game}/s{seed}"
    logf = os.path.join(RUNLOG, f"{kind}_{name}_{game}_s{seed}.log")
    if kind == "baseline":
        cmd = [sys.executable, os.path.join(HERE, "run_baseline.py"),
               "--algo", name, "--game", game, "--episodes", str(episodes),
               "--seed", str(seed)]
    else:
        cmd = [sys.executable, os.path.join(HERE, "run_ours.py"),
               "--game", GAME_OS[game], "--method", name,
               "--episodes", str(episodes), "--seed", str(seed),
               "--eval_every", str(eval_every)]
    t0 = time.time()
    with open(logf, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS="1",
                                    MKL_NUM_THREADS="1",
                                    OPENBLAS_NUM_THREADS="1",
                                    PYTHONPATH=os.path.dirname(HERE)))
    return tag, r.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000000)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--games", default="KuhnPoker,LeducPoker")
    ap.add_argument("--algos",
                    default="NFSP,OSDeepCFR,DREAM,VRDeepDCFRPlus,VRDeepPDCFRPlus")
    ap.add_argument("--ours", default="neural,sampling")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--eval_every", type=int, default=20000)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    games = [g for g in args.games.split(",") if g]
    algos = [a for a in args.algos.split(",") if a]
    ours = [m for m in args.ours.split(",") if m]

    jobs = []
    for g in games:
        for s in seeds:
            for a in algos:
                if not os.path.exists(baseline_csv(a, g, s)):
                    jobs.append(("baseline", a, g, s, args.episodes,
                                 args.eval_every))
            for m in ours:
                if not os.path.exists(ours_csv(m, g, s)):
                    jobs.append(("ours", m, g, s, args.episodes,
                                 args.eval_every))
    print(f"campaign: {len(jobs)} jobs (skipped existing), "
          f"workers={args.workers}, episodes={args.episodes}")
    for j in jobs:
        print("   queued", j[0], j[1], j[2], "s%d" % j[3])
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for fut in as_completed(futs):
            tag, rc, dt = fut.result()
            done += 1
            flag = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[{done}/{len(jobs)}] {flag} {tag} ({dt:.0f}s) "
                  f"| elapsed {time.time()-t0:.0f}s", flush=True)
    print(f"campaign done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
