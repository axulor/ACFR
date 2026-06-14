# A-CFR code — structure

Reorganized 2026-06-13 (Ubuntu phase). Core library is **flat** (so all
`from games import ...` style imports keep working); everything else is in
topic subdirs. Run any script with the core dir on the path:

```bash
export PYTHONPATH=/home/axulor/ACFR/code
```

## Environments (conda)
- **acfr** — our algorithm dev/diagnostics. Python 3.10, numpy, torch 2.11+cu128
  (CUDA), matplotlib. `conda activate acfr`.
- **deeppdcfr** — baseline harness + the head-to-head comparison (our code
  also runs here). Python 3.9, open_spiel 1.4, tensorflow 2.15, torch 2.8,
  sacred. `conda activate deeppdcfr`. Built from `baselines/DeepPDCFR`.

## Core library (flat, importable)
| file | contents |
|---|---|
| `games.py` | EFG tree engine; build_kuhn / build_leduc / build_liars_dice |
| `exploitability.py` | exact best-response + NashConv |
| `algorithms.py` | CFR / CFR+ / tabular A-CFR (`run_acfr`) |
| `sampling.py` | sampled A-CFR (λ-estimator, `run_sacfr`) + OS-MCCFR |
| `neural_acfr.py` | neural A-CFR (`NeuralACFR`); recipe switches incl. `anchor_ema` |
| `features.py` | generalizing feature encodings (native games) |
| `os_adapter.py` | **OpenSpiel → our EFG tree** (`build_openspiel`); algos run unchanged |

## Subdirs
- `experiments/` — historical phase drivers `run_phase0..4c.py` (research log
  docs `01..23` in repo root explain each).
- `diagnostics/` — the torch-2.11 last-iterate dig: `diag_rebound.py`,
  `diag_ablate.py`, `diag_leduc.py`, `run_leduc_reverify.py`.
- `comparison/` — **head-to-head vs baselines (current focus)**:
  - `run_ours.py` — A-CFR (last-iterate) on an OpenSpiel game; logs
    exploitability (=NashConv/2, OpenSpiel metric) vs episodes.
  - `run_baseline.py` — runs one DeepPDCFR baseline, parses its
    exploitability curve to the same CSV format.
  - `campaign.py` — orchestrates the (method × game × seed) matrix with a
    concurrency cap; idempotent (skips existing CSVs).
  - `analyze.py` — overlays curves, seed-averages, summary metrics, figures.
  - outputs: `logs_ours/`, `logs_baseline/`, `logs_run/`, `figs/`.

## The comparison (goal)
Head-to-head: **A-CFR last-iterate** vs the DeepPDCFR baselines
(NFSP, OSDeepCFR, DREAM, VRDeepDCFRPlus, VRDeepPDCFRPlus = AAAI'26 SOTA) and
ESCHER — same OpenSpiel game, same exploitability metric, x-axis = episodes
sampled. Baselines converge the **average** strategy (+ average net + 1e6
reservoir buffer); A-CFR reports the **last** iterate (no average net, 2e4
buffer). See `baselines/` (gitignored) and repo memory.
