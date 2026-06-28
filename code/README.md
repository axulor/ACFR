# A-CFR code

The core library is flat, so `from games import ...` style imports work when
this directory is on the path. Running `python code/reproduce.py` from the
repository root already puts it on the path.

## Core library

| file | contents |
|---|---|
| `games.py` | extensive-form game tree engine; `build_kuhn`, `build_leduc`, `build_liars_dice` |
| `exploitability.py` | exact best response and NashConv |
| `algorithms.py` | CFR, CFR+, and tabular A-CFR (`run_acfr`) |
| `sampling.py` | model-free sampled A-CFR (`run_sacfr`, lambda-estimator) plus an OS-MCCFR baseline |
| `neural_acfr.py` | neural A-CFR (`NeuralACFR`) |
| `features.py` | generalizing feature encodings for the native games |
| `os_adapter.py` | OpenSpiel state to our EFG tree (`build_openspiel`); algorithms run unchanged |
| `reproduce.py` | tabular validation: last-iterate convergence and the `O(tau)` gap floor, with figures |

## Baseline comparison (`comparison/`)

Head-to-head of A-CFR (last iterate) against Deep CFR baselines on the same
OpenSpiel game and the same exploitability metric (NashConv / 2), with the
x-axis being episodes sampled. The baselines converge the average strategy with
an average network and a large reservoir buffer; A-CFR reports the last iterate
with no average network and a small buffer.

| file | role |
|---|---|
| `run_ours.py` | run A-CFR on an OpenSpiel game, log exploitability vs episodes |
| `run_baseline.py` | run one DeepPDCFR baseline, parse its curve to the same CSV format |
| `run_escher.py` | run the ESCHER baseline |
| `campaign.py` | orchestrate the (method x game x seed) matrix, idempotent |
| `analyze.py` | seed-average, overlay curves, summary metrics, figures |

Baselines: NFSP, OSDeepCFR, DREAM, VR-DeepDCFR+, VR-DeepPDCFR+, ESCHER. This
comparison requires OpenSpiel (`pip install open_spiel`); the tabular
`reproduce.py` does not.
