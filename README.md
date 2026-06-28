# A-CFR: Anchored-Regularization CFR

A model-free deep counterfactual regret minimization (CFR) method with
last-iterate convergence. A-CFR removes the machinery that standard Deep CFR
depends on (the average-strategy network, the large reservoir buffer, and
history-network storage) and instead converges the last iterate directly,
through an anchored, regularized update. It is built on a small, self-contained
extensive-form game engine (Kuhn poker, Leduc poker, Liar's Dice).

This is research code for a paper in preparation. The tabular and sampling
layers validate the theory; the neural layer and the OpenSpiel baseline
comparison are under active development.

## The update

The per-infoset update is multiplicative, anchored to a slowly moving reference
policy `mu` (snapshot every `K` steps) at a fixed temperature `tau`:

    sigma^{t+1}(a|I)  proportional to
        sigma^t(a|I)^{1/(1+eta*tau)}
        * mu(a|I)^{eta*tau/(1+eta*tau)}
        * exp( eta * q^t(I,a) / (1+eta*tau) )

A fixed anchor gives last-iterate linear convergence to the regularized
equilibrium, with a gap floor of order `O(tau)`. A moving anchor drives the
last-iterate NashConv to zero with no temperature annealing.

## Repository

| path | contents |
|---|---|
| `code/games.py` | extensive-form game engine: Kuhn, Leduc, Liar's Dice |
| `code/exploitability.py` | exact best response and NashConv |
| `code/algorithms.py` | CFR, CFR+, and tabular A-CFR |
| `code/sampling.py` | model-free sampled A-CFR (lambda-estimator) |
| `code/neural_acfr.py` | neural A-CFR |
| `code/features.py` | feature encodings for the three games |
| `code/os_adapter.py` | OpenSpiel adapter (the algorithms run unchanged) |
| `code/reproduce.py` | reproduces the tabular convergence results and figures |
| `code/comparison/` | head-to-head against Deep CFR baselines (NFSP, OSDeepCFR, DREAM, ESCHER, VR-DeepPDCFR+) |
| `paper/` | LaTeX write-up of the theory (statements and proofs, in progress) |

## Quickstart

```bash
pip install -r requirements.txt
python code/reproduce.py --quick     # about 30 seconds: sanity checks plus figures
python code/reproduce.py             # full tabular validation
```

This checks, on Kuhn and Leduc poker, that A-CFR's last iterate converges while
the last iterates of CFR and CFR+ oscillate, and that the gap floor scales as
`O(tau)`. Figures and CSVs are written to `results/`.

The neural method and the OpenSpiel baseline comparison are documented in
[`code/README.md`](code/README.md).

## Author

Maintained by Yichao Wang (https://github.com/axulor). Part of ongoing
collaborative research toward a machine-learning venue.
