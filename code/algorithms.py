# -*- coding: utf-8 -*-
"""Tabular algorithms: CFR, CFR+, and A-CFR (anchored-regularization CFR).

A-CFR per-infoset update (full feedback, simultaneous):

    sigma^{t+1}(a|I)  ∝  sigma^t(a|I)^{1/(1+eta*tau)}
                       * mu(a|I)^{eta*tau/(1+eta*tau)}
                       * exp( eta * q^t(I,a) / (1+eta*tau) )

where q^t(I,a) is the (unnormalized, opponent+chance reach weighted)
counterfactual action value — i.e. exactly the sequence-form gradient slice.
Anchor mu: 'fixed' | 'periodic' (hard reset every K iters) | 'ema'.
"""
import math
import numpy as np
from games import Terminal, Chance, Decision


# ---------------------------------------------------------------- utilities

def uniform_strategy(game):
    return {I: np.ones(na) / na for I, (p, na) in game.infosets.items()}


def cf_values(game, sigma, player):
    """One full traversal. Returns:
    q:     {infoset of `player` -> np.array of unnormalized counterfactual
            action values (opponent+chance reach weighted, summed over h)}
    reach: {infoset of `player` -> own reach prob (summed over h; identical
            across h in perfect recall, but summing is the standard form)}
    reach_opp: {infoset -> total opponent*chance reach mass; normalizer
            turning q into advantage-scale values}
    v_root: expected utility of the profile for `player`.
    """
    q = {}
    reach = {}
    reach_opp = {}

    def rec(node, p0, p1, pc):
        if isinstance(node, Terminal):
            return node.u if player == 0 else -node.u
        if isinstance(node, Chance):
            return sum(pr * rec(c, p0, p1, pc * pr)
                       for pr, c in node.children)
        s = sigma[node.infoset]
        if node.player == player:
            own = p0 if player == 0 else p1
            opp = p1 if player == 0 else p0
            vals = []
            for a, c in enumerate(node.children):
                if player == 0:
                    vals.append(rec(c, p0 * s[a], p1, pc))
                else:
                    vals.append(rec(c, p0, p1 * s[a], pc))
            vals = np.asarray(vals)
            w = opp * pc
            if node.infoset not in q:
                q[node.infoset] = np.zeros(len(vals))
                reach[node.infoset] = 0.0
                reach_opp[node.infoset] = 0.0
            q[node.infoset] += w * vals
            reach[node.infoset] += own
            reach_opp[node.infoset] += w
            return float(np.dot(s, vals))
        else:
            v = 0.0
            for a, c in enumerate(node.children):
                if s[a] == 0.0:
                    continue
                if node.player == 0:
                    v += s[a] * rec(c, p0 * s[a], p1, pc)
                else:
                    v += s[a] * rec(c, p0, p1 * s[a], pc)
            return v

    v_root = rec(game.root, 1.0, 1.0, 1.0)
    return q, reach, reach_opp, v_root


def _regret_matching(R):
    pos = np.maximum(R, 0.0)
    s = pos.sum()
    if s <= 0:
        return np.ones(len(R)) / len(R)
    return pos / s


def normalize_avg(avg):
    out = {}
    for I, c in avg.items():
        s = c.sum()
        out[I] = c / s if s > 0 else np.ones(len(c)) / len(c)
    return out


# ---------------------------------------------------------------- CFR / CFR+

def run_cfr(game, iters, plus=False, noise_std=0.0, seed=0,
            eval_every=10, eval_fn=None):
    """Vanilla CFR (simultaneous, uniform avg) or CFR+ (alternating,
    RM+, linear averaging). eval_fn(sigma_last, sigma_avg) -> dict of
    metrics, called every eval_every iters.

    noise_std: additive Gaussian noise on counterfactual values, injected
    at the SAME normalized scale as in run_acfr (i.e. scaled back by
    reach_opp * max|u|), for fair robustness comparisons."""
    sigma = uniform_strategy(game)
    R = {I: np.zeros(na) for I, (p, na) in game.infosets.items()}
    avg = {I: np.zeros(na) for I, (p, na) in game.infosets.items()}
    rng = np.random.default_rng(seed)
    u_max = game.max_abs_u
    log = []

    def noisy(qa, ro_I):
        if noise_std <= 0:
            return qa
        return qa + noise_std * u_max * ro_I * rng.standard_normal(len(qa))

    for t in range(1, iters + 1):
        if plus:
            for p in (0, 1):
                q, reach, ro, _ = cf_values(game, sigma, p)
                for I, qa in q.items():
                    qa = noisy(qa, ro[I])
                    v = float(np.dot(sigma[I], qa))
                    R[I] = np.maximum(R[I] + (qa - v), 0.0)
                    sigma[I] = _regret_matching(R[I])
                    avg[I] += t * reach[I] * sigma[I]
        else:
            updates = []
            for p in (0, 1):
                q, reach, ro, _ = cf_values(game, sigma, p)
                for I, qa in q.items():
                    qa = noisy(qa, ro[I])
                    v = float(np.dot(sigma[I], qa))
                    updates.append((I, qa - v, reach[I]))
            for I, r, rch in updates:
                avg[I] += rch * sigma[I]
                R[I] += r
            for I, r, rch in updates:
                sigma[I] = _regret_matching(R[I])
        if eval_fn is not None and (t % eval_every == 0 or t == 1):
            m = eval_fn(sigma, normalize_avg(avg))
            m["iter"] = t
            log.append(m)
    return sigma, normalize_avg(avg), log


# ---------------------------------------------------------------- A-CFR

def run_acfr(game, iters, eta=0.1, tau=0.5,
             anchor_mode="fixed", K=100, ema_beta=None,
             normalize=False, scale_utilities=False, alternating=False,
             tol0=0.02, tol_decay=0.5, tol_min=1e-10, max_phase=2000,
             superphase=None, eta_decay=0.5,
             noise_std=0.0, seed=0,
             eval_every=10, eval_fn=None, snapshot_every=None,
             sigma0=None, eps=1e-15):
    """Anchored-regularization CFR.

    noise_std: additive Gaussian noise on the (normalized-scale) values,
    for the P1-A oracle robustness experiments.

    normalize=True: divide q(I,.) by the infoset's opponent+chance reach
    mass (advantage scale; = balanced dilated weights). The normalizer is
    FROZEN within an anchor phase and refreshed whenever the anchor moves,
    keeping the within-phase analysis clean.

    scale_utilities=True: additionally divide q by max|u| of the game, so
    normalized values live in [-1,1] (L~1). Required for stability of the
    inner loop when game utilities are large (e.g. Leduc max 13): the
    contraction condition is roughly eta <= tau*c/L^2. Evaluation metrics
    stay in ORIGINAL utility units.

    alternating=True: Gauss-Seidel updates (player 0's infosets updated
    from a fresh pass, then player 1 against the new sigma_0). Empirically
    more stable, like CFR+'s alternation.

    anchor_mode: 'fixed' | 'periodic' (every K iters) | 'ema'
                 | 'adaptive' (move anchor when mean per-infoset L1 change
                   of sigma < tol; then tol *= tol_decay). Implements the
                   eps_k inner-accuracy condition of Thm 2. NOTE: Phase 0b
                   found 'periodic' both simpler and faster; it is the
                   recommended default (linear last-iterate empirically).

    Returns (sigma, log, snapshots).
    """
    sigma = sigma0 if sigma0 is not None else uniform_strategy(game)
    sigma = {I: np.array(s, dtype=float) for I, s in sigma.items()}
    anchor = {I: s.copy() for I, s in sigma.items()}
    a_ = 1.0 / (1.0 + eta * tau)          # weight on log sigma^t
    b_ = (eta * tau) / (1.0 + eta * tau)  # weight on log mu
    c_ = eta / (1.0 + eta * tau)          # weight on q
    n_iset = len(sigma)
    u_div = game.max_abs_u if scale_utilities else 1.0
    rng = np.random.default_rng(seed)
    normalizer = {}                       # frozen per anchor phase
    tol, phase_len = tol0, 0
    cur_eta, cur_K, moves_done = eta, K, 0
    log, snapshots = [], []

    def updated(I, qa):
        if normalize:
            qv = qa / (normalizer[I] * u_div)
        else:
            qv = qa / u_div
        if noise_std > 0:
            qv = qv + noise_std * rng.standard_normal(len(qv))
        s = np.clip(sigma[I], eps, None)
        m = np.clip(anchor[I], eps, None)
        logits = a_ * np.log(s) + b_ * np.log(m) + c_ * qv
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    for t in range(1, iters + 1):
        a_ = 1.0 / (1.0 + cur_eta * tau)
        b_ = (cur_eta * tau) / (1.0 + cur_eta * tau)
        c_ = cur_eta / (1.0 + cur_eta * tau)
        delta = 0.0
        if alternating:
            for pl in (0, 1):
                qp, _, rop, _ = cf_values(game, sigma, pl)
                if normalize:
                    for I, w in rop.items():
                        if I not in normalizer:
                            normalizer[I] = max(w, 1e-12)
                for I, qa in qp.items():
                    ns = updated(I, qa)
                    delta += float(np.abs(ns - sigma[I]).sum())
                    sigma[I] = ns
        else:
            q0, _, ro0, _ = cf_values(game, sigma, 0)
            q1, _, ro1, _ = cf_values(game, sigma, 1)
            if normalize:
                for ro in (ro0, ro1):
                    for I, w in ro.items():
                        if I not in normalizer:
                            normalizer[I] = max(w, 1e-12)
            new_sigma = dict(sigma)
            for q in (q0, q1):
                for I, qa in q.items():
                    new_sigma[I] = updated(I, qa)
            delta = sum(float(np.abs(new_sigma[I] - sigma[I]).sum())
                        for I in sigma)
            sigma = new_sigma
        delta /= n_iset
        phase_len += 1
        # anchor update
        moved = False
        if anchor_mode == "periodic" and phase_len >= cur_K:
            moved = True
        elif anchor_mode == "ema":
            beta = ema_beta if ema_beta is not None else 1.0 / K
            for I in anchor:
                anchor[I] = (1 - beta) * anchor[I] + beta * sigma[I]
            if normalize and t % max(1, K) == 0:
                normalizer = {}  # refresh occasionally under ema
        elif anchor_mode == "adaptive" and (delta < tol
                                            or phase_len >= max_phase):
            moved = True
            tol = max(tol * tol_decay, tol_min)
        if moved:
            anchor = {I: s.copy() for I, s in sigma.items()}
            phase_len = 0
            moves_done += 1
            if superphase and moves_done % superphase == 0:
                cur_eta *= eta_decay      # stochastic eps_k schedule:
                cur_K = int(cur_K * 2)    # noise floor halves, phase doubles
            if normalize:
                normalizer = {}  # recompute next iter under new anchor
        if eval_fn is not None and (t % eval_every == 0 or t == 1):
            m = eval_fn(sigma, None)
            m["iter"] = t
            log.append(m)
        if snapshot_every and t % snapshot_every == 0:
            snapshots.append((t, {I: s.copy() for I, s in sigma.items()}))
    return sigma, log, snapshots


def strategy_l1_distance(s1, s2):
    return sum(float(np.abs(s1[I] - s2[I]).sum()) for I in s1)
