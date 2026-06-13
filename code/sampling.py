# -*- coding: utf-8 -*-
"""P1: model-free sampled A-CFR with the lambda-estimator, plus an
OS-MCCFR baseline.

lambda-estimator (per sampled trajectory, traverser p, sampled action â):

    qbar(h,a) = Qp(h,a) + 1{a=â} * (lam/xi(â|I)) * ( vbar(h') - Qp(h,â) )
    vbar(h)   = sum_a sigma(a|I) qbar(h,a);   vbar(terminal) = u_p(z)

  lam=1  -> VR-MCCFR/DREAM baseline-corrected estimator (unbiased)
  lam=0  -> ESCHER-style pure value estimate (zero IS variance, biased
            by Q error)
  'adaptive' -> lam(h) = clip(kappa * |td_err_ema(h)| / u_max, lo, 1)

Q is a tabular history-action value table (player-0 utility perspective),
trained online by sampled expected SARSA — the tabular analog of the
history value network of DREAM / AAAI'26.
"""
import numpy as np
from games import Terminal, Chance, Decision
from algorithms import uniform_strategy, _regret_matching, normalize_avg


class QTable:
    """Tabular Q(h,a) for player-0 utility; TD via sampled expected SARSA."""
    def __init__(self, alpha=0.2, td_init=1.0):
        self.q = {}        # id(node) -> np.array
        self.td_ema = {}   # id(node) -> float, EMA of |TD error|
        self.alpha = alpha
        self.td_init = td_init

    def get(self, node):
        nid = id(node)
        if nid not in self.q:
            self.q[nid] = np.zeros(len(node.actions))
            self.td_ema[nid] = self.td_init  # distrust initially (~u_max)
        return self.q[nid]

    def update(self, node, a_idx, target):
        q = self.get(node)
        td = target - q[a_idx]
        q[a_idx] += self.alpha * td

    def note_correction(self, node, corr):
        """Track the magnitude of the actual estimator correction term
        |vbar(h') - Q(h,a)|. This measures whether Q is WRONG (bias),
        not merely unstable -- the right signal for adaptive lambda.
        (TD self-consistency error was the previous, flawed signal: Q can
        be stably wrong while tracking a moving target.)"""
        nid = id(node)
        self.td_ema[nid] = 0.95 * self.td_ema.get(nid, self.td_init) \
            + 0.05 * abs(corr)


def _next_decision_value(node, sigma, Q):
    """Bootstrapped player-0 value of a node: terminal utility, or the
    expected-SARSA value at the next decision node (chance folded by
    one-sample through the actual trajectory is avoided: chance nodes
    are enumerated exactly here, they are cheap)."""
    if isinstance(node, Terminal):
        return node.u
    if isinstance(node, Chance):
        return sum(p * _next_decision_value(c, sigma, Q)
                   for p, c in node.children)
    q = Q.get(node)
    s = sigma[node.infoset]
    return float(np.dot(s, q))


def sacfr_episode(game, sigma, traverser, xi_eps, lam, Q, rng,
                  visited_out):
    """Sample one trajectory; fill visited_out with (infoset, advantage
    estimate vector) for the traverser; update Q along the way.
    Returns vbar at root (traverser perspective)."""
    sign = 1.0 if traverser == 0 else -1.0

    def rec(node):
        if isinstance(node, Terminal):
            return sign * node.u
        if isinstance(node, Chance):
            probs = np.array([p for p, _ in node.children])
            k = rng.choice(len(probs), p=probs / probs.sum())
            return rec(node.children[k][1])
        s = sigma[node.infoset]
        if node.player == traverser:
            xi = (1 - xi_eps) * s + xi_eps / len(s)
            a = rng.choice(len(s), p=xi / xi.sum())
            child = node.children[a]
            vbar_child = rec(child)
            qp = sign * Q.get(node)              # traverser perspective
            l_h = lam(node) if callable(lam) else lam
            qbar = qp.copy()
            qbar[a] += (l_h / xi[a]) * (vbar_child - qp[a])
            Q.note_correction(node, vbar_child - qp[a])
            vbar = float(np.dot(s, qbar))
            visited_out.append((node.infoset, qbar - vbar))
            # Q update: bootstrapped expected-SARSA target, player-0 units
            t0 = _next_decision_value(child, sigma, Q)
            Q.update(node, a, t0)
            return vbar
        else:
            a = rng.choice(len(s), p=s / s.sum())
            child = node.children[a]
            v = rec(child)
            t0 = _next_decision_value(child, sigma, Q)
            Q.update(node, a, t0)
            return v

    return rec(game.root)


def run_sacfr(game, episodes, eta=0.5, tau=0.1, K_ep=2000,
              lam=1.0, kappa=4.0, lam_lo=0.0,
              xi_eps=0.4, q_alpha=0.2, batch_size=16,
              superphase=4, eta_decay=0.5,
              eval_every=2000, eval_fn=None, seed=0, eps=1e-15):
    """Sampled anchored CFR (model-free, trajectory feedback).

    Advantage estimates are accumulated over `batch_size` episodes and
    AVERAGED per infoset before one anchored-MD update (variance / visits;
    the tabular analog of the deep version's per-iteration batch). Anchor
    moves every K_ep episodes; every `superphase` anchor moves the
    stepsize decays (eta *= eta_decay) and the phase doubles (K_ep *= 2)
    -- the stochastic eps_k schedule (noise floor halves per superphase).

    lam: float in [0,1], or 'adaptive' (per-history from Q's TD-error EMA).
    """
    rng = np.random.default_rng(seed)
    sigma = uniform_strategy(game)
    anchor = {I: s.copy() for I, s in sigma.items()}
    u_max = game.max_abs_u
    Q = QTable(alpha=q_alpha, td_init=u_max)

    if lam == "adaptive":
        def lam_fn(node):
            t = Q.td_ema[id(node)] if id(node) in Q.td_ema else 1.0
            return float(min(1.0, max(lam_lo, kappa * t / u_max)))
        lam_use = lam_fn
    else:
        lam_use = float(lam)

    log = []
    ep_in_phase, phases_done = 0, 0
    cur_eta, cur_K = eta, K_ep
    batch_sum, batch_cnt, in_batch = {}, {}, 0
    for ep in range(1, episodes + 1):
        traverser = ep % 2
        visited = []
        sacfr_episode(game, sigma, traverser, xi_eps, lam_use, Q, rng,
                      visited)
        for I, adv in visited:
            if I in batch_sum:
                batch_sum[I] += adv
                batch_cnt[I] += 1
            else:
                batch_sum[I] = adv.copy()
                batch_cnt[I] = 1
        in_batch += 1
        if in_batch >= batch_size:
            a_ = 1.0 / (1.0 + cur_eta * tau)
            b_ = (cur_eta * tau) / (1.0 + cur_eta * tau)
            c_ = cur_eta / (1.0 + cur_eta * tau)
            for I, ssum in batch_sum.items():
                qv = (ssum / batch_cnt[I]) / u_max
                s = np.clip(sigma[I], eps, None)
                m = np.clip(anchor[I], eps, None)
                logits = a_ * np.log(s) + b_ * np.log(m) + c_ * qv
                logits -= logits.max()
                p = np.exp(logits)
                sigma[I] = p / p.sum()
            batch_sum, batch_cnt, in_batch = {}, {}, 0
        ep_in_phase += 1
        if ep_in_phase >= cur_K:
            anchor = {I: s.copy() for I, s in sigma.items()}
            ep_in_phase = 0
            phases_done += 1
            if superphase and phases_done % superphase == 0:
                cur_eta *= eta_decay
                cur_K = int(cur_K * 2)
        if eval_fn is not None and (ep % eval_every == 0 or ep == 1):
            m = eval_fn(sigma, None)
            m["episode"] = ep
            log.append(m)
    return sigma, log


# ------------------------------------------------------- OS-MCCFR baseline

def run_os_mccfr(game, episodes, xi_eps=0.6,
                 eval_every=2000, eval_fn=None, seed=0):
    """Standard outcome-sampling MCCFR (regret matching, uniform avg,
    epsilon-greedy traverser sampling, IS-corrected estimates)."""
    rng = np.random.default_rng(seed)
    sigma = uniform_strategy(game)
    R = {I: np.zeros(na) for I, (p, na) in game.infosets.items()}
    avg = {I: np.zeros(na) for I, (p, na) in game.infosets.items()}

    def episode(traverser):
        sign = 1.0 if traverser == 0 else -1.0

        def rec(node, pi_xi):
            """pi_xi: traverser's own sampling prob root->node (opponent &
            chance are on-policy, so their reach ratios cancel).
            Returns (u_traverser(z), ratio) where ratio is the product of
            sigma(a)/xi(a) over traverser nodes strictly BELOW node."""
            if isinstance(node, Terminal):
                return sign * node.u, 1.0
            if isinstance(node, Chance):
                probs = np.array([p for p, _ in node.children])
                k = rng.choice(len(probs), p=probs / probs.sum())
                return rec(node.children[k][1], pi_xi)
            s = sigma[node.infoset]
            if node.player == traverser:
                xi = (1 - xi_eps) * s + xi_eps / len(s)
                xi = xi / xi.sum()
                a = rng.choice(len(s), p=xi)
                u_z, ratio = rec(node.children[a], pi_xi * xi[a])
                # canonical OS estimate: vhat(I,a_sampled) =
                #   u_z * sigma_tail / (xi_anc * xi(a) * xi_tail)
                qhat = np.zeros(len(s))
                qhat[a] = u_z * ratio / (pi_xi * xi[a])
                vhat = qhat[a] * s[a]
                R[node.infoset] += qhat - vhat
                sigma[node.infoset] = _regret_matching(R[node.infoset])
                return u_z, ratio * s[a] / xi[a]
            else:
                a = rng.choice(len(s), p=s / s.sum())
                avg[node.infoset] += s
                u_z, ratio = rec(node.children[a], pi_xi)
                return u_z, ratio

        rec(game.root, 1.0)

    log = []
    for ep in range(1, episodes + 1):
        episode(ep % 2)
        if eval_fn is not None and (ep % eval_every == 0 or ep == 1):
            m = eval_fn(normalize_avg(avg), None)
            m2 = eval_fn(sigma, None)
            m = {"nc_avg": m["nc_last"], "nc_last": m2["nc_last"],
                 "episode": ep}
            log.append(m)
    return sigma, normalize_avg(avg), log
