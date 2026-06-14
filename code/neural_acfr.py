# -*- coding: utf-8 -*-
"""P3: neural A-CFR (bridge version).

Bridge design: one-hot infoset/history encodings, so the networks are a
smooth parameterization of the tabular algorithm. This isolates exactly
what is NEW in the neural version:
  (1) distillation error delta -> floor ~ delta/(1-rho), no accumulation
      over time (Prop 4);
  (2) stability of bootstrapped (never-reset) training across anchor
      phases.
At-scale versions replace one-hot with feature encodings and the anchor
table with an EMA/snapshot anchor NETWORK; the loop is unchanged.

Networks: policy MLP pi_theta (distilled to closed-form anchored-MD
targets at visited infosets); Q MLP over histories (sampled expected
SARSA). Sigma/Q are cached as tables via batched forward passes once per
update cycle -- exact NashConv evaluation stays available.
"""
import numpy as np
import torch
import torch.nn as nn

from games import Terminal, Chance, Decision


def _collect_nodes(game):
    """Enumerate decision nodes and infosets; build indexes."""
    iset_ids = {}                      # infoset -> int
    node_list = []                     # all decision nodes
    def rec(node):
        if isinstance(node, Terminal):
            return
        if isinstance(node, Chance):
            for _, c in node.children:
                rec(c)
            return
        if node.infoset not in iset_ids:
            iset_ids[node.infoset] = len(iset_ids)
        node_list.append(node)
        for c in node.children:
            rec(c)
    rec(game.root)
    node_ids = {id(n): i for i, n in enumerate(node_list)}
    return iset_ids, node_list, node_ids


class MLP(nn.Module):
    def __init__(self, d_in, d_out, width=64, depth=3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(nn.Linear(d, d_out))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class NeuralACFR:
    def __init__(self, game, eta=0.5, tau=0.1, K_ep=2000, batch_size=32,
                 lam=1.0, xi_eps=0.4, superphase=4, eta_decay=0.5,
                 eta_min=0.0, grow_distill=False, max_distill=320,
                 distill_steps=20, q_steps=40, lr=1e-2, q_lr=1e-2,
                 width=64, seed=0, device="cpu",
                 replay_targets=False, diag=False,
                 q_anchor=False, q_replay=0, q_batch=512,
                 encoding="onehot", lr_couple=False, anchor_ema=1.0,
                 feature_fn=None):
        """eta_min / grow_distill: Prop-4 schedule corollary. The neural
        floor is max(noise ~ eta*sigma^2/(B*tau), approx ~ delta/(eta*tau)).
        Decaying eta with FIXED distillation accuracy delta makes the
        approx floor double per superphase (the N1-smoke pathology).
        Remedies: grow_distill=True doubles distill_steps whenever eta
        halves (delta/eta kept ~constant); eta_min freezes the decay."""
        self.g = game
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.iset_ids, self.node_list, self.node_ids = _collect_nodes(game)
        self.n_iset = len(self.iset_ids)
        self.n_hist = len(self.node_list)
        self.amax = max(len(n.actions) for n in self.node_list)
        self.u_max = game.max_abs_u
        self.dev = torch.device(device)
        # encoding: "onehot" = bridge (smooth reparameterization of the
        # table); "features" = generalizing encoding (P4) from games.py
        # meta via features.py -- parameters shared across infosets.
        if encoding == "features":
            if feature_fn is not None:      # e.g. OpenSpiel info_state_tensor
                I_np, H_np = feature_fn(self.iset_ids, self.node_list)
            else:
                from features import feature_matrices
                I_np, H_np = feature_matrices(self.iset_ids, self.node_list)
            self.I_enc = torch.tensor(I_np, device=self.dev).float()
            self.H_enc = torch.tensor(H_np, device=self.dev).float()
        else:
            self.I_enc = torch.eye(self.n_iset, device=self.dev)
            self.H_enc = torch.eye(self.n_hist, device=self.dev)
        self.pi = MLP(self.I_enc.shape[1], self.amax, width).to(self.dev)
        self.qn = MLP(self.H_enc.shape[1], self.amax, width).to(self.dev)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(self.qn.parameters(), q_lr)
        # per-infoset action masks / counts
        self.n_act = np.zeros(self.n_iset, dtype=int)
        for n in self.node_list:
            self.n_act[self.iset_ids[n.infoset]] = len(n.actions)
        self.eta, self.tau, self.K_ep = eta, tau, K_ep
        self.B, self.lam, self.xi_eps = batch_size, lam, xi_eps
        self.superphase, self.eta_decay = superphase, eta_decay
        self.eta_min, self.grow_distill = eta_min, grow_distill
        self.max_distill = max_distill
        self.distill_steps, self.q_steps = distill_steps, q_steps
        # replay_targets: keep the latest anchored-MD target for every
        # infoset seen so far and distill over ALL of them each cycle
        # (kills cross-batch forgetting; at scale: a bounded target
        # replay buffer -- still NO average-strategy network, targets
        # are current-policy targets).
        self.replay_targets = replay_targets
        self.target_table = {}
        # diag: per-update distillation diagnostics (P3c). Measures the
        # actual delta of Prop 4 instead of assuming delta ~ 1/steps.
        self.diag = diag
        self.diag_log = []
        # P3d Q stabilizers. C0 diagnosis: the slow drift is Q-side --
        # q_rmse grows over time and inflates estimator variance
        # ~ (V-Q)^2/xi, raising the noise floor. q_anchor: bootstrap
        # targets from the Q snapshot taken at each anchor move (target
        # network mapped onto the anchor structure). q_replay>0: keep a
        # buffer of (node_id, action, child_ref) transitions and train Q
        # each cycle on a fresh-bootstrapped random minibatch of q_batch.
        self.q_anchor = q_anchor
        self.q_replay, self.q_batch = q_replay, q_batch
        self.q_buffer = []
        # lr_couple (P3f): the surviving late-horizon floor is policy-
        # side per-update SGD noise (interference), which does NOT
        # shrink with eta -- random-walk variance ~ sigma_w^2/(eta*tau)
        # grows as eta anneals (the true Prop-4 delta is update VARIANCE,
        # not fit bias). Coupling lr to eta (lr *= eta_decay on each
        # eta decay) makes noise ~ eta^2 => floor ~ eta keeps falling.
        self.lr_couple = lr_couple
        # anchor_ema (2026-06-13): soft anchor. At each K-step the anchor
        # only moves a fraction `anchor_ema` toward the current policy
        # (1.0 = hard snapshot = original behavior). <1 gives the anchor
        # inertia -> a stronger restoring force that damps the init-
        # dependent last-iterate drift seen on torch 2.11 / Leduc. Still
        # a current-policy construct (NO average-strategy net).
        self.anchor_ema = anchor_ema
        # adaptive lambda (P4d): lambda* is interior and game-dependent
        # (Leduc 0.5, liars5 0.75 -- variance-compounding vs Q-bias
        # balance). Per-history lam(h) = clip(kappa*EMA|corr|/u_max,
        # lam_lo, 1), where corr = vbar(h')-Q(h,a) is the estimator's
        # actual correction (P1d signal: measures Q WRONGNESS, not
        # instability). EMA starts at u_max => lam=1 (unbiased) until Q
        # earns trust. Enable with lam='adaptive'.
        self.lam_adaptive = (lam == "adaptive")
        if self.lam_adaptive:
            self.lam = 1.0
        self.kappa, self.lam_lo = 4.0, 0.0
        self.corr_ema = np.full(self.n_hist, float(self.u_max))
        self.refresh_sigma()
        self.anchor = {I: s.copy() for I, s in self.sigma.items()}
        self.refresh_q()
        self.q_tab_anchor = self.q_tab.copy()

    # ---------------- cached tables from the nets ----------------
    def refresh_sigma(self):
        with torch.no_grad():
            logits = self.pi(self.I_enc).cpu().numpy()
        self.sigma = {}
        for I, i in self.iset_ids.items():
            na = self.n_act[i]
            l = logits[i, :na] - logits[i, :na].max()
            p = np.exp(l)
            self.sigma[I] = p / p.sum()

    def refresh_q(self):
        with torch.no_grad():
            self.q_tab = self.qn(self.H_enc).cpu().numpy()

    def _ndv(self, node, qtab):
        """Bootstrapped player-0 value of a node under the given Q table
        (chance enumerated exactly; next decision node via expected
        SARSA)."""
        if isinstance(node, Terminal):
            return node.u
        if isinstance(node, Chance):
            return sum(p * self._ndv(c, qtab) for p, c in node.children)
        qv = qtab[self.node_ids[id(node)], :len(node.actions)]
        return float(np.dot(self.sigma[node.infoset], qv))

    # ---------------- one sampled episode (lambda estimator) -----
    def episode(self, traverser, visited, q_trans):
        sign = 1.0 if traverser == 0 else -1.0
        rng = self.rng

        def rec(node):
            if isinstance(node, Terminal):
                return sign * node.u
            if isinstance(node, Chance):
                probs = np.array([p for p, _ in node.children])
                k = rng.choice(len(probs), p=probs / probs.sum())
                return rec(node.children[k][1])
            s = self.sigma[node.infoset]
            nid = self.node_ids[id(node)]
            if node.player == traverser:
                xi = (1 - self.xi_eps) * s + self.xi_eps / len(s)
                xi = xi / xi.sum()
                a = rng.choice(len(s), p=xi)
                vb_child = rec(node.children[a])
                qp = sign * self.q_tab[nid, :len(s)]
                qbar = qp.copy()
                corr = vb_child - qp[a]
                if self.lam_adaptive:
                    self.corr_ema[nid] = (0.95 * self.corr_ema[nid]
                                          + 0.05 * abs(corr))
                    l_h = min(1.0, max(self.lam_lo,
                                       self.kappa * self.corr_ema[nid]
                                       / self.u_max))
                else:
                    l_h = self.lam
                qbar[a] += (l_h / xi[a]) * corr
                vbar = float(np.dot(s, qbar))
                visited.append((node.infoset, qbar - vbar))
                q_trans.append((nid, a, node.children[a]))
                return vbar
            else:
                a = rng.choice(len(s), p=s / s.sum())
                v = rec(node.children[a])
                q_trans.append((nid, a, node.children[a]))
                return v

        return rec(self.g.root)

    # ---------------- training steps -----------------------------
    def distill(self, targets):
        """targets: {infoset -> prob vector}; KL(target || pi_theta)."""
        ids = [self.iset_ids[I] for I in targets]
        X = self.I_enc[ids]
        T = torch.zeros(len(ids), self.amax, device=self.dev)
        M = torch.zeros(len(ids), self.amax, device=self.dev)
        for r, I in enumerate(targets):
            na = self.n_act[self.iset_ids[I]]
            T[r, :na] = torch.tensor(targets[I], device=self.dev)
            M[r, :na] = 1.0
        for _ in range(self.distill_steps):
            logits = self.pi(X)
            logp = torch.log_softmax(
                logits.masked_fill(M == 0, -1e9), dim=1)
            loss = -(T * logp).sum(dim=1).mean()
            self.opt_pi.zero_grad(); loss.backward(); self.opt_pi.step()

    def train_q(self, q_trans):
        if not q_trans:
            return
        if self.q_replay:
            self.q_buffer.extend(q_trans)
            if len(self.q_buffer) > self.q_replay:
                self.q_buffer = self.q_buffer[-self.q_replay:]
            k = min(self.q_batch, len(self.q_buffer))
            idx = self.rng.choice(len(self.q_buffer), size=k,
                                  replace=False)
            trans = [self.q_buffer[i] for i in idx]
        else:
            trans = q_trans
        # bootstrap targets computed at train time, from the anchored Q
        # snapshot if q_anchor else the current Q table (default path is
        # numerically identical to the old visit-time computation: q_tab
        # and sigma are static within a batch).
        qtab = self.q_tab_anchor if self.q_anchor else self.q_tab
        ids = torch.tensor([t[0] for t in trans], device=self.dev)
        acts = torch.tensor([t[1] for t in trans], device=self.dev)
        tg = torch.tensor([self._ndv(t[2], qtab) for t in trans],
                          dtype=torch.float32, device=self.dev)
        X = self.H_enc[ids]
        for _ in range(self.q_steps):
            pred = self.qn(X).gather(1, acts[:, None]).squeeze(1)
            loss = ((pred - tg) ** 2).mean()
            self.opt_q.zero_grad(); loss.backward(); self.opt_q.step()

    # ---------------- diagnostics (P3c) ---------------------------
    def q_rmse_exact(self):
        """RMSE of q_tab against the exact on-policy value Q^sigma
        (player-0 perspective) under the CURRENT cached sigma."""
        def val(node):
            if isinstance(node, Terminal):
                return node.u
            if isinstance(node, Chance):
                return sum(p * val(c) for p, c in node.children)
            s = self.sigma[node.infoset]
            return float(sum(s[i] * val(c)
                             for i, c in enumerate(node.children)))
        err, cnt = 0.0, 0
        for n_ in self.node_list:
            nid = self.node_ids[id(n_)]
            for i, c in enumerate(n_.children):
                err += (self.q_tab[nid, i] - val(c)) ** 2
                cnt += 1
        return (err / cnt) ** 0.5

    def _diag_record(self, ep, eta_now, targets, train_targets, sig_pre):
        kl = l1 = 0.0
        for I, t in targets.items():
            p = np.clip(self.sigma[I], 1e-12, None)
            t_ = np.clip(t, 1e-12, None)
            kl += float(np.sum(t_ * np.log(t_ / p)))
            l1 += float(np.abs(t - self.sigma[I]).sum())
        n = max(len(targets), 1)
        interf, m = 0.0, 0
        for I in self.sigma:
            if I not in train_targets:
                interf += float(np.abs(self.sigma[I] - sig_pre[I]).sum())
                m += 1
        self.diag_log.append({
            "episode": ep, "eta": eta_now,
            "distill_steps": self.distill_steps,
            "kl_fit": kl / n, "l1_fit": l1 / n,
            "interference_l1": interf / max(m, 1),
            "q_rmse": self.q_rmse_exact()})

    # ---------------- main loop ----------------------------------
    def run(self, episodes, eval_every=2000, eval_fn=None):
        log = []
        ep_in_phase, phases = 0, 0
        cur_eta, cur_K = self.eta, self.K_ep
        batch_adv, batch_cnt, in_batch, q_trans = {}, {}, 0, []
        for ep in range(1, episodes + 1):
            visited = []
            self.episode(ep % 2, visited, q_trans)
            for I, adv in visited:
                if I in batch_adv:
                    batch_adv[I] += adv; batch_cnt[I] += 1
                else:
                    batch_adv[I] = adv.copy(); batch_cnt[I] = 1
            in_batch += 1
            if in_batch >= self.B:
                a_ = 1.0 / (1.0 + cur_eta * self.tau)
                b_ = (cur_eta * self.tau) / (1.0 + cur_eta * self.tau)
                c_ = cur_eta / (1.0 + cur_eta * self.tau)
                targets = {}
                for I, ssum in batch_adv.items():
                    qv = (ssum / batch_cnt[I]) / self.u_max
                    s = np.clip(self.sigma[I], 1e-15, None)
                    m = np.clip(self.anchor[I], 1e-15, None)
                    lg = a_ * np.log(s) + b_ * np.log(m) + c_ * qv
                    lg -= lg.max()
                    p = np.exp(lg)
                    targets[I] = p / p.sum()
                if self.replay_targets:
                    self.target_table.update(
                        {I: t.copy() for I, t in targets.items()})
                    train_targets = self.target_table
                else:
                    train_targets = targets
                sig_pre = ({I: s.copy() for I, s in self.sigma.items()}
                           if self.diag else None)
                self.distill(train_targets)
                self.train_q(q_trans)
                self.refresh_sigma()
                self.refresh_q()
                if self.diag:
                    self._diag_record(ep, cur_eta, targets, train_targets,
                                      sig_pre)
                batch_adv, batch_cnt, in_batch, q_trans = {}, {}, 0, []
            ep_in_phase += 1
            if ep_in_phase >= cur_K:
                # anchor snapshot (at scale: EMA/copy of the policy NET).
                # anchor_ema<1 => soft snapshot (anchor lags policy, more
                # restoring inertia); ==1 => hard snapshot (original).
                b = self.anchor_ema
                if b >= 1.0:
                    self.anchor = {I: s.copy() for I, s in self.sigma.items()}
                    self.q_tab_anchor = self.q_tab.copy()
                else:
                    self.anchor = {I: (1 - b) * self.anchor[I] + b * s
                                   for I, s in self.sigma.items()}
                    self.q_tab_anchor = ((1 - b) * self.q_tab_anchor
                                         + b * self.q_tab)
                ep_in_phase = 0
                phases += 1
                if self.superphase and phases % self.superphase == 0:
                    new_eta = max(cur_eta * self.eta_decay, self.eta_min)
                    if new_eta < cur_eta:
                        if self.grow_distill:
                            self.distill_steps = min(self.max_distill,
                                                     self.distill_steps * 2)
                        if self.lr_couple:
                            for opt in (self.opt_pi, self.opt_q):
                                for grp in opt.param_groups:
                                    grp["lr"] *= self.eta_decay
                        # K* ~ 1/ln(1+eta*tau): K doubles only while eta
                        # actually halves; frozen eta => frozen K.
                        # (P3b's frozen arm used the old always-double
                        # semantics; changed 2026-06-12, see doc 17.)
                        cur_K = int(cur_K * 2)
                    cur_eta = new_eta
            if eval_fn is not None and (ep % eval_every == 0 or ep == 1):
                m = eval_fn(self.sigma, None)
                m["episode"] = ep
                log.append(m)
        return log
