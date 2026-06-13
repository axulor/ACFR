# -*- coding: utf-8 -*-
"""P4: real feature encodings (replacing the bridge one-hot).

The bridge version (P3) proved the loop is sound when the nets are a
smooth reparameterization of the table. P4's question: what is the
structure of the distillation/Q error under a GENERALIZING encoding --
the kind an at-scale agent would use (ranks, board, pot state, betting
line), where parameters are shared across infosets.

Encoders consume Decision.meta (games.py) and return per-infoset /
per-history feature matrices aligned with the agent's own index maps,
so they slot directly into NeuralACFR in place of identity matrices.

Kuhn  : card one-hot(3) + betting line (2 pos x {none,p,b}) [+ both
        cards & to_act for histories]
Leduc : own rank(3) + board(4: none+3) + pair flag + round + raises(3)
        + facing flag + contribs/13 + betting line (8 pos x {k,c,r,f,|})
        [+ both ranks & both pair flags & to_act for histories]
"""
import numpy as np


# ---------------------------------------------------------------- kuhn

_K_ACT = {"p": 0, "b": 1}


def _kuhn_line(h, n_pos=2):
    v = np.zeros(n_pos * 3)
    for i in range(n_pos):
        if i < len(h):
            v[i * 3 + 1 + _K_ACT[h[i]]] = 1.0
        else:
            v[i * 3 + 0] = 1.0          # "none" slot
    return v


def _kuhn_iset(meta):
    own = meta["c0"] if meta["to_act"] == 0 else meta["c1"]
    card = np.zeros(3); card[own] = 1.0
    return np.concatenate([card, _kuhn_line(meta["hist"]),
                           [float(meta["to_act"])]])


def _kuhn_hist(meta):
    c0 = np.zeros(3); c0[meta["c0"]] = 1.0
    c1 = np.zeros(3); c1[meta["c1"]] = 1.0
    return np.concatenate([c0, c1, _kuhn_line(meta["hist"]),
                           [float(meta["to_act"])]])


# --------------------------------------------------------------- leduc

_L_ACT = {"k": 0, "c": 1, "r": 2, "f": 3, "|": 4}
_L_POS = 8
_L_POT = 13.0                              # max contribution


def _leduc_line(h):
    v = np.zeros(_L_POS * 6)
    for i in range(_L_POS):
        if i < len(h):
            v[i * 6 + 1 + _L_ACT[h[i]]] = 1.0
        else:
            v[i * 6 + 0] = 1.0
    return v


def _leduc_common(meta):
    board = np.zeros(4)
    board[0 if meta["board"] < 0 else 1 + meta["board"]] = 1.0
    raises = np.zeros(3); raises[meta["raises"]] = 1.0
    c0, c1 = meta["contrib"]
    return np.concatenate([
        board, raises,
        [float(meta["rnd"] == 2), float(meta["facing"]),
         c0 / _L_POT, c1 / _L_POT],
        _leduc_line(meta["hist"])])


def _leduc_iset(meta):
    own = meta["r0"] if meta["to_act"] == 0 else meta["r1"]
    rank = np.zeros(3); rank[own] = 1.0
    pair = float(meta["board"] == own)
    return np.concatenate([rank, [pair, float(meta["to_act"])],
                           _leduc_common(meta)])


def _leduc_hist(meta):
    r0 = np.zeros(3); r0[meta["r0"]] = 1.0
    r1 = np.zeros(3); r1[meta["r1"]] = 1.0
    p0 = float(meta["board"] == meta["r0"])
    p1 = float(meta["board"] == meta["r1"])
    return np.concatenate([r0, r1, [p0, p1, float(meta["to_act"])],
                           _leduc_common(meta)])


# --------------------------------------------------------- liars dice

def _liars_common(meta):
    n_bids = 2 * meta["sides"]
    made = np.zeros(n_bids)
    for b in meta["bid_seq"]:
        made[b] = 1.0
    last = np.zeros(n_bids + 1)
    last[0 if meta["last"] is None else 1 + meta["last"]] = 1.0
    return np.concatenate([made, last, [float(meta["to_act"])]])


def _liars_iset(meta):
    own = meta["d0"] if meta["to_act"] == 0 else meta["d1"]
    die = np.zeros(meta["sides"]); die[own] = 1.0
    return np.concatenate([die, _liars_common(meta)])


def _liars_hist(meta):
    d0 = np.zeros(meta["sides"]); d0[meta["d0"]] = 1.0
    d1 = np.zeros(meta["sides"]); d1[meta["d1"]] = 1.0
    return np.concatenate([d0, d1, _liars_common(meta)])


# ------------------------------------------------------------ assembly

_ENC = {"kuhn": (_kuhn_iset, _kuhn_hist),
        "leduc": (_leduc_iset, _leduc_hist),
        "liars": (_liars_iset, _liars_hist)}


def feature_matrices(iset_ids, node_list):
    """Build (I_feat, H_feat) numpy matrices aligned with the agent's
    iset_ids ({infoset -> idx}) and node_list (decision nodes in agent
    order). One representative meta per infoset (any node in the infoset
    has identical observable meta fields by perfect recall)."""
    metas = {}
    for n in node_list:
        if n.meta is None:
            raise ValueError("game tree has no meta; rebuild games.py")
        metas.setdefault(n.infoset, n.meta)
    game_name = node_list[0].meta["game"]
    enc_i, enc_h = _ENC[game_name]
    I_rows = [None] * len(iset_ids)
    for I, idx in iset_ids.items():
        I_rows[idx] = enc_i(metas[I])
    H_rows = [enc_h(n.meta) for n in node_list]
    return (np.asarray(I_rows, dtype=np.float32),
            np.asarray(H_rows, dtype=np.float32))
