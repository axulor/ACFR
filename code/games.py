# -*- coding: utf-8 -*-
"""Generic extensive-form game tree engine + Kuhn / Leduc poker.

Tree is built explicitly once. Node types: Terminal / Chance / Decision.
Utilities are for player 0 (zero-sum: u1 = -u0).
Infoset keys are strings; all histories in an infoset share the same depth
and action set (perfect recall).
"""
import sys
sys.setrecursionlimit(100000)


class Terminal:
    __slots__ = ("u",)
    def __init__(self, u):
        self.u = float(u)  # utility for player 0


class Chance:
    __slots__ = ("children",)
    def __init__(self, children):
        self.children = children  # list of (prob, node)


class Decision:
    __slots__ = ("player", "infoset", "actions", "children", "meta")
    def __init__(self, player, infoset, actions, children, meta=None):
        self.player = player          # 0 or 1
        self.infoset = infoset        # string key
        self.actions = actions        # list of action labels
        self.children = children      # list of nodes, aligned with actions
        self.meta = meta              # dict of state info for feature
                                      # encoders (P4); None for legacy use


class Game:
    def __init__(self, root, name):
        self.root = root
        self.name = name
        # registry: infoset -> (player, n_actions)
        self.infosets = {}
        self._collect(root)
        self.max_abs_u = self._max_u(root)

    def _max_u(self, node):
        if isinstance(node, Terminal):
            return abs(node.u)
        if isinstance(node, Chance):
            return max(self._max_u(c) for _, c in node.children)
        return max(self._max_u(c) for c in node.children)

    def _collect(self, node):
        if isinstance(node, Terminal):
            return
        if isinstance(node, Chance):
            for _, c in node.children:
                self._collect(c)
            return
        if node.infoset in self.infosets:
            p, na = self.infosets[node.infoset]
            assert p == node.player and na == len(node.actions), \
                f"inconsistent infoset {node.infoset}"
        else:
            self.infosets[node.infoset] = (node.player, len(node.actions))
        for c in node.children:
            self._collect(c)


# ----------------------------------------------------------------------
# Kuhn poker. Cards 0<1<2, one each, ante 1, bet size 1.
# Actions: 'p' (pass/check/fold) and 'b' (bet/call).
# Terminal histories: pp, bb, bp, pbp, pbb.
# ----------------------------------------------------------------------

def _kuhn_node(c0, c1, h):
    if h in ("pp", "bb", "bp", "pbp", "pbb"):
        if h == "bp":     # p0 bet, p1 folded
            return Terminal(+1)
        if h == "pbp":    # p1 bet, p0 folded
            return Terminal(-1)
        amount = 1 if h == "pp" else 2
        return Terminal(amount if c0 > c1 else -amount)
    player = len(h) % 2  # "":p0, "p"/"b":p1, "pb":p0
    card = c0 if player == 0 else c1
    infoset = f"{card}:{h}"
    children = [_kuhn_node(c0, c1, h + a) for a in ("p", "b")]
    meta = {"game": "kuhn", "c0": c0, "c1": c1, "hist": h,
            "to_act": player}
    return Decision(player, infoset, ["p", "b"], children, meta)


def build_kuhn():
    deals = [(a, b) for a in range(3) for b in range(3) if a != b]
    children = [(1.0 / 6.0, _kuhn_node(c0, c1, "")) for c0, c1 in deals]
    return Game(Chance(children), "kuhn")


# ----------------------------------------------------------------------
# Leduc poker. Deck = {J,Q,K} x 2 suits (ranks 0,1,2). Each player one
# private card; one board card after round 1. Ante 1. Bet sizes: 2 (round 1),
# 4 (round 2), max 2 raises per round. Player 0 acts first in both rounds.
# Actions: 'k' check, 'c' call, 'r' raise, 'f' fold.
# Showdown: pair (private==board) beats non-pair; otherwise higher rank;
# equal ranks tie.
# ----------------------------------------------------------------------

_DECK = [0, 0, 1, 1, 2, 2]
_RAISE = {1: 2, 2: 4}


def _leduc_showdown(r0, r1, board, contrib):
    p0_pair, p1_pair = (r0 == board), (r1 == board)
    if p0_pair and not p1_pair:
        w = 0
    elif p1_pair and not p0_pair:
        w = 1
    elif r0 > r1:
        w = 0
    elif r1 > r0:
        w = 1
    else:
        return Terminal(0.0)
    # contribs are equal at showdown; winner wins loser's contribution
    return Terminal(contrib[1] if w == 0 else -contrib[0])


def _leduc_round_end(rnd, r0, r1, board, contrib, hist, deal_left):
    if rnd == 1:
        # deal board card: each remaining deck index equally likely
        ch = []
        for bi in deal_left:
            rest = list(deal_left)
            rest.remove(bi)
            ch.append((1.0 / len(deal_left),
                       _leduc_bet(2, r0, r1, _DECK[bi], list(contrib), 0,
                                  hist + "|", None, 0, rest)))
        return Chance(ch)
    return _leduc_showdown(r0, r1, board, contrib)


def _leduc_bet(rnd, r0, r1, board, contrib, raises, hist, last, to_act,
               deal_left):
    """One betting state. `last` = previous action in this round (None at
    round start). Facing a bet iff contribs are unequal."""
    facing = contrib[0] != contrib[1]
    if facing:
        actions = ["f", "c"] + (["r"] if raises < 2 else [])
    else:
        actions = ["k"] + (["r"] if raises < 2 else [])
    own = r0 if to_act == 0 else r1
    b = board if board is not None else -1
    infoset = f"{own}:{b}:{hist}"
    meta = {"game": "leduc", "r0": r0, "r1": r1, "board": b, "rnd": rnd,
            "raises": raises, "facing": int(facing),
            "contrib": tuple(contrib), "hist": hist, "to_act": to_act}
    children = []
    for a in actions:
        nc = list(contrib)
        if a == "f":
            # folder loses own contribution
            u = contrib[1] if to_act == 1 else -contrib[0]
            children.append(Terminal(u))
            continue
        if a == "c":
            nc[to_act] = nc[1 - to_act]
            children.append(_leduc_round_end(rnd, r0, r1, board, nc,
                                             hist + a, deal_left))
            continue
        if a == "k":
            if last == "k":
                children.append(_leduc_round_end(rnd, r0, r1, board, nc,
                                                 hist + a, deal_left))
            else:
                children.append(_leduc_bet(rnd, r0, r1, board, nc, raises,
                                           hist + a, "k", 1 - to_act,
                                           deal_left))
            continue
        if a == "r":
            diff = nc[1 - to_act] - nc[to_act]
            nc[to_act] += diff + _RAISE[rnd]
            children.append(_leduc_bet(rnd, r0, r1, board, nc, raises + 1,
                                       hist + a, "r", 1 - to_act, deal_left))
            continue
    return Decision(to_act, infoset, actions, children, meta)


def build_leduc():
    n = len(_DECK)
    children = []
    deals = [(i, j) for i in range(n) for j in range(n) if i != j]
    for i, j in deals:
        rest = [k for k in range(n) if k != i and k != j]
        node = _leduc_bet(1, _DECK[i], _DECK[j], None, [1, 1], 0, "", None,
                          0, rest)
        children.append((1.0 / len(deals), node))
    return Game(Chance(children), "leduc")


# ----------------------------------------------------------------------
# Liar's Dice, 2 players x 1 die with `sides` faces (OpenSpiel
# liars_dice topology; no wild ones). Bids are (quantity, face) with
# quantity in {1,2}, ordered by q*sides+face; each move must strictly
# raise the bid or call "liar" (only after a first bid). On a call:
# count dice showing the bid face; bidder wins +-1.
# Bid index b in [0, 2*sides): quantity = 1 + b//sides, face = b%sides.
# ----------------------------------------------------------------------

def _ld_node(d0, d1, last, to_act, sides, bid_seq):
    n_bids = 2 * sides
    actions, children = [], []
    for b in range(last + 1 if last is not None else 0, n_bids):
        actions.append(f"b{b}")
        children.append(_ld_node(d0, d1, b, 1 - to_act, sides,
                                 bid_seq + (b,)))
    if last is not None:
        q, f = 1 + last // sides, last % sides
        cnt = int(d0 == f) + int(d1 == f)
        bidder = 1 - to_act
        bidder_wins = cnt >= q
        u = 1.0 if (bidder == 0) == bidder_wins else -1.0
        actions.append("liar")
        children.append(Terminal(u))
    own = d0 if to_act == 0 else d1
    infoset = f"{own}:{','.join(map(str, bid_seq))}"
    meta = {"game": "liars", "d0": d0, "d1": d1, "last": last,
            "to_act": to_act, "sides": sides, "bid_seq": bid_seq}
    return Decision(to_act, infoset, actions, children, meta)


def build_liars_dice(sides=4):
    deals = [(a, b) for a in range(sides) for b in range(sides)]
    children = [(1.0 / len(deals), _ld_node(d0, d1, None, 0, sides, ()))
                for d0, d1 in deals]
    return Game(Chance(children), f"liars{sides}")
