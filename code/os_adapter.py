# -*- coding: utf-8 -*-
"""OpenSpiel -> A-CFR adapter (Ubuntu phase).

Builds our explicit EFG tree (games.py: Terminal/Chance/Decision/Game)
from any 2-player zero-sum sequential-imperfect-information OpenSpiel
game, so the tabular/sampling/neural algorithms run UNCHANGED with exact
NashConv. Mapping:
  is_terminal      -> Terminal(returns()[0])            (player-0 utility)
  is_chance_node   -> Chance([(prob, child), ...])      (chance_outcomes)
  decision node    -> Decision(current_player,
                               information_state_string(),
                               legal-action labels, children, meta)
The infoset key = OpenSpiel's information_state_string (perfect recall);
this is exactly the key our algorithms index on.

Full-tree construction: fine for Kuhn (12 isets) / Leduc (936) / small
liar's dice -- gives exact best-response NashConv for the baseline
comparison. Large games need a non-enumerative path (future work).

Usage:
    from os_adapter import build_openspiel
    g = build_openspiel("leduc_poker")        # -> games.Game
    from exploitability import nash_conv
    print(nash_conv(g, uniform_strategy(g)))
"""
import pyspiel
from games import Terminal, Chance, Decision, Game


def build_openspiel(game_name, params=None, max_nodes=5_000_000,
                    with_features=False):
    """Return a games.Game built by enumerating the OpenSpiel game tree.

    game_name: e.g. "kuhn_poker", "leduc_poker",
               "liars_dice(dice_sides=6)".
    params:    optional dict of game parameters (alternative to the
               string form).
    with_features: also store generalizing feature vectors in each
        Decision.meta: 'fi' = acting player's information_state_tensor
        (policy-net input -> generalizes across infosets, like the
        baselines), 'fh' = concat of BOTH players' info tensors (Q-net
        input -> distinguishes histories within an infoset).
    """
    game = (pyspiel.load_game(game_name, params) if params
            else pyspiel.load_game(game_name))
    gt = game.get_type()
    assert game.num_players() == 2, "adapter assumes 2 players"
    assert gt.utility == pyspiel.GameType.Utility.ZERO_SUM, \
        "adapter assumes zero-sum (u1 = -u0)"

    n_nodes = [0]

    def rec(state):
        n_nodes[0] += 1
        if n_nodes[0] > max_nodes:
            raise RuntimeError(f"tree exceeds {max_nodes} nodes; "
                               "game too large for full enumeration")
        if state.is_terminal():
            return Terminal(state.returns()[0])
        if state.is_chance_node():
            ch = [(p, rec(state.child(a)))
                  for a, p in state.chance_outcomes()]
            return Chance(ch)
        pl = state.current_player()
        legal = state.legal_actions()
        infoset = state.information_state_string()
        labels = [state.action_to_string(pl, a) for a in legal]
        meta = {"game": gt.short_name, "os_player": pl,
                "os_actions": list(legal)}
        if with_features:
            meta["fi"] = list(state.information_state_tensor(pl))
            meta["fh"] = (list(state.information_state_tensor(0))
                          + list(state.information_state_tensor(1)))
        children = [rec(state.child(a)) for a in legal]
        return Decision(pl, infoset, labels, children, meta)

    root = rec(game.new_initial_state())
    g = Game(root, gt.short_name)
    g.os_n_nodes = n_nodes[0]
    return g


def openspiel_feature_fn(g):
    """feature_fn(iset_ids, node_list) -> (I_np, H_np) for NeuralACFR, using
    the OpenSpiel info_state_tensor features stored by
    build_openspiel(with_features=True). I_np rows = per-infoset acting-player
    tensor (generalizing policy features); H_np rows = per-node both-player
    tensor (history features for the Q baseline)."""
    import numpy as np

    def fn(iset_ids, node_list):
        iset_fi = {}
        for n in node_list:
            if n.infoset not in iset_fi:
                iset_fi[n.infoset] = n.meta["fi"]
        d_i = len(next(iter(iset_fi.values())))
        d_h = len(node_list[0].meta["fh"])
        I_np = np.zeros((len(iset_ids), d_i), dtype=np.float32)
        for infoset, idx in iset_ids.items():
            I_np[idx] = iset_fi[infoset]
        H_np = np.zeros((len(node_list), d_h), dtype=np.float32)
        for j, n in enumerate(node_list):
            H_np[j] = n.meta["fh"]
        return I_np, H_np

    return fn


if __name__ == "__main__":
    # self-check against our native engines / known Nash values
    import time
    from exploitability import nash_conv, profile_value
    from algorithms import uniform_strategy, run_cfr

    for name, want_isets in [("kuhn_poker", 12), ("leduc_poker", 936)]:
        t0 = time.time()
        g = build_openspiel(name)
        n_iset = len(g.infosets)
        print(f"{name}: {n_iset} infosets, {g.os_n_nodes} nodes, "
              f"max|u|={g.max_abs_u} ({time.time()-t0:.2f}s)")
        # solve with CFR, check NashConv -> 0
        _, avg, _ = run_cfr(g, 200)
        nc = nash_conv(g, avg)[0]
        print(f"  CFR(200) avg NashConv = {nc:.4f}")
        if name == "kuhn_poker":
            v = profile_value(g, avg)
            print(f"  Kuhn game value (p0) = {v:.5f} (exact -1/18="
                  f"{-1/18:.5f})")
