"""
Tests for core/mcts.py

Run: pytest tests/test_mcts.py -v

Note: MCTS involves stochasticity (Dirichlet noise). Tests are written to be
robust to randomness — they check statistical properties rather than exact values.
The tiny model used here is randomly initialised; tests verify structure and
correctness, not strength.
"""
import numpy as np
import pytest
import torch
from collections import defaultdict

from core.game import Checkers, P1_MAN, P2_MAN, P1_KING, EMPTY
from core.encoder import StateEncoder
from core.model import AlphaNet
from core.mcts import MCTS, Node, _expand, _backpropagate


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def game():
    return Checkers()


@pytest.fixture
def enc(game):
    return StateEncoder(game)


@pytest.fixture
def net(enc, game):
    """Tiny untrained model — fast on CPU, sufficient for structural tests."""
    return AlphaNet(
        num_channels=enc.num_channels,
        action_size=game.action_size,
        num_resblocks=1,
        num_hidden=8,
    )


@pytest.fixture
def mcts(game, enc, net):
    return MCTS(game, enc, net, num_simulations=20, dirichlet_eps=0.0)


def blank_state(game):
    board = np.zeros((game.row_count, game.col_count), dtype=np.int8)
    return {
        "board": board,
        "repetitions": defaultdict(int),
        "no_progress": 0,
        "jump_again": None,
        "history": [board.copy()],
    }


def place(state, pieces):
    for (r, c), v in pieces.items():
        state['board'][r, c] = v
    state['history'][-1] = state['board'].copy()
    return state


# ── Node unit tests ───────────────────────────────────────────────────────────

class TestNode:
    def test_is_leaf_when_no_children(self, game):
        state = game.get_initial_state()
        node  = Node(state, 1)
        assert node.is_leaf

    def test_not_leaf_after_expansion(self, game, enc, net):
        state  = game.get_initial_state()
        node   = Node(state, 1)
        mcts_  = MCTS(game, enc, net, num_simulations=1, dirichlet_eps=0)
        net.eval()
        policy, _, valid = mcts_._network_eval(node)
        _expand(node, policy, valid, game)
        assert not node.is_leaf

    def test_q_value_zero_before_visits(self, game):
        node = Node(game.get_initial_state(), 1)
        assert node.q_value == 0.0

    def test_q_value_after_update(self, game):
        node = Node(game.get_initial_state(), 1)
        node.visit_count = 3
        node.value_sum   = 1.5
        assert node.q_value == pytest.approx(0.5)

    def test_ucb_prefers_high_prior_unvisited(self, game):
        """An unvisited child with high prior should beat a visited child."""
        state    = game.get_initial_state()
        visited  = Node(state, 1, prior=0.1)
        visited.visit_count = 10
        visited.value_sum   = 5.0  # Q = 0.5

        unvisited = Node(state, 1, prior=0.9)

        parent_n = 11
        assert unvisited.ucb(parent_n, c=1.5) > visited.ucb(parent_n, c=1.5)

    def test_best_child_returns_highest_ucb(self, game):
        parent = Node(game.get_initial_state(), 1)
        parent.visit_count = 10
        child_low  = Node(game.get_initial_state(), -1, prior=0.1)
        child_high = Node(game.get_initial_state(), -1, prior=0.9)
        parent.children = {0: child_low, 1: child_high}
        action, child = parent.best_child(c=1.5)
        assert action == 1
        assert child is child_high


# ── Backpropagation ───────────────────────────────────────────────────────────

class TestBackpropagate:
    def test_visit_counts_increment(self, game):
        nodes = [Node(game.get_initial_state(), p) for p in [1, -1, 1]]
        _backpropagate(nodes, value=1.0)
        assert all(n.visit_count == 1 for n in nodes)

    def test_value_negated_for_opponent(self, game):
        """
        Leaf player is 1 with value +1. Parent (-1) should receive -1.
        """
        leaf   = Node(game.get_initial_state(), player=1)
        parent = Node(game.get_initial_state(), player=-1)
        _backpropagate([parent, leaf], value=1.0)
        assert leaf.value_sum   == pytest.approx( 1.0)
        assert parent.value_sum == pytest.approx(-1.0)

    def test_value_same_for_same_player(self, game):
        """Multi-jump: two consecutive nodes with the same player."""
        node1 = Node(game.get_initial_state(), player=1)
        node2 = Node(game.get_initial_state(), player=1)
        _backpropagate([node1, node2], value=0.8)
        # Both have the same player as the leaf (node2) → both get +0.8
        assert node1.value_sum == pytest.approx(0.8)
        assert node2.value_sum == pytest.approx(0.8)


# ── Full search ───────────────────────────────────────────────────────────────

class TestSearch:
    def test_probs_sum_to_one(self, mcts, game):
        state = game.get_initial_state()
        probs = mcts.search(state, 1, temperature=1.0)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_probs_shape(self, mcts, game):
        state = game.get_initial_state()
        probs = mcts.search(state, 1)
        assert probs.shape == (game.action_size,)

    def test_only_legal_moves_have_mass(self, mcts, game):
        """No probability should appear on an illegal action."""
        state  = game.get_initial_state()
        valid  = game.get_valid_moves(state, 1)
        probs  = mcts.search(state, 1)
        # Illegal actions must be zero
        assert np.all(probs[valid == 0] == 0.0)

    def test_temperature_zero_is_one_hot(self, mcts, game):
        state = game.get_initial_state()
        probs = mcts.search(state, 1, temperature=0)
        assert probs.max() == 1.0
        assert (probs > 0).sum() == 1

    def test_finds_immediate_win(self, game, enc, net):
        """
        With one P1 man and one P2 man in jump range, MCTS with sufficient
        simulations should heavily favour the winning capture.
        """
        mcts_ = MCTS(game, enc, net, num_simulations=50, dirichlet_eps=0)
        state = blank_state(game)
        # P1 at (4,3) can jump P2 at (3,4) → lands at (2,5), only P1 piece left
        place(state, {(4, 3): P1_MAN, (3, 4): P2_MAN})
        valid = game.get_valid_moves(state, 1)
        probs = mcts_.search(state, 1, temperature=0)
        # The only legal move should be the capture
        assert (probs > 0).sum() == 1
        assert np.argmax(probs) in np.where(valid)[0]


# ── Eval cache ────────────────────────────────────────────────────────────────

class TestEvalCache:
    def test_cache_populated_on_first_eval(self, game, enc, net):
        mcts_ = MCTS(game, enc, net, num_simulations=1, dirichlet_eps=0)
        net.eval()
        node  = Node(game.get_initial_state(), 1)
        mcts_._network_eval(node)
        assert len(mcts_._eval_cache) == 1

    def test_cache_hit_returns_same_result(self, game, enc, net):
        mcts_ = MCTS(game, enc, net, num_simulations=1, dirichlet_eps=0)
        net.eval()
        node = Node(game.get_initial_state(), 1)
        p1, v1, valid1 = mcts_._network_eval(node)
        p2, v2, valid2 = mcts_._network_eval(node)
        np.testing.assert_array_equal(p1, p2)
        assert v1 == v2
        np.testing.assert_array_equal(valid1, valid2)
        # Second call is a hit — cache size unchanged
        assert len(mcts_._eval_cache) == 1

    def test_different_player_is_separate_cache_entry(self, game, enc, net):
        mcts_ = MCTS(game, enc, net, num_simulations=1, dirichlet_eps=0)
        net.eval()
        state = game.get_initial_state()
        mcts_._network_eval(Node(state, 1))
        mcts_._network_eval(Node(state, -1))
        assert len(mcts_._eval_cache) == 2

    def test_cache_evicts_oldest_when_full(self, game, enc, net):
        """With cache_size=1, a second unique position evicts the first."""
        mcts_ = MCTS(game, enc, net, num_simulations=1, dirichlet_eps=0, cache_size=1)
        net.eval()
        s1 = game.get_initial_state()
        s2 = blank_state(game)
        place(s2, {(5, 0): P1_MAN})
        mcts_._network_eval(Node(s1, 1))
        mcts_._network_eval(Node(s2, 1))
        assert len(mcts_._eval_cache) == 1
