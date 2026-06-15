"""
Monte Carlo Tree Search (MCTS)
==============================
AlphaZero-style MCTS that uses a neural network to guide search.

Why MCTS?
---------
A neural network alone gives a reasonable policy and value estimate, but it
can still make blunders because it only looks at the board once. MCTS wraps
the network in a search that *simulates* many possible continuations and
accumulates evidence across them. The result is a much stronger policy than
the raw network output.

The Four Steps (repeated N times per search)
--------------------------------------------

  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌────────────────┐
  │  SELECT  │────►│ EVALUATE │────►│  EXPAND  │────►│ BACKPROPAGATE  │
  └──────────┘     └──────────┘     └──────────┘     └────────────────┘

  SELECT        Walk the tree from root to a leaf by always choosing
                the child with the highest UCB score (see Node.ucb).

  EVALUATE      Ask the neural network: "how promising is this position?"
                If the position is terminal (game over), use the real outcome.

  EXPAND        Create child nodes for every legal move from this position,
                storing the network's policy probability as each child's prior.

  BACKPROPAGATE Walk back to the root, updating visit counts and value sums.
                Crucially: negate the value at each step, because parent and
                child have opposite perspectives (my win is your loss).

After N simulations, actions are sampled proportional to visit counts.
The most-visited children are the moves MCTS "believes in."

UCB — balancing exploration and exploitation
--------------------------------------------
At each node, the next child to visit is the one maximising:

    UCB(a) = Q(a)  +  c · P(a) · √N_parent / (1 + N_a)
              ────     ──────────────────────────────────
           exploit              explore

  Q(a)       = average backed-up value — "what did past simulations find?"
  P(a)       = neural network prior    — "what does the network think?"
  N_a        = visit count             — falls as a node is explored more
  c          = exploration constant    — higher = more adventurous search

Dirichlet noise (self-play only)
---------------------------------
Without noise, the agent always explores the same moves the network prefers.
Adding Dirichlet noise to root priors forces occasional exploration of
lower-prior moves, which can reveal better strategies during training.

Player perspective and value negation
--------------------------------------
Values are always stored from the perspective of the player at that node.
When backpropagating, we compare each ancestor's player to the leaf's player:
  same player  → add +value  (good for us = good for our ancestor)
  diff player  → add -value  (good for us = bad for our opponent-ancestor)
This correctly handles multi-jump chains where the same player moves twice.

Performance notes
-----------------
_network_eval caches (board_bytes, player) → (policy, value, valid) in an
LRU dict so repeated visits to the same position during one game skip the
GPU round-trip.  The valid-move array computed here is passed straight into
_expand so get_valid_moves is called at most once per unique leaf position.
model.eval() is the caller's responsibility; MCTS never toggles it.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from core.game import Checkers
from core.encoder import StateEncoder
from core.model import AlphaNet


# ── Tree node ─────────────────────────────────────────────────────────────────

class Node:
    """
    One node in the MCTS search tree.

    Each node represents a unique game position reachable from the root.
    Two statistics accumulate across all simulations that pass through it:

      visit_count  N(s,a) — how many simulations visited this node.
      value_sum    W(s,a) — total backed-up value across all visits.

    Together they form the Q-value: Q = W / N (average outcome).

    Attributes
    ----------
    state       : Game state dict at this position.
    player      : Whose turn it is to move here.
    prior       : P(s,a) — the network's prior probability for the action
                  that *led to* this node (set by the parent on expansion).
    visit_count : N(s,a).
    value_sum   : W(s,a).
    children    : Maps action index → child Node (populated on expansion).
    """

    __slots__ = ('state', 'player', 'prior', 'visit_count', 'value_sum', 'children')

    def __init__(self, state: dict, player: int, prior: float = 0.0):
        self.state       = state
        self.player      = player
        self.prior       = prior
        self.visit_count = 0
        self.value_sum   = 0.0
        self.children: dict[int, Node] = {}

    @property
    def q_value(self) -> float:
        """Average backed-up value. Returns 0 if never visited."""
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def is_leaf(self) -> bool:
        """True if this node has never been expanded (no children yet)."""
        return len(self.children) == 0

    def ucb(self, parent_visit_count: int, c: float) -> float:
        """
        Upper Confidence Bound score.

            UCB(a) = Q(a) + c · P(a) · √N_parent / (1 + N_a)

        The exploration term naturally shrinks as N_a grows, so MCTS
        automatically shifts from broad exploration to focused exploitation.

        Parameters
        ----------
        parent_visit_count : Visit count of this node's parent.
        c                  : Exploration constant (higher = more exploring).
        """
        explore = c * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.q_value + explore

    def best_child(self, c: float) -> tuple[int, Node]:
        """Return (action, child) with the highest UCB score."""
        return max(
            self.children.items(),
            key=lambda kv: kv[1].ucb(self.visit_count, c),
        )


# ── MCTS controller ───────────────────────────────────────────────────────────

class MCTS:
    """
    Runs MCTS from a given game state and returns action probabilities.

    Quick usage
    -----------
    mcts   = MCTS(game, encoder, model)
    probs  = mcts.search(state, player, temperature=1.0)
    action = np.random.choice(game.action_size, p=probs)

    Parameters
    ----------
    game            : Checkers game engine.
    encoder         : StateEncoder — converts state dicts to tensors.
    model           : AlphaNet — provides policy and value estimates.
                      Must already be in eval mode; MCTS never calls .eval().
    num_simulations : Simulations per search call. More = stronger, slower.
    c_puct          : UCB exploration constant (typical: 1.0–2.0).
    dirichlet_eps   : Weight of Dirichlet noise at root (0 = disabled).
    dirichlet_alpha : Concentration of noise distribution (lower = more spread).
    device          : PyTorch device for inference.
    cache_size      : Max unique positions to cache per MCTS instance. Each
                      entry stores policy + value + valid-move array (~700 B).
    """

    def __init__(
        self,
        game: Checkers,
        encoder: StateEncoder,
        model: AlphaNet,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_eps: float = 0.25,
        dirichlet_alpha: float = 0.3,
        device: torch.device | None = None,
        cache_size: int = 2048,
    ):
        self.game            = game
        self.encoder         = encoder
        self.model           = model
        self.num_simulations = num_simulations
        self.c_puct          = c_puct
        self.dirichlet_eps   = dirichlet_eps
        self.dirichlet_alpha = dirichlet_alpha
        self.device          = device or torch.device('cpu')

        # LRU cache: (board_bytes, player) → (policy, value, valid).
        # Keyed on current board only — history approximation is acceptable
        # for a performance cache since the network is deterministic for any
        # given board state.  Entries accumulate across search() calls within
        # the same game instance, maximising transposition reuse.
        self._eval_cache: OrderedDict[
            tuple[bytes, int], tuple[np.ndarray, float, np.ndarray]
        ] = OrderedDict()
        self._cache_size  = cache_size
        self._last_root: Node | None = None  # retained for raw_visit_probs()

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, state: dict, player: int, temperature: float = 1.0) -> np.ndarray:
        """
        Run MCTS and return a probability distribution over actions.

        Shape: (action_size,). Only legal actions have non-zero probability.

        Temperature controls how "sharp" the distribution is:
          τ > 1  → more uniform (good for early-game exploration)
          τ → 0  → concentrates on the single most-visited move (exploitation)

        Parameters
        ----------
        state       : Current game state dict.
        player      : Current player (+1 or -1).
        temperature : Sharpness of the output distribution.
        """
        root = Node(state, player)

        # Expand root before simulations so selection has children to pick from
        policy, _, valid = self._network_eval(root)
        _expand(root, policy, valid, self.game)
        self._add_dirichlet_noise(root)

        # Forced move: only one legal action — skip all but 1 simulation.
        # The single sim propagates a real value back to root so root_value()
        # still works for resign detection.
        n_sims = 1 if len(root.children) == 1 else self.num_simulations
        for _ in range(n_sims):
            # ── Step 1: SELECT ───────────────────────────────────────────
            leaf, path = self._select(root)

            # ── Steps 2+3: EVALUATE + EXPAND ────────────────────────────
            value = self._evaluate_and_expand(leaf)

            # ── Step 4: BACKPROPAGATE ────────────────────────────────────
            _backpropagate(path, value)

        self._last_root = root  # save so raw_visit_probs() can be called after
        return self._action_probs(root, temperature)

    def raw_visit_probs(self) -> np.ndarray:
        """
        Return visit-count proportions from the most recent search call.

        Equivalent to calling search() again with temperature=1, but free
        because the root node is already built.  Use this for visualisation
        instead of the temperature-adjusted training probs — late-game moves
        use temperature=0 (one-hot) which is uninformative for a heatmap.
        """
        if self._last_root is None:
            raise RuntimeError("No search has been performed yet")
        return self._action_probs(self._last_root, temperature=1.0)

    def root_value(self) -> float:
        """
        Average backed-up Q-value at the root from the most recent search.

        Represents the expected outcome from the root player's perspective
        after all simulations, in [-1, 1].  Values near -1 indicate the
        model believes it is losing heavily — used for resign detection.
        """
        if self._last_root is None:
            raise RuntimeError("No search has been performed yet")
        return self._last_root.q_value

    # ── MCTS steps ────────────────────────────────────────────────────────────

    def _select(self, root: Node) -> tuple[Node, list[Node]]:
        """
        Step 1 — SELECT.

        Follow the highest-UCB child at each level until we reach a leaf
        (a node that has never been expanded). Collect the full path so
        backpropagation can update every node visited.
        """
        node = root
        path = [root]

        while not node.is_leaf:
            _, node = node.best_child(self.c_puct)
            path.append(node)

        return node, path

    def _evaluate_and_expand(self, node: Node) -> float:
        """
        Steps 2 + 3 — EVALUATE then EXPAND.

        If the position is terminal return the actual game result (no
        network call needed). Otherwise, query the network and expand.

        Returns the value at this node from node.player's perspective.
        """
        value, is_terminal = self.game.get_value_and_terminated(node.state, node.player)
        if is_terminal:
            return float(value)

        policy, value, valid = self._network_eval(node)
        _expand(node, policy, valid, self.game)
        return value

    def _network_eval(
        self, node: Node
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Query the neural network for a masked policy, value estimate, and
        valid-move array.  Results are cached by (board_bytes, player) so
        repeated visits to the same position skip the network entirely.

        The state is encoded from node.player's perspective. Illegal-move
        logits are set to -inf before softmax so they get zero probability.

        Caller is responsible for putting the model in eval mode before
        constructing this MCTS instance.

        Returns
        -------
        policy : np.ndarray (action_size,) — masked, normalised probabilities.
        value  : float — network's value estimate for node.player.
        valid  : np.ndarray (action_size,) bool — legal-move mask (reused by
                 _expand to avoid a redundant get_valid_moves call).
        """
        cache_key = (node.state['board'].tobytes(), node.player)
        if cache_key in self._eval_cache:
            self._eval_cache.move_to_end(cache_key)
            return self._eval_cache[cache_key]

        tensor = (
            self.encoder.encode(node.state, node.player)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            logits, value_t = self.model(tensor)

        # Zero out illegal moves in log-space before normalising
        valid  = self.game.get_valid_moves(node.state, node.player)
        mask   = torch.from_numpy(valid.astype(bool)).to(self.device)
        logits[0][~mask] = float('-inf')
        policy = F.softmax(logits[0], dim=0).cpu().numpy()
        value  = float(value_t.item())

        # Store; evict oldest entry when capacity exceeded
        if len(self._eval_cache) >= self._cache_size:
            self._eval_cache.popitem(last=False)
        self._eval_cache[cache_key] = (policy, value, valid)

        return policy, value, valid

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add_dirichlet_noise(self, root: Node) -> None:
        """
        Blend Dirichlet noise into the root's prior probabilities.

        Noisy prior:  P'(a) = (1 - ε) · P(a)  +  ε · η(a)
        where η ~ Dirichlet(α).

        This prevents the search from always following the same network-
        preferred lines during self-play training.
        """
        if self.dirichlet_eps == 0 or not root.children:
            return
        actions = list(root.children)
        noise   = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        eps     = self.dirichlet_eps
        for action, eta in zip(actions, noise):
            c = root.children[action]
            c.prior = (1 - eps) * c.prior + eps * eta

    def _action_probs(self, root: Node, temperature: float) -> np.ndarray:
        """
        Convert root visit counts to a probability distribution.

            π(a) ∝ N(a)^(1/τ)

        τ = 1 → proportional to visit counts.
        τ → 0 → one-hot on the most-visited action (greedy).
        """
        counts = np.zeros(self.game.action_size, dtype=np.float32)
        for action, child in root.children.items():
            counts[action] = child.visit_count

        if temperature == 0:
            best          = int(np.argmax(counts))
            counts[:]     = 0.0
            counts[best]  = 1.0
            return counts

        counts = counts ** (1.0 / temperature)
        return counts / counts.sum()


# ── Module-level helpers (no self dependency) ─────────────────────────────────

def _expand(
    node: Node,
    policy: np.ndarray,
    valid: np.ndarray,
    game: Checkers,
) -> None:
    """
    Expand *node* by creating a child for every legal action.

    Each child receives the network's prior probability for the action
    that leads to it. Children start with zero visits; they'll be
    evaluated and expanded during future simulations.

    Parameters
    ----------
    valid : Legal-move mask from _network_eval — passed in to avoid a
            redundant get_valid_moves call on the same position.
    """
    for action in np.where(valid)[0]:
        next_state  = game.get_next_state(node.state, int(action), node.player)
        # During a multi-jump the same player moves again
        next_player = (
            node.player if next_state['jump_again']
            else game.get_opponent(node.player)
        )
        node.children[int(action)] = Node(
            next_state, next_player, prior=float(policy[action])
        )


def _backpropagate(path: list[Node], value: float) -> None:
    """
    Update every node on the path from leaf back to root.

    We compare each ancestor's player to the leaf's player to decide the
    sign of the value — this correctly handles multi-jump chains where the
    same player occupies consecutive nodes in the path.

      same player as leaf  → add +value  (their win is our win)
      different player     → add -value  (their win is our loss)
    """
    leaf_player = path[-1].player
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum   += value if node.player == leaf_player else -value
