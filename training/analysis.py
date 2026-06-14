"""
Game Quality Analysis
=====================
Computes metrics that reveal *how* the model is playing, not just *whether*
it wins. These go beyond win/loss rates to answer questions like:

  "Is the model exploring a variety of openings, or memorising one line?"
  "Are games getting shorter as the model improves?"
  "Does the model's value estimate actually correlate with who wins?"
  "Is the policy becoming too confident (collapsing) or too diffuse?"

All functions take game records produced by the evaluator or self-play module
and return plain dicts suitable for direct MLflow logging.

Metrics reference
-----------------
game_length_stats
    Mean / std / min / max number of half-moves per game. Improving models
    tend to produce shorter games as they capitalise on advantages faster.

outcome_distribution
    Raw counts of {p1_wins, p2_wins, draws}. A large draw rate suggests
    the model is learning to play passively (safe but weak).

colour_win_rates
    Win rate broken down by which colour the winner played. Should trend
    toward 50/50; a persistent colour bias indicates the model has learned
    an imbalanced strategy.

opening_entropy
    Shannon entropy of the first-move distribution across a batch of games.
    High entropy = diverse openings. Low entropy = always playing the same
    opening (dangerous: exploitable by an opponent who prepares against it).

policy_entropy
    Average Shannon entropy of the MCTS output distributions across all
    recorded moves. Naturally decreases as training progresses (the model
    becomes more decisive). A sudden collapse to near-zero is a warning sign.

value_calibration_mae
    Mean absolute error between each recorded position's value estimate and
    the actual game outcome. A well-calibrated model should have low MAE;
    high MAE means the value head is unreliable.

avg_pieces_remaining
    Average total pieces on board when games end. Fewer pieces = games end
    by capture; more pieces = games end by draw or stalemate.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from training.evaluator import GameRecord


def compute_game_metrics(records: list[GameRecord]) -> dict[str, float]:
    """
    Compute all game-level quality metrics from a list of GameRecord objects.

    Returns a flat dict of metric_name → float suitable for mlflow.log_metrics().

    Parameters
    ----------
    records : List of GameRecord from a tournament or self-play batch.
    """
    if not records:
        return {}

    lengths   = [r.num_moves for r in records]
    pieces    = [r.pieces_remaining for r in records]

    p1_wins  = sum(1 for r in records if r.challenger_is_p1  and r.outcome > 0)
    p1_wins += sum(1 for r in records if not r.challenger_is_p1 and r.outcome < 0)
    p2_wins  = sum(1 for r in records if r.challenger_is_p1  and r.outcome < 0)
    p2_wins += sum(1 for r in records if not r.challenger_is_p1 and r.outcome > 0)
    draws    = sum(1 for r in records if r.outcome == 0)
    total    = len(records)

    return {
        # Game length
        "game_length_mean":   float(np.mean(lengths)),
        "game_length_std":    float(np.std(lengths)),
        "game_length_min":    float(np.min(lengths)),
        "game_length_max":    float(np.max(lengths)),
        # Outcome distribution
        "p1_win_rate":        p1_wins / total,
        "p2_win_rate":        p2_wins / total,
        "draw_rate":          draws   / total,
        # Material
        "avg_pieces_remaining": float(np.mean(pieces)),
    }


def compute_opening_entropy(first_actions: list[int], action_size: int) -> float:
    """
    Shannon entropy of the distribution of first moves across a batch of games.

    Entropy is normalised to [0, 1] by dividing by log2(action_size), so
    1.0 = perfectly uniform openings, 0.0 = always the same first move.

    Parameters
    ----------
    first_actions : List of action indices chosen as the first move in each game.
    action_size   : Total number of possible actions (for normalisation).
    """
    if not first_actions:
        return 0.0

    counts = Counter(first_actions)
    total  = len(first_actions)
    probs  = [c / total for c in counts.values()]
    raw_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    max_entropy = math.log2(action_size)
    return raw_entropy / max_entropy if max_entropy > 0 else 0.0


def compute_policy_entropy(policy_batch: list[np.ndarray]) -> float:
    """
    Average Shannon entropy of MCTS output distributions over a set of moves.

    Each policy is a probability distribution over actions. Entropy measures
    how "spread out" the probability mass is:
      High entropy → uncertain, exploratory
      Low entropy  → confident, decisive

    A healthy training signal shows entropy decreasing over iterations as the
    model becomes more decisive, but never collapsing to near-zero.

    Parameters
    ----------
    policy_batch : List of policy arrays, each shape (action_size,).
    """
    if not policy_batch:
        return 0.0

    entropies = []
    for policy in policy_batch:
        # Only consider non-zero entries to avoid log(0)
        p       = policy[policy > 0]
        entropy = -float(np.sum(p * np.log2(p)))
        entropies.append(entropy)

    return float(np.mean(entropies))



