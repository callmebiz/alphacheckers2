"""
Self-Play
=========
Generates training data by having the current best model play games against
itself using MCTS. The resulting (state, policy, outcome) tuples are the
raw material for the network's supervised learning step.

How a training example is created
----------------------------------
At each move during a self-play game:
  1. Run MCTS from the current position — produces a visit-count distribution π(a).
  2. Record the tuple (encoded_state, π, current_player).
  3. Sample an action from π and apply it to get the next state.

Once the game ends with outcome z ∈ {+1, 0, -1}:
  4. Assign value targets: for every recorded (state, π, player),
     the value target is  +z  if player == winner, else  -z.

This retrospective assignment is necessary because we don't know the outcome
until the game is over. The network learns to predict z from the position.

Temperature schedule
--------------------
For the first `temp_drop_move` plies (half-moves), actions are sampled with
temperature τ_init — producing diverse, exploratory openings. After that,
temperature drops to τ_final (usually 0, i.e. greedy). This balance ensures:
  - A wide variety of opening positions in the training data (prevents memorisation)
  - Precise endgame play that the network actually needs to learn

Per-move search statistics
--------------------------
After each MCTS search, raw_visit_probs() (temperature=1 proportions) is used
to compute per-move entropy and top-1 probability. These measure how spread out
the model's "attention" is across legal moves:

  high entropy  → many plausible moves, uncertain position
  low entropy   → one dominant move, model is confident
  high top1     → similar to low entropy — single move dominates

Aggregate stats (mean, min, std) across all moves this iteration are returned
alongside training examples for MLflow logging.

Game replay format
------------------
Each game is saved as a JSON file in the run's replay directory so it can be
viewed in the web UI. Two probability arrays are stored per move:

  probs       — raw visit proportions (temperature=1), used for the heatmap.
                Always informative — late-game moves don't collapse to one-hot.
  train_probs — temperature-adjusted probs actually used for training/sampling.
                One-hot for late-game moves (temperature=0). Stored so the UI
                can show "what the training signal looked like" alongside the
                actual MCTS confidence distribution.

Parallelism note
----------------
Self-play is CPU-bound (MCTS dominates). For large-scale training, consider
running multiple self-play processes concurrently with separate worker scripts.
The current implementation is sequential for simplicity and Windows compat.
"""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import as_completed
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from core.game import Checkers
from core.encoder import StateEncoder
from core.model import AlphaNet
from core.mcts import MCTS
from training.config import RunConfig


# ── Self-play statistics ──────────────────────────────────────────────────────

@dataclass
class SelfPlayStats:
    """
    Aggregate statistics from a batch of self-play games.

    Game-outcome fields
    -------------------
    num_games, p1_wins, p2_wins, draws, total_moves — straightforward counts.

    Per-move search statistics
    --------------------------
    Computed from the raw (temperature=1) MCTS visit distributions for every
    move in every game this iteration.  These capture how spread or concentrated
    the model's consideration of legal moves is at each position.

      move_entropy_mean  — mean Shannon entropy (bits) across all moves.
                           Naturally falls as training progresses and the model
                           becomes more decisive.
      move_entropy_min   — minimum single-move entropy; the most decisive
                           decision made this iteration.
      move_entropy_std   — std of per-move entropies; high std means the model
                           is very sure about some positions but uncertain about
                           others (healthy variety).
      top1_prob_mean     — mean of max(visit_probs) per move; intuitively the
                           average "confidence" of the best move chosen.
    """
    num_games:         int
    p1_wins:           int
    p2_wins:           int
    draws:             int
    total_moves:       int
    move_entropy_mean: float = 0.0
    move_entropy_min:  float = 0.0
    move_entropy_std:  float = 0.0
    top1_prob_mean:    float = 0.0

    @property
    def p1_win_rate(self) -> float:
        return self.p1_wins / self.num_games if self.num_games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.num_games if self.num_games else 0.0

    @property
    def avg_game_length(self) -> float:
        return self.total_moves / self.num_games if self.num_games else 0.0


def _move_entropy(probs: np.ndarray) -> float:
    """Shannon entropy (bits) of a probability vector; 0-probability bins excluded."""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


# ── Parallel worker ───────────────────────────────────────────────────────────

def _selfplay_worker(args: tuple):
    """
    Entry point for each self-play worker process.

    Must be a module-level function so Python's multiprocessing spawn mode
    (used on Windows) can pickle and import it in the child process.

    All game/model objects are reconstructed here from scratch — nothing is
    shared across process boundaries except the model weights (state_dict,
    a plain dict of CPU tensors) and the RunConfig (a picklable dataclass).
    """
    weights, config, iteration, game_idx, save_replay, replay_dir, mlflow_run_name = args
    game    = Checkers()
    encoder = StateEncoder(game)
    model   = AlphaNet(
        num_channels=encoder.num_channels,
        action_size=game.action_size,
        num_resblocks=config.model.num_resblocks,
        num_hidden=config.model.num_hidden,
    )
    model.load_state_dict(weights)
    model.eval()
    return play_game(
        game, encoder, model, config, torch.device("cpu"),
        save_replay=save_replay,
        replay_dir=replay_dir,
        iteration=iteration,
        mlflow_run_name=mlflow_run_name,
    )


# ── Single game ───────────────────────────────────────────────────────────────

def play_game(
    game: Checkers,
    encoder: StateEncoder,
    model: AlphaNet,
    config: RunConfig,
    device: torch.device,
    save_replay: bool = False,
    replay_dir: str | None = None,
    iteration: int = 0,
    mlflow_run_name: str = "",
) -> tuple[list[tuple[torch.Tensor, np.ndarray, float]], int, int,
           list[float], list[float], list[bool]]:
    """
    Play one complete self-play game and return training examples + move stats.

    Parameters
    ----------
    game            : Checkers engine instance.
    encoder         : StateEncoder for the game.
    model           : Current best model (used read-only — not trained here).
    config          : RunConfig holding MCTS and training hyperparameters.
    device          : Device the model lives on.
    save_replay     : If True, write a JSON replay file to replay_dir.
    replay_dir      : Directory to write replay files (required if save_replay).
    iteration       : Current training iteration (stored in replay metadata).
    mlflow_run_name : MLflow run name stored in replay JSON for UI labelling.

    Returns
    -------
    (examples, absolute_winner, num_moves, move_entropies, move_top1_probs, move_is_forced)

    examples         — list of (encoded_state, mcts_policy, value_target).
    absolute_winner  — 1=P1 won, -1=P2 won, 0=draw.
    num_moves        — total half-moves played.
    move_entropies   — per-move Shannon entropy of raw visit distributions (bits).
    move_top1_probs  — per-move maximum visit probability.
    move_is_forced   — True when there was exactly one legal move (forced capture);
                       used to exclude trivially-zero entropies from the min stat.
    """
    mc = config.mcts
    mcts = MCTS(
        game, encoder, model,
        num_simulations=mc.num_simulations,
        c_puct=mc.c_puct,
        dirichlet_eps=mc.dirichlet_eps,
        dirichlet_alpha=mc.dirichlet_alpha,
        device=device,
    )

    state           = game.get_initial_state()
    player          = 1
    move_number     = 0
    recorded: list[tuple[torch.Tensor, np.ndarray, int]] = []
    replay_moves: list[dict] = []
    move_entropies:  list[float] = []
    move_top1_probs: list[float] = []
    move_is_forced:  list[bool]  = []
    resign_counts: dict[int, int] = {1: 0, -1: 0}
    resigned = False

    while True:
        value, terminated = game.get_value_and_terminated(state, player)
        if terminated:
            outcome = value
            break

        # Temperature: explore early, exploit late
        temp = mc.temperature_init if move_number < mc.temp_drop_move else mc.temperature_final

        encoded    = encoder.encode(state, player)
        train_prob = mcts.search(state, player, temperature=temp)
        raw_prob   = mcts.raw_visit_probs()  # temp=1 proportions — free, root already built

        # Resignation check — only after the model has seen enough of the game.
        # Uses the root Q-value (MCTS-averaged, not raw NN output) for stability.
        if mc.resign_threshold < 1.0 and move_number >= mc.resign_min_move:
            root_val = mcts.root_value()
            if root_val < -mc.resign_threshold:
                resign_counts[player] += 1
                if resign_counts[player] >= mc.resign_consecutive:
                    outcome  = -1.0  # resigning player loses
                    resigned = True
                    break
            else:
                resign_counts[player] = 0

        # Sample the actual move from the training distribution
        action = int(np.random.choice(game.action_size, p=train_prob))

        recorded.append((encoded, train_prob, player))

        # Per-move spread stats — flag forced captures (n_legal==1) so the
        # caller can exclude trivially-zero entropies from the min stat.
        n_legal = int(np.count_nonzero(raw_prob))
        move_entropies.append(_move_entropy(raw_prob))
        move_top1_probs.append(float(raw_prob.max()))
        move_is_forced.append(n_legal == 1)

        if save_replay:
            # Store only non-zero probability entries (sparse dict).
            # Dense arrays are ~170 floats; typical MCTS visits 5-20 actions,
            # so sparse dicts cut prob storage by ~90%.
            replay_moves.append({
                "board":       game.board_to_list(state),
                "player":      player,
                "action":      action,
                "probs":       {str(i): round(float(p), 5) for i, p in enumerate(raw_prob)   if p > 0},
                "train_probs": {str(i): round(float(p), 5) for i, p in enumerate(train_prob) if p > 0},
            })

        state = game.get_next_state(state, action, player)

        if state["jump_again"] is None:
            player = game.get_opponent(player)
            move_number += 1

    # `player` is the terminal player: the one who resigned, couldn't move, or
    # whose turn it is at the end.  `outcome` is from that player's perspective.
    terminal_player = player
    examples = [
        (enc_state, policy, outcome if p == terminal_player else -outcome)
        for enc_state, policy, p in recorded
    ]

    if outcome < 0:
        absolute_winner = game.get_opponent(player)
    elif outcome > 0:
        absolute_winner = player
    else:
        absolute_winner = 0

    if save_replay and replay_dir:
        _write_replay(
            replay_dir, iteration, replay_moves, outcome,
            len(recorded), absolute_winner, mlflow_run_name, resigned=resigned,
        )

    return examples, absolute_winner, len(recorded), move_entropies, move_top1_probs, move_is_forced


# ── Batch generation ──────────────────────────────────────────────────────────

def generate_games(
    n_games: int,
    game: Checkers,
    encoder: StateEncoder,
    model: AlphaNet,
    config: RunConfig,
    device: torch.device,
    iteration: int = 0,
    save_replay_every: int = 5,
    stop_fn=lambda: False,
    mlflow_run_name: str = "",
    executor=None,
) -> tuple[list[tuple[torch.Tensor, np.ndarray, float]], SelfPlayStats]:
    """
    Play *n_games* self-play games and return all training examples + stats.

    One replay file is saved every *save_replay_every* games so the UI can
    show representative games without storing every single one.

    Parameters
    ----------
    n_games           : Number of games to generate this iteration.
    game              : Checkers engine.
    encoder           : StateEncoder.
    model             : Current best model.
    config            : RunConfig.
    device            : Device the model lives on.
    iteration         : Current training iteration (for replay metadata).
    save_replay_every : Save a replay JSON every N games.
    mlflow_run_name   : MLflow run name stored in each replay for UI labelling.
    executor          : If a ProcessPoolExecutor is provided, games are run in
                        parallel worker processes.  When None, games run
                        sequentially in the calling process.

    Returns
    -------
    (all_examples, stats) — training examples plus SelfPlayStats for logging.
    """
    if executor is not None:
        return _generate_games_parallel(
            n_games, model, config, iteration,
            save_replay_every, stop_fn, mlflow_run_name, executor,
        )

    # ── Sequential path ───────────────────────────────────────────────────────
    model.eval()
    all_examples: list[tuple[torch.Tensor, np.ndarray, float]] = []
    p1_wins = p2_wins = draws = total_moves = 0
    all_entropies:  list[float] = []
    all_top1_probs: list[float] = []
    all_is_forced:  list[bool]  = []
    os.makedirs(config.replay_dir, exist_ok=True)

    bar = tqdm(range(n_games), desc="  self-play", unit="game", leave=False)
    for game_idx in bar:
        if stop_fn():
            break
        save = (game_idx % save_replay_every == 0)
        examples, winner, num_moves, m_ent, m_top1, m_forced = play_game(
            game, encoder, model, config, device,
            save_replay=save,
            replay_dir=config.replay_dir,
            iteration=iteration,
            mlflow_run_name=mlflow_run_name,
        )
        all_examples.extend(examples)
        total_moves += num_moves
        all_entropies.extend(m_ent)
        all_top1_probs.extend(m_top1)
        all_is_forced.extend(m_forced)
        if winner == 1:
            p1_wins += 1
        elif winner == -1:
            p2_wins += 1
        else:
            draws += 1
        bar.set_postfix(examples=len(all_examples))

    return _build_stats(p1_wins, p2_wins, draws, total_moves, all_examples,
                        all_entropies, all_top1_probs, all_is_forced)


def _generate_games_parallel(
    n_games: int,
    model: AlphaNet,
    config: RunConfig,
    iteration: int,
    save_replay_every: int,
    stop_fn,
    mlflow_run_name: str,
    executor,
) -> tuple[list, "SelfPlayStats"]:
    """Parallel implementation: submits all games to the pool, collects futures."""
    os.makedirs(config.replay_dir, exist_ok=True)
    # Snapshot weights once; workers each load a copy — safe, no shared state.
    weights = {k: v.cpu() for k, v in model.state_dict().items()}

    futures = []
    for game_idx in range(n_games):
        if stop_fn():
            break
        save = (game_idx % save_replay_every == 0)
        args = (weights, config, iteration, game_idx, save, config.replay_dir, mlflow_run_name)
        futures.append(executor.submit(_selfplay_worker, args))

    all_examples: list = []
    p1_wins = p2_wins = draws = total_moves = 0
    all_entropies:  list[float] = []
    all_top1_probs: list[float] = []
    all_is_forced:  list[bool]  = []

    bar = tqdm(total=len(futures), desc="  self-play", unit="game", leave=False)
    for fut in as_completed(futures):
        examples, winner, num_moves, m_ent, m_top1, m_forced = fut.result()
        all_examples.extend(examples)
        total_moves += num_moves
        all_entropies.extend(m_ent)
        all_top1_probs.extend(m_top1)
        all_is_forced.extend(m_forced)
        if winner == 1:
            p1_wins += 1
        elif winner == -1:
            p2_wins += 1
        else:
            draws += 1
        bar.set_postfix(examples=len(all_examples))
        bar.update(1)
    bar.close()

    return _build_stats(p1_wins, p2_wins, draws, total_moves, all_examples,
                        all_entropies, all_top1_probs, all_is_forced)


def _build_stats(
    p1_wins, p2_wins, draws, total_moves, all_examples,
    all_entropies, all_top1_probs, all_is_forced,
):
    """Shared stat-computation used by both sequential and parallel paths."""
    num_completed = p1_wins + p2_wins + draws
    ent_arr  = np.array(all_entropies,  dtype=np.float64) if all_entropies  else np.zeros(1)
    top1_arr = np.array(all_top1_probs, dtype=np.float64) if all_top1_probs else np.zeros(1)
    free_entropies = [e for e, forced in zip(all_entropies, all_is_forced) if not forced]
    ent_min = float(min(free_entropies)) if free_entropies else float(ent_arr.min())
    stats = SelfPlayStats(
        num_games=num_completed,
        p1_wins=p1_wins,
        p2_wins=p2_wins,
        draws=draws,
        total_moves=total_moves,
        move_entropy_mean=float(ent_arr.mean()),
        move_entropy_min=ent_min,
        move_entropy_std=float(ent_arr.std()),
        top1_prob_mean=float(top1_arr.mean()),
    )
    return all_examples, stats


# ── Replay persistence ────────────────────────────────────────────────────────

def _write_replay(
    replay_dir: str,
    iteration: int,
    moves: list[dict],
    outcome: float,
    num_moves: int,
    winner: int = 0,
    mlflow_run_name: str = "",
    resigned: bool = False,
) -> None:
    """Write a game replay JSON to replay_dir."""
    game_id = str(uuid.uuid4())[:8]
    payload = {
        "game_id":         game_id,
        "type":            "selfplay",
        "iteration":       iteration,
        "outcome":         outcome,
        "num_moves":       num_moves,
        "winner":          winner,
        "resigned":        resigned,
        "mlflow_run_name": mlflow_run_name,
        "moves":           moves,
    }
    path = os.path.join(replay_dir, f"iter{iteration:04d}_{game_id}.json")
    with open(path, "w") as f:
        json.dump(payload, f)
