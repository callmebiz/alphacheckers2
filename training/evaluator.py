"""
Model Evaluator
===============
After each training iteration, the newly trained model challenges the current
best model in a head-to-head tournament. The challenger is promoted to 'best'
only if it wins at least `promotion_threshold` of the games.

Why gate promotion?
-------------------
Without a promotion gate, training noise can cause the model to regress —
a batch of unlucky training examples might push the weights in a slightly
wrong direction. By requiring the new model to prove itself in actual games,
we ensure the 'best' model only ever improves.

Tournament design
-----------------
- N games total, alternating which model plays as Player 1 and Player 2.
  This removes colour bias from the results — a model that only wins as P1
  should not be promoted.
- First two half-moves (one per side) are sampled at temperature=1 from
  the honest MCTS visit distribution so each game explores a different
  opening. All subsequent moves are greedy (temperature=0).
- Dirichlet noise is disabled throughout — the tree is always built cleanly.

Returned statistics
-------------------
TournamentResult holds all the data needed for MLflow logging and UI display:
  wins / draws / losses     — raw game counts
  win_rate                  — wins / total (draws count as 0.5)
  win_rate_ci               — 95% Wilson score confidence interval
  game_stats                — per-game data for the analysis module
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from concurrent.futures import as_completed
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from core.game import Checkers
from core.encoder import StateEncoder
from core.model import AlphaNet
from core.mcts import MCTS
from training.config import RunConfig


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class GameRecord:
    """Statistics for a single tournament game."""
    challenger_is_p1: bool      # True if challenger played as Player 1
    outcome:          float     # +1 challenger wins, 0 draw, -1 champion wins
    num_moves:        int
    pieces_remaining: int       # total pieces on board at game end


@dataclass
class TournamentResult:
    """
    Aggregated results from a full tournament between challenger and champion.

    win_rate
        Fraction of games won by the challenger (draws count as 0.5).
        Promotion happens when win_rate >= promotion_threshold.

    win_rate_ci
        95% confidence interval [lo, hi] using the Wilson score method.
        A wide interval means the sample size is too small to be conclusive.
    """
    wins:       int
    draws:      int
    losses:     int
    win_rate:   float
    win_rate_ci: tuple[float, float]
    game_records: list[GameRecord] = field(default_factory=list)

    @property
    def total_games(self) -> int:
        return self.wins + self.draws + self.losses



# ── Parallel worker ───────────────────────────────────────────────────────────

# Per-worker cached state — same pattern as self_play._W.
_EW: dict = {}


def _eval_worker(args: tuple):
    """
    Entry point for each tournament worker process.

    Module-level so multiprocessing spawn can pickle it.  Both model objects
    are built once per worker process and reused across tournament games.
    Weights are reloaded only when the evaluation iteration changes.
    """
    (challenger_weights, champion_weights, config, game_idx,
     save_replay, replay_dir, mlflow_run_name, iteration) = args

    # One-time setup on first call in this worker process
    if not _EW:
        import os as _os
        _os.environ.setdefault("OMP_NUM_THREADS", "1")
        _os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        game    = Checkers()
        encoder = StateEncoder(game)

        def _build_empty():
            m = AlphaNet(
                num_channels=encoder.num_channels,
                action_size=game.action_size,
                num_resblocks=config.model.num_resblocks,
                num_hidden=config.model.num_hidden,
            )
            m.eval()
            return m

        _EW['game']       = game
        _EW['encoder']    = encoder
        _EW['challenger'] = _build_empty()
        _EW['champion']   = _build_empty()
        _EW['config']     = config
        _EW['ver']        = None

    # Reload weights only when entering a new tournament (new iteration)
    if _EW['ver'] != iteration:
        _EW['challenger'].load_state_dict(challenger_weights)
        _EW['challenger'].eval()
        _EW['champion'].load_state_dict(champion_weights)
        _EW['champion'].eval()
        _EW['ver'] = iteration

    game    = _EW['game']
    encoder = _EW['encoder']
    mc      = config.mcts
    device  = torch.device("cpu")

    eps    = config.eval.eval_noise_eps
    mcts_c = _make_mcts(_EW['challenger'], game, encoder, mc, device, noise=False, eval_noise_eps=eps)
    mcts_m = _make_mcts(_EW['champion'],   game, encoder, mc, device, noise=False, eval_noise_eps=eps)

    challenger_is_p1 = (game_idx % 2 == 0)
    p1_mcts, p2_mcts = (mcts_c, mcts_m) if challenger_is_p1 else (mcts_m, mcts_c)

    outcome_p1, num_moves, pieces, replay_moves = _play_eval_game(
        game, p1_mcts, p2_mcts, save_moves=save_replay
    )
    if save_replay and replay_moves is not None:
        _write_tournament_replay(
            replay_dir, iteration, replay_moves, outcome_p1,
            num_moves, challenger_is_p1, mlflow_run_name,
        )
    challenger_outcome = outcome_p1 if challenger_is_p1 else -outcome_p1
    return challenger_is_p1, challenger_outcome, num_moves, pieces


# ── Tournament ────────────────────────────────────────────────────────────────

def run_tournament(
    challenger: AlphaNet,
    champion: AlphaNet,
    game: Checkers,
    encoder: StateEncoder,
    config: RunConfig,
    device: torch.device,
    stop_fn=lambda: False,
    executor=None,
    iteration: int = 0,
    mlflow_run_name: str = "",
) -> TournamentResult:
    """
    Play `config.eval.tournament_games` games between challenger and champion.

    Games alternate which model plays as Player 1 so colour advantage averages
    out. First two half-moves are sampled at temperature=1 for opening diversity;
    all subsequent moves are greedy (τ=0). No Dirichlet noise throughout.

    Parameters
    ----------
    challenger : Newly trained model — trying to beat the current best.
    champion   : Current best model — defending its title.
    game       : Checkers engine.
    encoder    : StateEncoder.
    config     : RunConfig (reads eval.tournament_games, mcts settings).
    device     : Device for inference.

    Returns
    -------
    TournamentResult with full win/draw/loss breakdown and per-game records.
    """
    if executor is not None:
        return _run_tournament_parallel(
            challenger, champion, config, stop_fn, executor, iteration, mlflow_run_name
        )

    # ── Sequential path ───────────────────────────────────────────────────────
    ec = config.eval
    mc = config.mcts

    os.makedirs(config.replay_dir, exist_ok=True)

    # Build two MCTS instances — one per model, noise disabled for eval
    eps = config.eval.eval_noise_eps
    mcts_challenger = _make_mcts(challenger, game, encoder, mc, device, noise=False, eval_noise_eps=eps)
    mcts_champion   = _make_mcts(champion,   game, encoder, mc, device, noise=False, eval_noise_eps=eps)

    wins = draws = losses = 0
    records: list[GameRecord] = []

    bar = tqdm(range(ec.tournament_games), desc="  eval", unit="game", leave=False)
    for game_idx in bar:
        if stop_fn():
            break
        # Alternate who plays as P1 so colour advantage cancels out
        challenger_is_p1 = (game_idx % 2 == 0)

        if challenger_is_p1:
            p1_mcts, p2_mcts = mcts_challenger, mcts_champion
        else:
            p1_mcts, p2_mcts = mcts_champion, mcts_challenger

        # Save one representative game per tournament (first game, challenger as P1)
        save = (game_idx == 0)
        outcome_p1, num_moves, pieces, replay_moves = _play_eval_game(
            game, p1_mcts, p2_mcts, save_moves=save
        )
        if save and replay_moves is not None:
            _write_tournament_replay(
                config.replay_dir, iteration, replay_moves, outcome_p1,
                num_moves, challenger_is_p1, mlflow_run_name,
            )

        # Convert outcome to challenger's perspective
        challenger_outcome = outcome_p1 if challenger_is_p1 else -outcome_p1

        if challenger_outcome > 0:
            wins += 1
        elif challenger_outcome == 0:
            draws += 1
        else:
            losses += 1

        records.append(GameRecord(
            challenger_is_p1=challenger_is_p1,
            outcome=challenger_outcome,
            num_moves=num_moves,
            pieces_remaining=pieces,
        ))

        total_so_far = wins + draws + losses
        wr = (wins + 0.5 * draws) / total_so_far if total_so_far else 0.0
        bar.set_postfix(W=wins, D=draws, L=losses, wr=f"{wr:.0%}")

    total    = wins + draws + losses
    win_rate = (wins + 0.5 * draws) / total if total else 0.0
    ci       = _wilson_ci(wins + 0.5 * draws, total)

    return TournamentResult(
        wins=wins, draws=draws, losses=losses,
        win_rate=win_rate, win_rate_ci=ci,
        game_records=records,
    )


def _run_tournament_parallel(
    challenger: AlphaNet,
    champion: AlphaNet,
    config: RunConfig,
    stop_fn,
    executor,
    iteration: int = 0,
    mlflow_run_name: str = "",
) -> TournamentResult:
    """Parallel tournament: all games submitted to the pool at once."""
    ec = config.eval
    challenger_weights = {k: v.cpu() for k, v in challenger.state_dict().items()}
    champion_weights   = {k: v.cpu() for k, v in champion.state_dict().items()}

    os.makedirs(config.replay_dir, exist_ok=True)

    futures = []
    for game_idx in range(ec.tournament_games):
        if stop_fn():
            break
        save = (game_idx == 0)
        args = (
            challenger_weights, champion_weights, config, game_idx,
            save, config.replay_dir, mlflow_run_name, iteration,
        )
        futures.append(executor.submit(_eval_worker, args))

    wins = draws = losses = 0
    records: list[GameRecord] = []

    bar = tqdm(total=len(futures), desc="  eval", unit="game", leave=False)
    for fut in as_completed(futures):
        challenger_is_p1, challenger_outcome, num_moves, pieces = fut.result()
        if challenger_outcome > 0:
            wins += 1
        elif challenger_outcome == 0:
            draws += 1
        else:
            losses += 1
        records.append(GameRecord(
            challenger_is_p1=challenger_is_p1,
            outcome=challenger_outcome,
            num_moves=num_moves,
            pieces_remaining=pieces,
        ))
        total_so_far = wins + draws + losses
        wr = (wins + 0.5 * draws) / total_so_far if total_so_far else 0.0
        bar.set_postfix(W=wins, D=draws, L=losses, wr=f"{wr:.0%}")
        bar.update(1)
    bar.close()

    total    = wins + draws + losses
    win_rate = (wins + 0.5 * draws) / total if total else 0.0
    ci       = _wilson_ci(wins + 0.5 * draws, total)
    return TournamentResult(
        wins=wins, draws=draws, losses=losses,
        win_rate=win_rate, win_rate_ci=ci,
        game_records=records,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _play_eval_game(
    game: Checkers,
    p1_mcts: MCTS,
    p2_mcts: MCTS,
    save_moves: bool = False,
) -> tuple[float, int, int, list[dict] | None]:
    """
    Play one evaluation game and return
    (outcome_for_p1, num_moves, pieces_remaining, replay_moves).

    outcome_for_p1: +1 if P1 wins, 0 draw, -1 if P2 wins.
    replay_moves is None when save_moves=False.
    """
    state        = game.get_initial_state()
    player       = 1
    move_count   = 0
    mcts_map     = {1: p1_mcts, -1: p2_mcts}
    replay_moves: list[dict] | None = [] if save_moves else None

    while True:
        value, terminated = game.get_value_and_terminated(state, player)
        if terminated:
            pieces = int(np.abs(state["board"]).sum())
            return value if player == 1 else -value, move_count, pieces, replay_moves

        # First two half-moves (one per side) sampled at temperature=1 so each
        # game explores a different opening; the rest are greedy (temperature=0).
        temp   = 1.0 if move_count < 2 else 0.0
        probs  = mcts_map[player].search(state, player, temperature=temp)
        action = (
            int(np.random.choice(game.action_size, p=probs))
            if temp > 0
            else int(np.argmax(probs))
        )

        if save_moves:
            raw_prob = mcts_map[player].raw_visit_probs()
            replay_moves.append({
                "board":  game.board_to_list(state),
                "player": player,
                "action": action,
                "probs":  {str(i): round(float(p), 5) for i, p in enumerate(raw_prob) if p > 0},
            })

        state  = game.get_next_state(state, action, player)

        if state["jump_again"] is None:
            player = game.get_opponent(player)
        move_count += 1


def _write_tournament_replay(
    replay_dir: str,
    iteration: int,
    moves: list[dict],
    outcome_p1: float,
    num_moves: int,
    challenger_is_p1: bool,
    mlflow_run_name: str = "",
) -> None:
    """Write one tournament game replay JSON to replay_dir."""
    game_id  = str(uuid.uuid4())[:8]
    winner   = 1 if outcome_p1 > 0 else (-1 if outcome_p1 < 0 else 0)
    chall_won = (outcome_p1 > 0) if challenger_is_p1 else (outcome_p1 < 0)
    payload = {
        "game_id":           game_id,
        "type":              "tournament",
        "iteration":         iteration,
        "outcome":           outcome_p1,
        "num_moves":         num_moves,
        "winner":            winner,
        "resigned":          False,
        "challenger_is_p1":  challenger_is_p1,
        "challenger_won":    chall_won,
        "mlflow_run_name":   mlflow_run_name,
        "moves":             moves,
    }
    path = os.path.join(replay_dir, f"iter{iteration:04d}_tourn_{game_id}.json")
    with open(path, "w") as f:
        json.dump(payload, f)


def _make_mcts(
    model: AlphaNet,
    game: Checkers,
    encoder: StateEncoder,
    mc,
    device: torch.device,
    noise: bool,
    eval_noise_eps: float = 0.0,
) -> MCTS:
    """Construct an MCTS instance with optional Dirichlet noise."""
    return MCTS(
        game, encoder, model,
        num_simulations=mc.num_simulations,
        c_puct=mc.c_puct,
        dirichlet_eps=mc.dirichlet_eps if noise else eval_noise_eps,
        dirichlet_alpha=mc.dirichlet_alpha,
        device=device,
    )


def _wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score 95% confidence interval for a proportion.

    More accurate than the naive ±1.96·√(p(1-p)/n) at extreme proportions
    (near 0 or 1), which is exactly when we care most about precision.

    Returns (lower_bound, upper_bound).
    """
    if n == 0:
        return (0.0, 1.0)
    p     = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
