"""
MLflow Tracking
===============
Wraps all MLflow interactions behind a clean interface so the trainer
never imports mlflow directly and tracking concerns stay isolated here.

What gets logged and where
--------------------------
Params (logged once at run start — immutable after that):
  config.*           — every field of the RunConfig, flattened
  device             — resolved torch device string
  torch_version      — for reproducibility across machines

Metrics (logged once per training iteration as a time-series):
  loss/policy        — cross-entropy between network policy and MCTS probs
  loss/value         — MSE between network value and game outcome
  loss/total         — sum of the above
  train/lr           — current learning rate (after scheduler step)
  train/buffer_size  — number of examples in the replay buffer

  selfplay/avg_game_length   — mean game length this iteration
  selfplay/draw_rate         — fraction of games that ended in draws
  selfplay/p1_win_rate       — P1 win fraction (should trend toward ~50%)

  eval/elo                   — current best model ELO rating
  eval/win_rate              — challenger win rate in tournament
  eval/win_rate_ci_lo/hi     — 95% Wilson confidence interval bounds
  eval/tournament_games      — total games played in tournament
  eval/promoted              — 1 if challenger was promoted, 0 if not
  eval/avg_game_length       — mean game length in tournament
  eval/draw_rate             — draw fraction in tournament

  analysis/policy_entropy    — avg entropy of MCTS distributions
  analysis/value_mae         — value head calibration error
  analysis/opening_entropy   — diversity of first moves

Artifacts (files attached to the MLflow run):
  config.json          — full config as JSON (self-describing run)

MLflow UI
---------
Start the tracking UI (separate terminal) with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
Then open http://localhost:5000 in your browser.

Recent MLflow versions require a database backend — the old filesystem store
('mlruns/') is no longer supported. The SQLite URI used above creates a single
file (mlflow.db) in the project root and needs no extra setup.

Remote tracking
---------------
Point MLFLOW_TRACKING_URI to a remote server to share runs between machines
(laptop ↔ cloud GPU):
    export MLFLOW_TRACKING_URI=http://my-server:5000
    python train.py --config full
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import tempfile

import mlflow
import torch

from training.config import RunConfig
from training.evaluator import TournamentResult
from training.analysis import compute_game_metrics


class MLflowTracker:
    """
    Session wrapper around MLflow for one complete training run.

    Usage
    -----
    with MLflowTracker(config, device, experiment="baseline") as tracker:
        tracker.log_training(...)   # once per iteration
        tracker.log_selfplay(...)
        tracker.log_evaluation(...) # on eval iterations

    experiment
        MLflow experiment name — groups related runs for comparison in the UI.
        Use this to express the research question, not the config name.
        E.g. "architecture-search", "sims-ablation", "baseline-medium".
        Defaults to config.name if not provided.

    run_name
        Optional explicit name for this run. If omitted, a descriptive name
        is auto-generated from the key hyperparameters:
        "{config}-{resblocks}x{hidden}-{sims}sims"
        (e.g. "medium-8x96-200sims").
    """

    def __init__(
        self,
        config:     RunConfig,
        device:     torch.device,
        experiment: str | None = None,
        run_name:   str | None = None,
    ):
        self.config    = config
        self.device    = device
        self._experiment = experiment or config.name
        self._run_name_override = run_name
        self._active_run = None
        self._run_name   = ""

        mlflow.set_tracking_uri(config.mlflow_uri)
        _set_experiment(self._experiment)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @property
    def run_name(self) -> str:
        """The run name embedded in replay files for cross-referencing."""
        return self._run_name

    def start(self) -> None:
        """Open an MLflow run and log all static params and tags."""
        name = _auto_run_name(self.config, prefix=self._run_name_override)
        self._active_run = mlflow.start_run(run_name=name)
        try:
            self._run_name = self._active_run.info.run_name or name
        except AttributeError:
            self._run_name = name

        # Tags — metadata that isn't a hyperparameter.
        # git_commit ties every run to the exact code that produced it so you
        # can reproduce or diff runs months later.
        mlflow.set_tags({
            "git_commit": _git_hash(),
            "config":     self.config.name,
            "arch":       f"{self.config.model.num_resblocks}x{self.config.model.num_hidden}",
            "sims":       str(self.config.mcts.num_simulations),
            "device":     str(self.device),
        })

        # Params — all hyperparameters, flattened from the RunConfig dataclass.
        flat = _flatten_dataclass(self.config)
        mlflow.log_params(flat)
        mlflow.log_param("torch_version", torch.__version__)

        # Config JSON artifact — makes the run fully self-describing.
        cfg_dict = dataclasses.asdict(self.config)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(cfg_dict, f, indent=2)
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="config")
        os.unlink(tmp)

    def end(self) -> None:
        """Close the active MLflow run."""
        if self._active_run:
            mlflow.end_run()
            self._active_run = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.end()

    # ── Iteration logging ─────────────────────────────────────────────────────

    def log_training(
        self,
        step: int,
        policy_loss: float,
        value_loss:  float,
        lr:          float,
        buffer_size: int,
    ) -> None:
        """Log network training metrics for one iteration."""
        mlflow.log_metrics({
            "loss/policy":        policy_loss,
            "loss/value":         value_loss,
            "loss/total":         policy_loss + value_loss,
            "train/lr":           lr,
            "train/log10_lr":     math.log10(lr) if lr > 0 else -9.0,
            "train/buffer_size":  buffer_size,
        }, step=step)

    def log_selfplay(
        self,
        step: int,
        avg_game_length:    float,
        draw_rate:          float,
        p1_win_rate:        float,
        policy_entropy:     float,
        move_entropy_mean:  float = 0.0,
        move_entropy_min:   float = 0.0,
        move_entropy_std:   float = 0.0,
        top1_prob_mean:     float = 0.0,
    ) -> None:
        """Log self-play quality metrics for one iteration."""
        mlflow.log_metrics({
            "selfplay/avg_game_length":   avg_game_length,
            "selfplay/draw_rate":         draw_rate,
            "selfplay/p1_win_rate":       p1_win_rate,
            "selfplay/move_entropy_mean": move_entropy_mean,
            "selfplay/move_entropy_min":  move_entropy_min,
            "selfplay/move_entropy_std":  move_entropy_std,
            "selfplay/top1_prob_mean":    top1_prob_mean,
            "analysis/policy_entropy":    policy_entropy,
        }, step=step)

    def log_evaluation(
        self,
        step:     int,
        result:   TournamentResult,
        elo:      float,
        promoted: bool,
        value_mae:        float = 0.0,
        opening_entropy:  float = 0.0,
    ) -> None:
        """Log tournament and evaluation metrics for one iteration."""
        lo, hi = result.win_rate_ci
        # From compute_game_metrics, keep side-based outcome rates (p1/p2/draw)
        # and game-length / material stats.  Note: p1/p2 here refers to which
        # SIDE won (first-mover vs second-mover), not which model — distinct
        # from eval/win_rate which is the challenger's side-adjusted win rate.
        _keep = {
            "p1_win_rate", "p2_win_rate", "draw_rate",
            "game_length_mean", "game_length_std", "avg_pieces_remaining",
        }
        game_metrics = {
            f"eval/{k}": v
            for k, v in compute_game_metrics(result.game_records).items()
            if k in _keep
        }
        mlflow.log_metrics({
            "eval/elo":               elo,
            "eval/win_rate":          result.win_rate,
            "eval/win_rate_ci_lo":    lo,
            "eval/win_rate_ci_hi":    hi,
            "eval/promoted":          int(promoted),
            "eval/wins":              result.wins,
            "eval/draws":             result.draws,
            "eval/losses":            result.losses,
            "analysis/value_mae":     value_mae,
            "analysis/opening_entropy": opening_entropy,
            **game_metrics,
        }, step=step)

    def log_model_artifact(self, checkpoint_path: str, step: int) -> None:
        """Attach a checkpoint file to the current MLflow run."""
        mlflow.log_artifact(checkpoint_path, artifact_path=f"checkpoints/iter_{step:04d}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_hash() -> str:
    """Return the short git commit hash, or 'untracked' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "untracked"


def _auto_run_name(config: RunConfig, prefix: str | None = None) -> str:
    """
    Build a run name that always encodes the key hyperparameters.

    No prefix  → "medium-8x96-200sims"
    With prefix → "my-label-8x96-200sims"

    The param suffix is always appended so runs are self-describing in the
    MLflow UI without having to open individual run details.
    """
    base   = prefix if prefix else config.name
    suffix = f"{config.model.num_resblocks}x{config.model.num_hidden}-{config.mcts.num_simulations}sims"
    return f"{base}-{suffix}"


def _set_experiment(name: str) -> None:
    """
    Set the active MLflow experiment by name, handling the soft-delete case.

    MLflow soft-deletes experiments when you remove them from the UI — they
    are marked 'deleted' rather than purged. Calling mlflow.set_experiment()
    on a soft-deleted name raises an exception instead of just creating a new
    one. This helper restores it first if needed.
    """
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(name)
    if exp is not None and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(name)


def _flatten_dataclass(obj, prefix: str = "") -> dict[str, str]:
    """
    Recursively flatten a dataclass into a dict of dot-separated string keys.
    MLflow params must be strings, so all values are cast with str().
    """
    result = {}
    for f in dataclasses.fields(obj):
        val  = getattr(obj, f.name)
        key  = f"{prefix}{f.name}" if not prefix else f"{prefix}.{f.name}"
        if dataclasses.is_dataclass(val):
            result.update(_flatten_dataclass(val, prefix=key))
        else:
            result[key] = str(val)
    return result
