"""
Training Configuration
======================
All hyperparameters for a training run are defined here as typed Python
dataclasses. Using dataclasses (instead of JSON or plain dicts) gives:
  - IDE autocomplete on every field
  - Type checking at import time
  - Default values that self-document the intended range
  - Easy nesting without custom parsing

Structure
---------
RunConfig
  ├── ModelConfig       — network architecture
  ├── MCTSConfig        — search parameters
  ├── TrainingConfig    — optimiser, batch size, replay buffer
  └── EvalConfig        — tournament rules, ELO, promotion threshold

Three ready-made presets are provided at the bottom of this file:
  DEBUG   — completes in minutes on CPU; used for smoke-testing code changes
  DEV     — a few hours on GPU; good for iterating on hyperparameters
  FULL    — overnight run for a genuinely strong model
"""

from __future__ import annotations

import os
import torch
from dataclasses import dataclass, field


# ── Sub-configs ───────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Neural network architecture parameters."""
    num_resblocks: int = 6    # backbone depth — more = stronger, slower
    num_hidden: int    = 64   # convolutional width


@dataclass
class MCTSConfig:
    """
    Monte Carlo Tree Search parameters.

    temperature_init / temperature_final
        MCTS action probabilities are raised to the power 1/τ before sampling.
        High τ → near-uniform distribution (more exploration of self-play data).
        Low τ → greedy (used later in training and during evaluation).

    temp_drop_move
        Move number within a game after which temperature switches from
        temperature_init to temperature_final. Keeps opening play diverse
        while making endgame moves crisp.

    resign_threshold / resign_min_move / resign_consecutive
        A player resigns when their MCTS root Q-value (their expected outcome)
        falls below -resign_threshold for resign_consecutive of their own turns
        in a row, and at least resign_min_move half-moves have been played.
        Set resign_threshold = 1.0 to disable (value can never be < -1).
    """
    num_simulations:    int   = 200
    c_puct:             float = 1.5
    dirichlet_eps:      float = 0.25
    dirichlet_alpha:    float = 0.3
    temperature_init:   float = 1.0   # high temp for diverse opening play
    temperature_final:  float = 0.0   # greedy after temp_drop_move
    temp_drop_move:     int   = 30    # move threshold for temperature switch
    resign_threshold:   float = 0.95  # resign when root Q < -this; 1.0 = disabled
    resign_min_move:    int   = 10    # don't consider resigning before this half-move
    resign_consecutive: int   = 5     # consecutive own-turns below threshold to resign


@dataclass
class TrainingConfig:
    """
    Optimiser and data pipeline parameters.

    replay_buffer_size
        Circular buffer capacity in training examples. When full, the oldest
        examples are discarded. Larger buffers improve data diversity but
        may slow down learning if too much old data dilutes recent games.

    min_buffer_size
        Training is skipped until the buffer contains at least this many
        examples. Prevents the network from overfitting to tiny early batches.

    lr_milestones
        Iterations at which the learning rate is multiplied by lr_gamma.
        A stepped decay schedule keeps early training fast and late training
        stable.
    """
    num_iterations:      int        = 50
    num_self_play_games: int        = 50
    num_epochs:          int        = 4
    batch_size:          int        = 128
    lr:                  float      = 1e-3
    weight_decay:        float      = 1e-4
    lr_milestones:       list[int]  = field(default_factory=lambda: [30, 60, 80])
    lr_gamma:            float      = 0.1
    replay_buffer_size:  int        = 100_000
    min_buffer_size:     int        = 1_000
    grad_clip:           float      = 1.0   # max gradient norm; 0 = disabled
    num_workers:         int        = 1     # parallel processes for self-play/eval; 0 = cpu_count-1, 1 = sequential
    value_mix:           float      = 0.0   # 0 = pure game outcome; >0 mixes in MCTS Q-value as soft target


@dataclass
class EvalConfig:
    """
    Model evaluation and promotion parameters.

    promotion_threshold
        The new checkpoint must win at least this fraction of tournament
        games to replace the current best model. Set to 0.5 for "any
        improvement" or higher (e.g. 0.55) to require a statistically
        meaningful gain before promoting.

    elo_k
        ELO K-factor: how much a single game shifts ratings. Higher K
        means ratings react faster but are noisier.

    eval_every_n_iters
        Run evaluation every N training iterations. Setting this > 1 saves
        time when tournament games are expensive.
    """
    tournament_games:    int   = 40
    promotion_threshold: float = 0.55
    elo_k:               float = 32.0
    eval_every_n_iters:  int   = 1
    eval_noise_eps:      float = 0.1   # small Dirichlet noise during eval so games diverge


# ── Top-level run config ──────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """
    Complete configuration for a training run.

    device
        'auto' detects CUDA → MPS → CPU in that order.
        Override with 'cpu', 'cuda', or 'mps' to force a specific device.
        Self-play workers always run on CPU regardless of this setting
        (MCTS is memory-bound, not compute-bound, for small models).

    run_dir
        Root directory for checkpoints, replay files, and status JSON.
        A sub-directory named after `name` is created inside run_dir.

    mlflow_uri
        Where MLflow stores experiment data. Defaults to a local SQLite
        database file ('sqlite:///mlflow.db') — required by recent MLflow
        versions which dropped filesystem-backend support. Point to a remote
        URI (e.g. http://tracking-server:5000) to share runs between machines.

    seed
        Random seed for reproducibility. Covers Python, NumPy, and PyTorch.
    """
    name:               str            = "dev"
    model:              ModelConfig    = field(default_factory=ModelConfig)
    mcts:               MCTSConfig     = field(default_factory=MCTSConfig)
    training:           TrainingConfig = field(default_factory=TrainingConfig)
    eval:               EvalConfig     = field(default_factory=EvalConfig)
    device:             str            = "auto"
    run_dir:            str            = "runs"
    mlflow_uri:         str            = "sqlite:///mlflow.db"
    mlflow_experiment:  str            = ""   # overrides experiment name in UI; defaults to config.name
    mlflow_run_name:    str            = ""   # overrides auto-generated run name; "" = auto
    seed:               int            = 42

    def resolve_device(self) -> torch.device:
        """Return the torch.device to use for this run."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.run_dir, self.name, "checkpoints")

    @property
    def replay_dir(self) -> str:
        return os.path.join(self.run_dir, self.name, "replays")

    @property
    def status_path(self) -> str:
        return os.path.join(self.run_dir, self.name, "status.json")


# ── Presets ───────────────────────────────────────────────────────────────────

DEBUG = RunConfig(
    name="debug",
    model=ModelConfig(num_resblocks=2, num_hidden=16),
    mcts=MCTSConfig(num_simulations=10, temperature_init=1.0, temperature_final=0.0, temp_drop_move=10, resign_threshold=1.0),
    training=TrainingConfig(
        num_iterations=50, num_self_play_games=10, num_epochs=4,
        batch_size=32, replay_buffer_size=500, min_buffer_size=32,
        lr_milestones=[],
    ),
    eval=EvalConfig(tournament_games=50, eval_every_n_iters=1),
)

DEV = RunConfig(
    name="dev",
    model=ModelConfig(num_resblocks=6, num_hidden=64),
    mcts=MCTSConfig(num_simulations=100, temperature_init=1.0, temperature_final=0.0, temp_drop_move=30),
    training=TrainingConfig(
        num_iterations=100, num_self_play_games=50, num_epochs=4,
        batch_size=128, replay_buffer_size=50_000, min_buffer_size=500,
        lr_milestones=[20, 25],
    ),
    eval=EvalConfig(tournament_games=20, eval_every_n_iters=1),
)

MEDIUM = RunConfig(
    name="medium",
    model=ModelConfig(num_resblocks=8, num_hidden=96),
    mcts=MCTSConfig(num_simulations=200, temperature_init=1.0, temperature_final=0.0, temp_drop_move=30,
                    dirichlet_alpha=0.6),
    training=TrainingConfig(
        num_iterations=300, num_self_play_games=50, num_epochs=4,
        batch_size=256, replay_buffer_size=200_000, min_buffer_size=1_000,
        lr_milestones=[160, 240, 280],
        num_workers=1,
        value_mix=0.5,
    ),
    eval=EvalConfig(tournament_games=40, eval_every_n_iters=2),
)

FULL = RunConfig(
    name="full",
    model=ModelConfig(num_resblocks=12, num_hidden=128),
    mcts=MCTSConfig(num_simulations=200, temperature_init=1.0, temperature_final=0.0, temp_drop_move=30),
    training=TrainingConfig(
        num_iterations=100, num_self_play_games=100, num_epochs=4,
        batch_size=256, replay_buffer_size=500_000, min_buffer_size=2_000,
        lr_milestones=[60, 80, 90],
    ),
    eval=EvalConfig(tournament_games=40, eval_every_n_iters=1),
)

PRESETS: dict[str, RunConfig] = {"debug": DEBUG, "dev": DEV, "medium": MEDIUM, "full": FULL}
