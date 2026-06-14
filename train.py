"""
Training Entry Point
====================
Run AlphaZero self-play training from the command line.

Usage
-----
# Start a debug run (fast, CPU-friendly):
    python train.py --config debug

# Start a full development run:
    python train.py --config dev

# Resume from the latest checkpoint:
    python train.py --config dev --resume

# Resume from a specific checkpoint:
    python train.py --config dev --resume runs/dev/checkpoints/checkpoint_5.pt

# Override device (useful when moving between laptop and desktop):
    python train.py --config full --device cuda

# Start the MLflow UI (separate terminal, same directory):
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    # then open http://localhost:5000

Monitoring during training
--------------------------
The trainer writes runs/{config_name}/status.json after every iteration.
The web UI reads this file and shows live training progress — start the
UI server in a separate terminal while training is running:
    python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

import argparse

import numpy as np
import torch

from training.config import PRESETS, RunConfig
from training.checkpoints import find_latest
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaCheckers training")
    p.add_argument(
        "--config", choices=list(PRESETS), default="dev",
        help="Training preset: debug | dev | full  (default: dev)",
    )
    p.add_argument(
        "--resume", nargs="?", const="latest", default=None,
        metavar="CHECKPOINT_PATH",
        help="Resume training. Pass a path or omit path to auto-find latest.",
    )
    p.add_argument(
        "--device", default=None,
        help="Override device: cpu | cuda | mps  (default: auto-detect)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed override (default: from config)",
    )
    return p.parse_args()


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args   = parse_args()
    config: RunConfig = PRESETS[args.config]

    # Apply CLI overrides
    if args.device:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed

    seed_everything(config.seed)

    device = config.resolve_device()
    print(f"AlphaCheckers — config: {config.name} | device: {device}")

    # Resolve resume path
    resume_path = None
    if args.resume == "latest":
        resume_path = find_latest(config.checkpoint_dir)
        if resume_path:
            print(f"Auto-resuming from: {resume_path}")
        else:
            print("No checkpoint found — starting fresh.")
    elif args.resume:
        resume_path = args.resume
        print(f"Resuming from: {resume_path}")

    trainer = Trainer(config)
    trainer.train(resume_from=resume_path)


if __name__ == "__main__":
    main()
