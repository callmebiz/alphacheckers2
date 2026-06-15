"""
Checkpointing
=============
Saves and restores the complete training state so a run can be paused on
one machine and resumed on any other — including a different device.

What gets saved
---------------
A checkpoint is a single .pt file containing:
  model_state        — network weights
  optimizer_state    — Adam momentum + variance (critical for smooth resume)
  scheduler_state    — LR schedule position
  iteration          — which iteration we just finished
  replay_buffer      — all accumulated training examples
  elo_ratings        — current ELO ratings for all checkpoints
  config             — the RunConfig dict so the checkpoint is self-describing
  rng_states         — Python, NumPy, and PyTorch RNG states for exact replay

Device portability
------------------
Weights are always saved mapped to CPU (torch.device('cpu')). This means a
checkpoint saved on an RTX 3070 can be loaded on a CPU-only laptop without
any extra flags. The trainer moves the model to the target device after loading.

Naming convention
-----------------
  checkpoint_{iteration}.pt   — snapshot every N iterations
  checkpoint_best.pt          — the current best model (symlink-style copy)
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import random
import shutil
import time

import numpy as np
import torch

from training.config import RunConfig
from training.replay_buffer import ReplayBuffer


def save(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    iteration: int,
    buffer: ReplayBuffer,
    elo_ratings: dict[str, float],
    config: RunConfig,
) -> None:
    """
    Write a full training checkpoint to *path*.

    The model weights are detached to CPU before saving so the file can be
    loaded on any device. All other state is already CPU-resident.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}

    payload = {
        "model_state":     cpu_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "iteration":       iteration,
        "replay_buffer":   buffer.state_dict(),
        "elo_ratings":     elo_ratings,
        "config":          dataclasses.asdict(config),
        "rng_states": {
            "python":  random.getstate(),
            "numpy":   np.random.get_state(),
            "torch":   torch.get_rng_state(),
            "cuda":    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    torch.save(payload, path)

    # Small sidecar so the UI can show metadata without loading the full checkpoint
    sidecar = path.replace(".pt", ".json")
    with open(sidecar, "w") as f:
        json.dump({
            "iteration": iteration,
            "elo":       elo_ratings.get("best", 0.0),
            "buffer":    len(buffer),
            "saved_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f)


def load(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> tuple[int, ReplayBuffer, dict[str, float]]:
    """
    Restore training state from *path* and return (iteration, buffer, elo_ratings).

    The model is loaded to CPU first, then moved to *device*. This ensures
    cross-device compatibility regardless of where the checkpoint was created.

    Parameters
    ----------
    path      : Path to the .pt checkpoint file.
    model     : Uninitialised model with matching architecture.
    optimizer : Optimiser bound to model.parameters().
    scheduler : LR scheduler bound to the optimiser.
    device    : Target device for the model after loading.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(payload["model_state"])
    model.to(device)

    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])

    # Restore RNG states for reproducible resume
    rng = payload.get("rng_states", {})
    if rng.get("python"):
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])

    buffer      = ReplayBuffer.from_state_dict(payload["replay_buffer"])
    elo_ratings = payload.get("elo_ratings", {})
    iteration   = payload["iteration"]

    return iteration, buffer, elo_ratings


def save_best(src_path: str, checkpoint_dir: str) -> str:
    """
    Copy *src_path* to checkpoint_best.pt (+ sidecar .json) in the same directory.
    Returns the path of the best checkpoint file.
    """
    best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    shutil.copy2(src_path, best_path)
    src_json = src_path.replace(".pt", ".json")
    if os.path.exists(src_json):
        shutil.copy2(src_json, best_path.replace(".pt", ".json"))
    return best_path


def save_latest(src_path: str, checkpoint_dir: str) -> str:
    """
    Copy *src_path* to checkpoint_latest.pt (+ sidecar .json) in the same directory.
    Returns the path of the latest checkpoint file.
    """
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    shutil.copy2(src_path, latest_path)
    src_json = src_path.replace(".pt", ".json")
    if os.path.exists(src_json):
        shutil.copy2(src_json, latest_path.replace(".pt", ".json"))
    return latest_path


def prune_old_checkpoints(checkpoint_dir: str, keep: int = 3) -> None:
    """
    Delete numbered checkpoints beyond the most recent *keep*, preserving
    checkpoint_best.pt and checkpoint_latest.pt. Prevents disk exhaustion on long runs.
    """
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    files = sorted(
        [f for f in glob.glob(pattern)
         if not f.endswith("checkpoint_best.pt")
         and not f.endswith("checkpoint_latest.pt")],
        key=lambda p: int(
            os.path.basename(p).replace("checkpoint_", "").replace(".pt", "") or -1
        ),
    )
    for old in files[:-keep]:
        for path in (old, old.replace(".pt", ".json")):
            try:
                os.remove(path)
            except OSError:
                pass


def find_latest(checkpoint_dir: str) -> str | None:
    """
    Return the path of the most recent numbered checkpoint, or None if none exist.

    Scans for files matching 'checkpoint_*.pt' (excluding checkpoint_best.pt)
    and returns the one with the highest iteration number.
    """
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    files   = [
        f for f in glob.glob(pattern)
        if not f.endswith("checkpoint_best.pt")
    ]
    if not files:
        return None

    def _iteration(path: str) -> int:
        stem = os.path.basename(path).replace("checkpoint_", "").replace(".pt", "")
        try:
            return int(stem)
        except ValueError:
            return -1

    return max(files, key=_iteration)
