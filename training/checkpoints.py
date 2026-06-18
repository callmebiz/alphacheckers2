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
  promotion_count    — cumulative number of successful promotions so far
  config             — the RunConfig dict so the checkpoint is self-describing
  rng_states         — Python, NumPy, and PyTorch RNG states for exact replay

Device portability
------------------
Weights are always saved mapped to CPU (torch.device('cpu')). This means a
checkpoint saved on an RTX 3070 can be loaded on a CPU-only laptop without
any extra flags. The trainer moves the model to the target device after loading.

Naming convention
-----------------
  checkpoint_{iteration}.pt        — full snapshot every N iterations (pruned to keep=1)
  checkpoint_latest.pt             — copy of the most recent full snapshot
  checkpoint_best.pt               — copy of the best model so far (on promotion)
  checkpoint_eval_{iter:04d}.pt    — lightweight snapshot (model + config only) saved at
                                     every eval step; pruned locally to keep=5; all
                                     versions stored in S3 for later download
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
    promotion_count: int,
    config: RunConfig,
    mlflow_run_id: str = "",
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
        "promotion_count": promotion_count,
        "config":          dataclasses.asdict(config),
        "rng_states": {
            "python":  random.getstate(),
            "numpy":   np.random.get_state(),
            "torch":   torch.get_rng_state(),
            "cuda":    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    # Write to a temp file then rename so a mid-write SIGKILL never leaves a
    # partial checkpoint at the real path (os.replace is POSIX-atomic).
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)

    # Small sidecar so the UI can show metadata without loading the full checkpoint
    sidecar = path.replace(".pt", ".json")
    with open(sidecar, "w") as f:
        json.dump({
            "iteration":      iteration,
            "promotions":     promotion_count,
            "buffer":         len(buffer),
            "saved_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mlflow_run_id":  mlflow_run_id,
        }, f)


def load(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> tuple[int, ReplayBuffer, int]:
    """
    Restore training state from *path* and return (iteration, buffer, promotion_count).

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

    buffer          = ReplayBuffer.from_state_dict(payload["replay_buffer"])
    promotion_count = payload.get("promotion_count", 0)
    iteration       = payload["iteration"]

    return iteration, buffer, promotion_count


def _atomic_copy(src: str, dst: str) -> None:
    """Copy src → dst atomically (via tmp + rename) to survive mid-write SIGKILL."""
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def save_best(src_path: str, checkpoint_dir: str) -> str:
    """
    Copy *src_path* to checkpoint_best.pt (+ sidecar .json) in the same directory.
    Returns the path of the best checkpoint file.
    """
    best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    _atomic_copy(src_path, best_path)
    src_json = src_path.replace(".pt", ".json")
    if os.path.exists(src_json):
        _atomic_copy(src_json, best_path.replace(".pt", ".json"))
    return best_path


def save_latest(src_path: str, checkpoint_dir: str) -> str:
    """
    Copy *src_path* to checkpoint_latest.pt (+ sidecar .json) in the same directory.
    Returns the path of the latest checkpoint file.
    """
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    _atomic_copy(src_path, latest_path)
    src_json = src_path.replace(".pt", ".json")
    if os.path.exists(src_json):
        _atomic_copy(src_json, latest_path.replace(".pt", ".json"))
    return latest_path


def save_eval_snapshot(
    path: str,
    model: torch.nn.Module,
    config: RunConfig,
    iteration: int,
    promotion_count: int,
) -> None:
    """
    Save a lightweight checkpoint containing only model weights + config.

    Omits the replay buffer, optimizer, scheduler, and RNG states — typically
    10-50x smaller than a full checkpoint. Sufficient for playing against the
    model in the UI or for post-hoc analysis. Not suitable for resuming training.
    """
    import dataclasses as _dc
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state":     {k: v.cpu() for k, v in model.state_dict().items()},
        "config":          _dc.asdict(config),
        "iteration":       iteration,
        "promotion_count": promotion_count,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)

    sidecar = path.replace(".pt", ".json")
    with open(sidecar, "w") as f:
        json.dump({
            "iteration":      iteration,
            "promotions":     promotion_count,
            "saved_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "snapshot_type":  "eval",
        }, f)


def prune_eval_snapshots(checkpoint_dir: str, keep: int = 5) -> None:
    """Delete old eval snapshots, keeping the most recent *keep* on disk."""
    pattern = os.path.join(checkpoint_dir, "checkpoint_eval_*.pt")
    files   = sorted(glob.glob(pattern))   # lexicographic = chronological with zero-padded names
    for old in files[:-keep]:
        for path in (old, old.replace(".pt", ".json")):
            try:
                os.remove(path)
            except OSError:
                pass


def prune_old_checkpoints(checkpoint_dir: str, keep: int = 1) -> None:
    """
    Delete numbered checkpoints beyond the most recent *keep*, preserving
    checkpoint_best.pt, checkpoint_latest.pt, and checkpoint_eval_*.pt.
    """
    # Clean up any leftover temp files from interrupted atomic writes
    for tmp in glob.glob(os.path.join(checkpoint_dir, "*.tmp")):
        try:
            os.remove(tmp)
        except OSError:
            pass

    _skip = {"checkpoint_best.pt", "checkpoint_latest.pt"}
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    files = sorted(
        [f for f in glob.glob(pattern)
         if os.path.basename(f) not in _skip
         and not os.path.basename(f).startswith("checkpoint_eval_")],
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
    _skip   = {"checkpoint_best.pt", "checkpoint_latest.pt"}
    files   = [
        f for f in glob.glob(pattern)
        if os.path.basename(f) not in _skip
        and not os.path.basename(f).startswith("checkpoint_eval_")
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
