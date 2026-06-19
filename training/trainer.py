"""
Trainer
=======
Orchestrates the full AlphaZero training loop — the engine that ties every
other module together. One iteration of the loop does four things:

  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. SELF-PLAY    Generate games with the current best model + MCTS.  │
  │                 Add (state, policy, outcome) examples to the buffer. │
  │                                                                     │
  │ 2. TRAIN        Sample mini-batches from the buffer and update the   │
  │                 challenger network's weights.                        │
  │                                                                     │
  │ 3. EVALUATE     Run a tournament: challenger vs current best.        │
  │                 Promote the challenger if it wins ≥ threshold.       │
  │                                                                     │
  │ 4. LOG + SAVE   Write all metrics to MLflow, checkpoint to disk,     │
  │                 update the status.json the UI reads.                │
  └─────────────────────────────────────────────────────────────────────┘

Two learning modes (controlled by config.eval.gated):

  Gated (default, AlphaGo Zero style)
    Two model instances: best_model for self-play, model (challenger) for
    training. A tournament runs every eval_every_n_iters iterations; the
    challenger replaces best_model only if it wins >= promotion_threshold.
    Otherwise the challenger is reset to best_model's weights. Protects
    against regression at the cost of slower effective update rate.

  Continuous (hybrid style)
    Self-play always uses the latest model weights. A tournament still
    runs every eval_every_n_iters for benchmarking, and the best_model
    snapshot advances on promotion, but non-promotion never resets the
    challenger. Faster iteration throughput with no regression guard.

Mixed precision (AMP)
  On CUDA, torch.amp.autocast halves memory usage and speeds up training
  significantly with negligible accuracy loss. It is automatically disabled
  on CPU/MPS where the benefit is absent or unsupported.

Graceful shutdown
  The trainer catches SIGINT (Ctrl+C) and SIGTERM and saves a checkpoint
  before exiting, so a long run can be interrupted without losing progress.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import signal
import time
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Suppress mlflow's INFO chatter — errors and warnings still surface
logging.getLogger("mlflow").setLevel(logging.WARNING)

from core.game import Checkers
from core.encoder import StateEncoder
from core.model import AlphaNet
from training.config import RunConfig
from training.replay_buffer import ReplayBuffer
from training.self_play import generate_games
from training.evaluator import run_tournament
from training.analysis import (
    compute_opening_entropy,
    compute_policy_entropy,
)
from training import checkpoints


def _warn_low_disk(path: str, min_gb: float = 5.0) -> None:
    free_gb = shutil.disk_usage(path).free / 1024**3
    if free_gb < min_gb:
        print(f"WARNING: only {free_gb:.1f} GB free — checkpoints may fail if disk fills up.")


class Trainer:
    """
    Manages the full AlphaZero training loop for one RunConfig.

    Parameters
    ----------
    config : RunConfig defining all hyperparameters and paths.
    """

    def __init__(self, config: RunConfig, s3_bucket: str = "", mlflow_run_id: str = ""):
        self.config  = config
        self.device  = config.resolve_device()
        self.game    = Checkers()
        self.encoder = StateEncoder(self.game)

        tc = config.training
        mc = config.model

        # Challenger — the model being trained this iteration
        self.model = AlphaNet(
            num_channels=self.encoder.num_channels,
            action_size=self.game.action_size,
            num_resblocks=mc.num_resblocks,
            num_hidden=mc.num_hidden,
        ).to(self.device)

        # best_model is a frozen snapshot kept as the tournament opponent.
        # In gated mode it also drives self-play; in continuous mode self-play
        # uses self.model (latest) but the tournament still needs a snapshot.
        self.best_model = copy.deepcopy(self.model)
        self.best_model.eval()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=tc.lr_milestones, gamma=tc.lr_gamma
        )
        self.buffer    = ReplayBuffer(
            tc.replay_buffer_size,
            state_shape=(self.encoder.num_channels, self.game.row_count, self.game.col_count),
            policy_size=self.game.action_size,
        )

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.replay_dir,     exist_ok=True)

        self._mlflow_run_id = mlflow_run_id
        self._promotion_count: int = 0

        # AMP scaler — only active on CUDA
        self._use_amp  = (self.device.type == "cuda")
        self._scaler   = torch.cuda.amp.GradScaler() if self._use_amp else None

        # Persistent process pool for parallel self-play and tournament games.
        # Kept alive across iterations so worker processes stay warm (Windows
        # spawn mode has ~1-2 s startup cost per pool creation).
        # num_workers=1 → sequential (no pool); 0 → auto (cpu_count - 1).
        n_workers = config.training.num_workers
        if n_workers == 0:
            n_workers = max(1, (os.cpu_count() or 2) - 1)
        # Use spawn context on all platforms to avoid fork-safety deadlocks
        # when torch/numpy internal threads hold locks at fork time.
        self._executor = (
            ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn"))
            if n_workers > 1 else None
        )

        self._s3_bucket = s3_bucket
        self._run_segment: int = 0       # incremented each time training resumes from checkpoint
        self._iter_times: list[float] = []  # rolling window for ETA calculation
        self._shutdown = False
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def train(self, resume_from: str | None = None) -> None:
        """
        Run the full training loop, optionally resuming from a checkpoint.

        Parameters
        ----------
        resume_from : Path to a .pt checkpoint file. If None, starts fresh.
        """
        _warn_low_disk(self.config.checkpoint_dir)

        start_iter = 0
        if resume_from:
            start_iter, self.buffer, self._promotion_count, prev_segment = checkpoints.load(
                resume_from, self.model, self.optimizer, self.scheduler, self.device
            )
            self._run_segment = prev_segment + 1
            self.best_model.load_state_dict(self.model.state_dict())
            n_buf   = len(self.buffer)
            free_gb = shutil.disk_usage(self.config.checkpoint_dir).free / 1024**3
            print(
                f"Resumed iter {start_iter} | segment {self._run_segment} | "
                f"promotions {self._promotion_count} | buffer {n_buf:,} | disk free {free_gb:.1f} GB"
            )

        config = self.config
        tc     = config.training
        total_iters = tc.num_iterations

        outer = tqdm(
            range(start_iter + 1 if resume_from else start_iter, total_iters),
            desc=f"{config.name}",
            unit="iter",
            dynamic_ncols=True,
        )

        stop_fn = lambda: self._shutdown

        # Local import: keeps mlflow/pandas out of the spawn chain in worker processes.
        from training.tracking import MLflowTracker

        with MLflowTracker(
            config, self.device,
            experiment=config.mlflow_experiment or None,
            run_name=config.mlflow_run_name or None,
            run_id=self._mlflow_run_id or None,
        ) as tracker:
            for iteration in outer:
                if self._shutdown:
                    break

                t0 = time.time()

                # ── Step 1: Self-play ──────────────────────────────────────
                outer.set_description(f"{config.name} | self-play")
                t0_sp = time.time()
                sp_model = self.best_model if config.eval.gated else self.model
                examples, sp_stats = generate_games(
                    n_games=tc.num_self_play_games,
                    game=self.game,
                    encoder=self.encoder,
                    model=sp_model,
                    config=config,
                    device=self.device,
                    iteration=iteration,
                    stop_fn=stop_fn,
                    mlflow_run_name=tracker.run_name,
                    executor=self._executor,
                )
                self.buffer.add_many(examples)
                t_sp = time.time() - t0_sp

                if self._shutdown:
                    break  # stopped mid self-play; skip train/eval/log for this iteration

                sp_policies      = [e[1] for e in examples]
                sp_first_actions = [int(np.argmax(e[1])) for e in examples]

                # ── Step 2: Train network ──────────────────────────────────
                trained = False
                policy_loss = value_loss = value_mae = 0.0
                t_train = 0.0
                if len(self.buffer) >= tc.min_buffer_size:
                    outer.set_description(f"{config.name} | training")
                    t0_train = time.time()
                    policy_loss, value_loss, value_mae = self._train_epochs()
                    t_train = time.time() - t0_train
                    trained = True
                    self.scheduler.step()

                if self._shutdown:
                    break

                # ── Step 3: Evaluate ───────────────────────────────────────
                promoted = False
                result   = None
                t_eval   = 0.0

                if (iteration + 1) % config.eval.eval_every_n_iters == 0:
                    outer.set_description(f"{config.name} | eval")
                    t0_eval = time.time()
                    result = run_tournament(
                        self.model, self.best_model,
                        self.game, self.encoder, config, self.device,
                        stop_fn=stop_fn,
                        executor=self._executor,
                        iteration=iteration,
                        mlflow_run_name=tracker.run_name,
                    )
                    t_eval = time.time() - t0_eval

                    if self._shutdown:
                        break  # discard partial tournament; don't promote on incomplete results

                    promoted = result.win_rate >= config.eval.promotion_threshold
                    if promoted:
                        self._promotion_count += 1
                        self.best_model.load_state_dict(self.model.state_dict())
                        self.best_model.eval()
                        tqdm.write(
                            f"  ✓ iter {iteration+1}: promoted "
                            f"(win rate {result.win_rate:.0%}, #{self._promotion_count})"
                        )
                    elif config.eval.gated:
                        self.model.load_state_dict(self.best_model.state_dict())

                # ── Step 4: Log + checkpoint ───────────────────────────────
                outer.set_description(f"{config.name} | logging")
                lr = self.optimizer.param_groups[0]["lr"]

                if trained:
                    tracker.log_training(
                        step=iteration,
                        policy_loss=policy_loss,
                        value_loss=value_loss,
                        lr=lr,
                        buffer_size=len(self.buffer),
                    )
                tracker.log_selfplay(
                    step=iteration,
                    avg_game_length=sp_stats.avg_game_length,
                    draw_rate=sp_stats.draw_rate,
                    p1_win_rate=sp_stats.p1_win_rate,
                    policy_entropy=compute_policy_entropy(sp_policies),
                    move_entropy_mean=sp_stats.move_entropy_mean,
                    move_entropy_min=sp_stats.move_entropy_min,
                    move_entropy_std=sp_stats.move_entropy_std,
                    top1_prob_mean=sp_stats.top1_prob_mean,
                )
                if result is not None:
                    tracker.log_evaluation(
                        step=iteration,
                        result=result,
                        promotion_count=self._promotion_count,
                        value_mae=value_mae,
                        opening_entropy=compute_opening_entropy(
                            sp_first_actions, self.game.action_size
                        ),
                    )

                ckpt_path = os.path.join(
                    config.checkpoint_dir, f"checkpoint_{iteration}.pt"
                )
                checkpoints.save(
                    ckpt_path, self.best_model, self.optimizer, self.scheduler,
                    iteration, self.buffer, self._promotion_count, config,
                    mlflow_run_id=tracker.run_id,
                    run_segment=self._run_segment,
                )
                latest_path = checkpoints.save_latest(ckpt_path, config.checkpoint_dir)
                self._upload_to_s3(latest_path, "checkpoint_latest")
                self._upload_mlflow_db()
                if promoted:
                    best_path = os.path.join(config.checkpoint_dir, "checkpoint_best.pt")
                    checkpoints.save_eval_snapshot(
                        best_path, self.best_model, config, iteration, self._promotion_count
                    )
                    tracker.log_model_artifact(ckpt_path, iteration)
                    self._upload_to_s3(best_path, "checkpoint_best")
                    self._upload_versioned_best_to_s3(best_path, iteration)
                checkpoints.prune_old_checkpoints(config.checkpoint_dir, keep=1)

                if result is not None:
                    eval_stem = f"checkpoint_eval_{iteration:04d}"
                    eval_path = os.path.join(config.checkpoint_dir, f"{eval_stem}.pt")
                    checkpoints.save_eval_snapshot(
                        eval_path, self.model, config, iteration, self._promotion_count
                    )
                    self._upload_to_s3(eval_path, eval_stem)
                    checkpoints.prune_eval_snapshots(config.checkpoint_dir, keep=5)

                checkpoints.prune_old_replays(config.replay_dir, keep_iters=10)
                self._write_status(iteration, policy_loss, value_loss)

                # Update outer bar — one summary line per iteration
                elapsed = time.time() - t0
                self._iter_times.append(elapsed)
                if len(self._iter_times) > 10:
                    self._iter_times.pop(0)
                avg_iter = sum(self._iter_times) / len(self._iter_times)
                eta_hours = avg_iter * (total_iters - (iteration + 1)) / 3600

                free_gb = shutil.disk_usage(config.checkpoint_dir).free / 1024**3
                tracker.log_system(
                    step=iteration,
                    iter_time_seconds=elapsed,
                    disk_free_gb=free_gb,
                    selfplay_time_s=t_sp,
                    train_time_s=t_train,
                    eval_time_s=t_eval,
                    eta_hours=eta_hours,
                    run_segment=self._run_segment,
                )
                pf: dict = dict(
                    p=f"{policy_loss:.3f}",
                    v=f"{value_loss:.3f}",
                    t=f"{elapsed:.0f}s",
                    eta=f"{eta_hours:.1f}h",
                    seg=str(self._run_segment),
                    promo=str(self._promotion_count),
                )
                if result is not None:
                    pf["eval"] = "✓" if promoted else f"✗{result.win_rate:.0%}"
                outer.set_postfix(pf)
                outer.set_description(config.name)

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

        # Mark training stopped in status file so the UI updates immediately
        try:
            if os.path.exists(config.status_path):
                with open(config.status_path) as f:
                    s = json.load(f)
                s["is_training"] = False
                with open(config.status_path, "w") as f:
                    json.dump(s, f)
        except (OSError, json.JSONDecodeError):
            pass

        tqdm.write("\nTraining complete.")

    # ── Training step ─────────────────────────────────────────────────────────

    def _train_epochs(self) -> tuple[float, float, float]:
        """
        Run `num_epochs` passes through random mini-batches from the buffer.

        Returns (mean_policy_loss, mean_value_loss, value_mae) where value_mae
        is computed in eval mode on a fresh buffer sample after training — an
        unbiased estimate of the value head's absolute calibration error.
        """
        tc = self.config.training
        self.model.train()

        total_p_loss = total_v_loss = 0.0
        num_batches  = 0

        for _ in tqdm(range(tc.num_epochs), desc="  epochs", unit="ep", leave=False):
            states, policies, values = self.buffer.sample(
                min(tc.batch_size, len(self.buffer))
            )
            states   = states.to(self.device)
            policies = policies.to(self.device)
            values   = values.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self._use_amp):
                policy_logits, value_pred = self.model(states)

                # Policy: cross-entropy vs MCTS distribution
                policy_loss = F.cross_entropy(policy_logits, policies)

                # Value: MSE vs actual game outcome
                value_loss = F.mse_loss(value_pred, values)

                loss = policy_loss + value_loss

            if self._use_amp:
                self._scaler.scale(loss).backward()
                if tc.grad_clip > 0:
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                if tc.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
                self.optimizer.step()

            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()
            num_batches  += 1

        # Eval-mode value MAE on a fresh sample (no gradient, no dropout)
        self.model.eval()
        with torch.no_grad():
            val_states, _, val_values = self.buffer.sample(
                min(tc.batch_size, len(self.buffer))
            )
            val_states  = val_states.to(self.device)
            val_values  = val_values.to(self.device)
            _, val_pred = self.model(val_states)
            value_mae   = float(F.l1_loss(val_pred, val_values).item())

        return total_p_loss / num_batches, total_v_loss / num_batches, value_mae

    # ── Status file ───────────────────────────────────────────────────────────

    def _write_status(
        self,
        iteration:   int,
        policy_loss: float,
        value_loss:  float,
    ) -> None:
        """
        Write a JSON status file that the web UI polls for live training info.
        Located at config.status_path.
        """
        tc = self.config.training
        status = {
            "is_training":  not self._shutdown,
            "iteration":    iteration + 1,
            "total":        tc.num_iterations,
            "phase":        "complete" if iteration + 1 == tc.num_iterations else "training",
            "metrics": {
                "policy_loss": round(policy_loss, 4),
                "value_loss":  round(value_loss, 4),
                "promotions":  self._promotion_count,
            },
        }
        os.makedirs(os.path.dirname(self.config.status_path), exist_ok=True)
        with open(self.config.status_path, "w") as f:
            json.dump(status, f)

    # ── S3 upload ─────────────────────────────────────────────────────────────

    def _upload_to_s3(self, local_pt: str, dest_stem: str) -> None:
        """Upload a checkpoint .pt + .json to S3 under dest_stem."""
        if not self._s3_bucket:
            return
        import subprocess
        base = f"s3://{self._s3_bucket}/runs/{self.config.name}/checkpoints"
        failed = False
        for ext in (".pt", ".json"):
            local = local_pt.replace(".pt", ext)
            if not os.path.exists(local):
                continue
            dest = f"{base}/{dest_stem}{ext}"
            r = subprocess.run(["aws", "s3", "cp", local, dest],
                               capture_output=True, text=True)
            if r.returncode != 0:
                tqdm.write(f"  S3 upload failed ({dest_stem}{ext}): {r.stderr.strip()}")
                failed = True
        if not failed:
            tqdm.write(f"  ✓ S3: {base}/{dest_stem}.pt")

    def _upload_mlflow_db(self) -> None:
        """Upload mlflow.db to S3 namespaced by config.name so concurrent experiments don't overwrite each other."""
        if not self._s3_bucket:
            return
        import subprocess
        db = self.config.mlflow_uri.replace("sqlite:///", "")
        if not os.path.exists(db):
            return
        dest = f"s3://{self._s3_bucket}/mlflow-{self.config.name}.db"
        r = subprocess.run(["aws", "s3", "cp", db, dest],
                           capture_output=True, text=True)
        if r.returncode != 0:
            tqdm.write(f"  S3 mlflow.db upload failed: {r.stderr.strip()}")

    def _upload_versioned_best_to_s3(self, best_path: str, iteration: int) -> None:
        """Archive a versioned copy of the best checkpoint so prior bests aren't overwritten."""
        if not self._s3_bucket:
            return
        import subprocess
        base = f"s3://{self._s3_bucket}/runs/{self.config.name}/checkpoints/versions"
        stem = f"checkpoint_best_iter{iteration:04d}"
        failed = False
        for ext in (".pt", ".json"):
            local = best_path.replace(".pt", ext)
            if not os.path.exists(local):
                continue
            r = subprocess.run(["aws", "s3", "cp", local, f"{base}/{stem}{ext}"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                tqdm.write(f"  S3 versioned upload failed ({stem}{ext}): {r.stderr.strip()}")
                failed = True
        if not failed:
            tqdm.write(f"  ✓ S3 versioned: {base}/{stem}.pt")

    # ── Signal handling ───────────────────────────────────────────────────────

    def _handle_signal(self, signum, *_) -> None:
        """
        SIGINT (Ctrl+C) or SIGTERM (spot termination): stop after the current
        game so a clean checkpoint is written before exit.
        Second signal: force-quit immediately.
        """
        if self._shutdown:
            tqdm.write("\nForce quit.")
            os._exit(1)
        if signum == signal.SIGTERM:
            tqdm.write("\nSpot termination notice — stopping after current game.")
        else:
            tqdm.write("\nCtrl+C — stopping after current game. (Ctrl+C again to force quit.)")
        self._shutdown = True
