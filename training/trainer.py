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

Two model instances are maintained:
  best_model   — the current gold standard; used for self-play and as the
                 bar the challenger must beat. Never trained directly.
  model        — the challenger being trained. Gets a fresh copy of
                 best_model's weights each iteration as a starting point.

Why not just train the model in-place?
  If the model gets worse during a bad training step (unlucky batch, sharp
  gradient), we want the safety net of the last best_model. The promotion
  gate ensures the 'best' label only moves forward.

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
from training.elo import EloTracker
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

        # Best model — never trained; only updated on successful promotion
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

        elo_path  = os.path.join(config.run_dir, config.name, "elo.json")
        self.elo  = EloTracker(elo_path, k=config.eval.elo_k)

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
            start_iter, self.buffer, elo_ratings = checkpoints.load(
                resume_from, self.model, self.optimizer, self.scheduler, self.device
            )
            self.elo._ratings = elo_ratings
            self.best_model.load_state_dict(self.model.state_dict())
            cur_elo   = elo_ratings.get("best", 0.0)
            n_buf     = len(self.buffer)
            free_gb   = shutil.disk_usage(self.config.checkpoint_dir).free / 1024**3
            print(
                f"Resumed iter {start_iter} | ELO {cur_elo:.0f} | "
                f"buffer {n_buf:,} | disk free {free_gb:.1f} GB"
            )
        else:
            # Fresh run — discard any elo.json left over from a previous run.
            # EloTracker always loads from disk on init, which would carry stale
            # ratings into the new run and produce misleading starting ELO.
            self.elo._ratings = {}

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
                examples, sp_stats = generate_games(
                    n_games=tc.num_self_play_games,
                    game=self.game,
                    encoder=self.encoder,
                    model=self.best_model,
                    config=config,
                    device=self.device,
                    iteration=iteration,
                    stop_fn=stop_fn,
                    mlflow_run_name=tracker.run_name,
                    executor=self._executor,
                )
                self.buffer.add_many(examples)

                if self._shutdown:
                    break  # stopped mid self-play; skip train/eval/log for this iteration

                sp_policies      = [e[1] for e in examples]
                sp_first_actions = [int(np.argmax(e[1])) for e in examples]

                # ── Step 2: Train network ──────────────────────────────────
                trained = False
                policy_loss = value_loss = value_mae = 0.0
                if len(self.buffer) >= tc.min_buffer_size:
                    outer.set_description(f"{config.name} | training")
                    policy_loss, value_loss, value_mae = self._train_epochs()
                    trained = True
                    self.scheduler.step()

                if self._shutdown:
                    break

                # ── Step 3: Evaluate ───────────────────────────────────────
                promoted = False
                result   = None
                cur_elo  = self.elo.rating("best")

                if (iteration + 1) % config.eval.eval_every_n_iters == 0:
                    outer.set_description(f"{config.name} | eval")
                    result = run_tournament(
                        self.model, self.best_model,
                        self.game, self.encoder, config, self.device,
                        stop_fn=stop_fn,
                        executor=self._executor,
                        iteration=iteration,
                        mlflow_run_name=tracker.run_name,
                    )

                    if self._shutdown:
                        break  # discard partial tournament; don't promote on incomplete results

                    # Seed the challenger's ELO at best's current rating so any
                    # win rate > 50% (which promotion requires) strictly increases
                    # the "best" ELO on promotion, rather than comparing against
                    # the always-1000 default for a new key.
                    self.elo._ratings[f"iter_{iteration}"] = self.elo.rating("best")
                    self.elo.update_from_results(
                        f"iter_{iteration}", "best",
                        result.wins, result.draws, result.losses,
                    )

                    promoted = result.win_rate >= config.eval.promotion_threshold
                    if promoted:
                        self.best_model.load_state_dict(self.model.state_dict())
                        self.best_model.eval()
                        # The old "best" just took the tournament loss — its ELO
                        # went down.  Transfer the *winner's* (challenger's) ELO
                        # to the "best" slot so the displayed rating tracks the
                        # actual strength of the current champion.
                        self.elo._ratings["best"] = self.elo.rating(f"iter_{iteration}")
                    else:
                        self.model.load_state_dict(self.best_model.state_dict())


                    self.elo.save()
                    cur_elo = self.elo.rating("best")

                    if promoted:
                        tqdm.write(
                            f"  ✓ iter {iteration+1}: promoted "
                            f"(win rate {result.win_rate:.0%}, ELO {cur_elo:.0f})"
                        )

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
                        elo=cur_elo,
                        promoted=promoted,
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
                    iteration, self.buffer, self.elo.all_ratings(), config,
                    mlflow_run_id=tracker.run_id,
                )
                latest_path = checkpoints.save_latest(ckpt_path, config.checkpoint_dir)
                self._upload_to_s3(latest_path, "checkpoint_latest")
                if promoted:
                    best_path = checkpoints.save_best(ckpt_path, config.checkpoint_dir)
                    tracker.log_model_artifact(ckpt_path, iteration)
                    self._upload_to_s3(best_path, "checkpoint_best")
                checkpoints.prune_old_checkpoints(config.checkpoint_dir, keep=1)

                self._write_status(iteration, policy_loss, value_loss, cur_elo, promoted)

                # Update outer bar — one summary line per iteration
                elapsed = time.time() - t0
                free_gb = shutil.disk_usage(config.checkpoint_dir).free / 1024**3
                tracker.log_system(step=iteration, iter_time_seconds=elapsed, disk_free_gb=free_gb)
                pf: dict = dict(
                    elo=f"{cur_elo:.0f}",
                    p=f"{policy_loss:.3f}",
                    v=f"{value_loss:.3f}",
                    t=f"{elapsed:.0f}s",
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
        elo:         float,
        promoted:    bool,
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
                "elo":         round(elo, 1),
                "promoted":    promoted,
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
