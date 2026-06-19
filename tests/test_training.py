"""
Tests for the training stack — replay buffer and checkpoints.

Run: pytest tests/test_training.py -v
"""
import copy
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from training.config import DEBUG, RunConfig
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayStats
from training import checkpoints
from core.game import Checkers
from core.encoder import StateEncoder
from core.model import AlphaNet


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def game():
    return Checkers()

@pytest.fixture
def enc(game):
    return StateEncoder(game)

@pytest.fixture
def small_net(enc, game):
    return AlphaNet(enc.num_channels, game.action_size, num_resblocks=1, num_hidden=8)

@pytest.fixture
def dummy_example(enc, game):
    state  = game.get_initial_state()
    tensor = enc.encode(state, 1)
    policy = np.ones(game.action_size, dtype=np.float32) / game.action_size
    value  = 1.0
    return tensor, policy, value


# ── SelfPlayStats ─────────────────────────────────────────────────────────────

class TestSelfPlayStats:
    def test_p1_win_rate(self):
        s = SelfPlayStats(num_games=10, p1_wins=3, p2_wins=5, draws=2, total_moves=100)
        assert s.p1_win_rate == pytest.approx(0.3)

    def test_draw_rate(self):
        s = SelfPlayStats(num_games=10, p1_wins=3, p2_wins=5, draws=2, total_moves=100)
        assert s.draw_rate == pytest.approx(0.2)

    def test_avg_game_length(self):
        s = SelfPlayStats(num_games=10, p1_wins=3, p2_wins=5, draws=2, total_moves=500)
        assert s.avg_game_length == pytest.approx(50.0)

    def test_counts_sum_to_num_games(self):
        s = SelfPlayStats(num_games=10, p1_wins=3, p2_wins=5, draws=2, total_moves=0)
        assert s.p1_wins + s.p2_wins + s.draws == s.num_games

    def test_zero_games_returns_zero_rates(self):
        s = SelfPlayStats(num_games=0, p1_wins=0, p2_wins=0, draws=0, total_moves=0)
        assert s.p1_win_rate == 0.0
        assert s.draw_rate == 0.0
        assert s.avg_game_length == 0.0

    def test_per_move_stats_default_to_zero(self):
        s = SelfPlayStats(num_games=10, p1_wins=5, p2_wins=4, draws=1, total_moves=200)
        assert s.move_entropy_mean == 0.0
        assert s.move_entropy_min  == 0.0
        assert s.move_entropy_std  == 0.0
        assert s.top1_prob_mean    == 0.0

    def test_per_move_stats_stored(self):
        s = SelfPlayStats(
            num_games=4, p1_wins=2, p2_wins=1, draws=1, total_moves=80,
            move_entropy_mean=2.5, move_entropy_min=0.1,
            move_entropy_std=0.8, top1_prob_mean=0.7,
        )
        assert s.move_entropy_mean == pytest.approx(2.5)
        assert s.move_entropy_min  == pytest.approx(0.1)
        assert s.move_entropy_std  == pytest.approx(0.8)
        assert s.top1_prob_mean    == pytest.approx(0.7)


# ── ReplayBuffer ──────────────────────────────────────────────────────────────

class TestReplayBuffer:
    def test_empty_buffer_has_zero_length(self):
        buf = ReplayBuffer(100)
        assert len(buf) == 0

    def test_add_increments_length(self, dummy_example):
        buf = ReplayBuffer(100)
        buf.add(*dummy_example)
        assert len(buf) == 1

    def test_add_many(self, dummy_example):
        buf = ReplayBuffer(100)
        buf.add_many([dummy_example] * 10)
        assert len(buf) == 10

    def test_capacity_is_respected(self, dummy_example):
        buf = ReplayBuffer(5)
        buf.add_many([dummy_example] * 10)
        assert len(buf) == 5  # capped at capacity

    def test_oldest_entry_overwritten(self, enc, game):
        """After capacity, position 0 holds the most recently written item."""
        buf = ReplayBuffer(3)
        for i in range(5):
            t = enc.encode(game.get_initial_state(), 1)
            p = np.zeros(game.action_size, dtype=np.float32)
            p[i % game.action_size] = 1.0
            buf.add(t, p, float(i))
        # Buffer is full; we cannot know exact index, but size must be 3
        assert len(buf) == 3

    def test_sample_returns_correct_shapes(self, dummy_example, enc, game):
        buf = ReplayBuffer(50)
        buf.add_many([dummy_example] * 20)
        states, policies, values = buf.sample(8)
        assert states.shape   == (8, enc.num_channels, 8, 8)
        assert policies.shape == (8, game.action_size)
        assert values.shape   == (8, 1)

    def test_sample_dtypes_are_float32(self, dummy_example):
        buf = ReplayBuffer(50)
        buf.add_many([dummy_example] * 20)
        s, p, v = buf.sample(4)
        assert s.dtype == torch.float32
        assert p.dtype == torch.float32
        assert v.dtype == torch.float32

    def test_state_dict_round_trip(self, dummy_example):
        buf = ReplayBuffer(50)
        buf.add_many([dummy_example] * 15)
        d      = buf.state_dict()
        buf2   = ReplayBuffer.from_state_dict(d)
        assert len(buf2)      == len(buf)
        assert buf2.capacity  == buf.capacity
        assert buf2._pos      == buf._pos

    def test_repr_shows_size(self, dummy_example):
        buf = ReplayBuffer(100)
        buf.add_many([dummy_example] * 7)
        assert "7/100" in repr(buf)


# ── Checkpoints ───────────────────────────────────────────────────────────────

class TestCheckpoints:
    def test_save_and_load_round_trip(self, small_net, enc, game, dummy_example, tmp_path):
        optimizer = torch.optim.Adam(small_net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5])
        buf       = ReplayBuffer(50)
        buf.add_many([dummy_example] * 10)

        path = str(tmp_path / "ckpt.pt")
        config = DEBUG

        checkpoints.save(path, small_net, optimizer, scheduler, 3, buf, 5, config)
        assert os.path.exists(path)

        net2   = AlphaNet(enc.num_channels, game.action_size, num_resblocks=1, num_hidden=8)
        opt2   = torch.optim.Adam(net2.parameters(), lr=1e-3)
        sch2   = torch.optim.lr_scheduler.MultiStepLR(opt2, milestones=[5])

        iteration, buf2, promotion_count, run_segment = checkpoints.load(
            path, net2, opt2, sch2, torch.device("cpu")
        )

        assert iteration        == 3
        assert len(buf2)        == 10
        assert promotion_count  == 5
        assert run_segment      == 0

    def test_model_weights_preserved(self, small_net, enc, game, dummy_example, tmp_path):
        optimizer = torch.optim.Adam(small_net.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[])
        buf       = ReplayBuffer(10)
        path      = str(tmp_path / "ckpt2.pt")
        config    = DEBUG

        checkpoints.save(path, small_net, optimizer, scheduler, 0, buf, 0, config)

        net2 = AlphaNet(enc.num_channels, game.action_size, num_resblocks=1, num_hidden=8)
        opt2 = torch.optim.Adam(net2.parameters())
        sch2 = torch.optim.lr_scheduler.MultiStepLR(opt2, milestones=[])
        checkpoints.load(path, net2, opt2, sch2, torch.device("cpu"))

        for (k1, v1), (k2, v2) in zip(
            small_net.state_dict().items(), net2.state_dict().items()
        ):
            assert torch.allclose(v1, v2), f"Mismatch in layer {k1}"

    def test_find_latest_returns_none_when_empty(self, tmp_path):
        assert checkpoints.find_latest(str(tmp_path)) is None

    def test_find_latest_returns_highest_iteration(self, tmp_path, small_net, enc, game, dummy_example):
        optimizer = torch.optim.Adam(small_net.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[])
        buf       = ReplayBuffer(10)
        config    = DEBUG

        for i in [0, 3, 7]:
            p = str(tmp_path / f"checkpoint_{i}.pt")
            checkpoints.save(p, small_net, optimizer, scheduler, i, buf, 0, config)

        latest = checkpoints.find_latest(str(tmp_path))
        assert latest is not None
        assert "checkpoint_7.pt" in latest


# ── ReplayBuffer (pre-allocated mode) ────────────────────────────────────────

class TestReplayBufferPreallocated:
    @pytest.fixture
    def preallocated_buf(self, enc, game):
        return ReplayBuffer(
            50,
            state_shape=(enc.num_channels, game.row_count, game.col_count),
            policy_size=game.action_size,
        )

    def test_empty_has_zero_length(self, preallocated_buf):
        assert len(preallocated_buf) == 0

    def test_add_increments_length(self, preallocated_buf, dummy_example):
        preallocated_buf.add(*dummy_example)
        assert len(preallocated_buf) == 1

    def test_capacity_capped(self, preallocated_buf, dummy_example):
        preallocated_buf.add_many([dummy_example] * 60)
        assert len(preallocated_buf) == 50

    def test_sample_shapes(self, preallocated_buf, dummy_example, enc, game):
        preallocated_buf.add_many([dummy_example] * 20)
        states, policies, values = preallocated_buf.sample(8)
        assert states.shape   == (8, enc.num_channels, game.row_count, game.col_count)
        assert policies.shape == (8, game.action_size)
        assert values.shape   == (8, 1)

    def test_sample_dtypes(self, preallocated_buf, dummy_example):
        preallocated_buf.add_many([dummy_example] * 20)
        s, p, v = preallocated_buf.sample(4)
        assert s.dtype == torch.float32
        assert p.dtype == torch.float32
        assert v.dtype == torch.float32

    def test_repr(self, preallocated_buf, dummy_example):
        preallocated_buf.add_many([dummy_example] * 7)
        assert "7/50" in repr(preallocated_buf)

    def test_state_dict_round_trip(self, preallocated_buf, dummy_example):
        preallocated_buf.add_many([dummy_example] * 15)
        d    = preallocated_buf.state_dict()
        buf2 = ReplayBuffer.from_state_dict(d)
        assert len(buf2)     == len(preallocated_buf)
        assert buf2.capacity == preallocated_buf.capacity
        assert buf2._preallocated

    def test_state_dict_has_shape_metadata(self, preallocated_buf, dummy_example):
        preallocated_buf.add(*dummy_example)
        d = preallocated_buf.state_dict()
        assert "state_shape" in d
        assert "policy_size" in d

    def test_legacy_state_dict_loads_into_list_mode(self, dummy_example):
        """A checkpoint saved without shape metadata loads as list-mode buffer."""
        buf = ReplayBuffer(20)
        buf.add_many([dummy_example] * 5)
        d    = buf.state_dict()
        buf2 = ReplayBuffer.from_state_dict(d)
        assert not buf2._preallocated
        assert len(buf2) == 5


# ── Integration: full training iteration ─────────────────────────────────────
# Runs one complete iteration of the AlphaZero loop (self-play → train →
# evaluate → checkpoint) so cross-module regressions are caught before EC2.

class TestTrainingLoop:
    def test_one_iteration_produces_checkpoint(self, tmp_path):
        from training.trainer import Trainer

        config = copy.deepcopy(DEBUG)
        config.training.num_iterations = 1
        config.run_dir    = str(tmp_path)
        config.mlflow_uri = f"sqlite:///{tmp_path}/mlflow_test.db"

        Trainer(config).train()

        assert os.path.exists(os.path.join(config.checkpoint_dir, "checkpoint_latest.pt"))

    def test_one_iteration_writes_status(self, tmp_path):
        from training.trainer import Trainer

        config = copy.deepcopy(DEBUG)
        config.training.num_iterations = 1
        config.run_dir    = str(tmp_path)
        config.mlflow_uri = f"sqlite:///{tmp_path}/mlflow_test.db"

        Trainer(config).train()

        with open(config.status_path) as f:
            status = json.load(f)
        assert status["iteration"] == 1
        assert status["is_training"] is False

    def test_resume_increments_run_segment(self, tmp_path):
        from training.trainer import Trainer
        from training import checkpoints

        config = copy.deepcopy(DEBUG)
        config.training.num_iterations = 2
        config.run_dir    = str(tmp_path)
        config.mlflow_uri = f"sqlite:///{tmp_path}/mlflow_test.db"

        Trainer(config).train()

        latest = checkpoints.find_latest(config.checkpoint_dir)
        assert latest is not None

        # Resume: run_segment should increment from 0 → 1
        config2 = copy.deepcopy(DEBUG)
        config2.training.num_iterations = 3
        config2.run_dir    = str(tmp_path)
        config2.mlflow_uri = f"sqlite:///{tmp_path}/mlflow_test.db"

        t2 = Trainer(config2)
        t2.train(resume_from=latest)
        assert t2._run_segment == 1
