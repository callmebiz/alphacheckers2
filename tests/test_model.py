"""
Tests for core/model.py

Run: pytest tests/test_model.py -v
"""
import pytest
import torch

from core.game import Checkers
from core.encoder import StateEncoder
from core.model import AlphaNet, ResBlock


@pytest.fixture
def game():
    return Checkers()


@pytest.fixture
def enc(game):
    return StateEncoder(game)


@pytest.fixture
def net(enc, game):
    # Small config so tests run fast on CPU
    return AlphaNet(
        num_channels=enc.num_channels,
        action_size=game.action_size,
        num_resblocks=2,
        num_hidden=16,
    )


# ── Output shapes ─────────────────────────────────────────────────────────────

class TestOutputShapes:
    def test_policy_shape(self, net, enc, game):
        x = torch.zeros(1, enc.num_channels, 8, 8)
        policy, _ = net(x)
        assert policy.shape == (1, game.action_size)

    def test_value_shape(self, net, enc):
        x = torch.zeros(1, enc.num_channels, 8, 8)
        _, value = net(x)
        assert value.shape == (1, 1)

    def test_batch_dimension(self, net, enc, game):
        batch = 4
        x = torch.zeros(batch, enc.num_channels, 8, 8)
        policy, value = net(x)
        assert policy.shape == (batch, game.action_size)
        assert value.shape  == (batch, 1)


# ── Value range ───────────────────────────────────────────────────────────────

class TestValueRange:
    def test_value_in_minus1_to_1(self, net, enc):
        """Tanh output must always be in [-1, 1]."""
        x = torch.randn(8, enc.num_channels, 8, 8)
        _, value = net(x)
        assert value.min().item() >= -1.0 - 1e-6
        assert value.max().item() <=  1.0 + 1e-6

    def test_value_not_identically_zero(self, net, enc):
        """A randomly initialised network should not produce all-zero values."""
        x = torch.randn(4, enc.num_channels, 8, 8)
        _, value = net(x)
        assert value.abs().sum().item() > 0


# ── ResBlock ──────────────────────────────────────────────────────────────────

class TestResBlock:
    def test_shape_preserved(self):
        block = ResBlock(16)
        x = torch.randn(2, 16, 8, 8)
        assert block(x).shape == x.shape

    def test_skip_connection_active(self):
        """Output should differ from input (block is not identity by default)."""
        block = ResBlock(16)
        x = torch.randn(1, 16, 8, 8)
        assert not torch.allclose(block(x), x)


# ── Forward pass with real state ──────────────────────────────────────────────

class TestWithRealState:
    def test_forward_on_encoded_initial_state(self, net, enc, game):
        state  = game.get_initial_state()
        tensor = enc.encode(state, 1).unsqueeze(0)
        policy, value = net(tensor)
        assert policy.shape == (1, game.action_size)
        assert -1.0 <= value.item() <= 1.0
