"""
Tests for core/encoder.py

Run: pytest tests/test_encoder.py -v
"""
import numpy as np
import pytest
import torch
from collections import defaultdict

from core.game import Checkers, P1_MAN, P1_KING, P2_MAN, P2_KING, EMPTY
from core.encoder import StateEncoder, PLANES_PER_STEP, EXTRA_PLANES


@pytest.fixture
def game():
    return Checkers()


@pytest.fixture
def enc(game):
    return StateEncoder(game)


def blank_state(game):
    board = np.zeros((game.row_count, game.col_count), dtype=np.int8)
    return {
        "board": board,
        "repetitions": defaultdict(int),
        "no_progress": 0,
        "jump_again": None,
        "history": [board.copy()],
    }


# ── Shape ─────────────────────────────────────────────────────────────────────

class TestShape:
    def test_output_is_tensor(self, enc, game):
        state = game.get_initial_state()
        out = enc.encode(state, 1)
        assert isinstance(out, torch.Tensor)

    def test_output_shape(self, enc, game):
        state = game.get_initial_state()
        out = enc.encode(state, 1)
        assert out.shape == (enc.num_channels, 8, 8)

    def test_num_channels_formula(self, game, enc):
        expected = PLANES_PER_STEP * game.history_timesteps + EXTRA_PLANES
        assert enc.num_channels == expected

    def test_dtype_is_float32(self, enc, game):
        state = game.get_initial_state()
        assert enc.encode(state, 1).dtype == torch.float32


# ── Piece planes ──────────────────────────────────────────────────────────────

class TestPiecePlanes:
    def test_p1_perspective_my_men_plane(self, enc, game):
        """P1 men appear in 'my men' channel (0) when encoding for player 1."""
        state = blank_state(game)
        state['board'][5, 2] = P1_MAN
        state['history'][-1] = state['board'].copy()
        out = enc.encode(state, 1)
        # Most-recent timestep is last in history → last block of channels
        t = game.history_timesteps - 1
        assert out[t * PLANES_PER_STEP + 0, 5, 2] == 1.0  # my men
        assert out[t * PLANES_PER_STEP + 2, 5, 2] == 0.0  # opp men

    def test_p2_perspective_my_men_plane(self, enc, game):
        """P2 men appear in 'my men' channel (0) when encoding for player -1."""
        state = blank_state(game)
        state['board'][2, 3] = P2_MAN
        state['history'][-1] = state['board'].copy()
        out = enc.encode(state, -1)
        t = game.history_timesteps - 1
        # Board is rotated 180° for player -1; (2,3) → (5,4)
        assert out[t * PLANES_PER_STEP + 0, 5, 4] == 1.0

    def test_p1_king_in_king_plane(self, enc, game):
        state = blank_state(game)
        state['board'][3, 2] = P1_KING
        state['history'][-1] = state['board'].copy()
        out = enc.encode(state, 1)
        t = game.history_timesteps - 1
        assert out[t * PLANES_PER_STEP + 1, 3, 2] == 1.0  # my kings

    def test_opponent_pieces_in_opp_channels(self, enc, game):
        state = blank_state(game)
        state['board'][2, 1] = P2_MAN
        state['history'][-1] = state['board'].copy()
        out = enc.encode(state, 1)
        t = game.history_timesteps - 1
        assert out[t * PLANES_PER_STEP + 2, 2, 1] == 1.0  # opp men
        assert out[t * PLANES_PER_STEP + 0, 2, 1] == 0.0  # not my men

    def test_initial_state_has_nonzero_pieces(self, enc, game):
        state = game.get_initial_state()
        out = enc.encode(state, 1)
        assert out.sum() > 0


# ── Scalar planes ─────────────────────────────────────────────────────────────

class TestScalarPlanes:
    def _base(self, game):
        return PLANES_PER_STEP * game.history_timesteps

    def test_colour_plane_p1(self, enc, game):
        state = game.get_initial_state()
        out = enc.encode(state, 1)
        assert out[self._base(game) + 2].unique().tolist() == [1.0]

    def test_colour_plane_p2(self, enc, game):
        state = game.get_initial_state()
        out = enc.encode(state, -1)
        assert out[self._base(game) + 2].unique().tolist() == [0.0]

    def test_no_progress_ratio_zero_at_start(self, enc, game):
        state = game.get_initial_state()
        out = enc.encode(state, 1)
        assert float(out[self._base(game) + 1, 0, 0]) == pytest.approx(0.0)

    def test_no_progress_ratio_halfway(self, enc, game):
        state = game.get_initial_state()
        state['no_progress'] = game.draw_move_limit // 2
        out = enc.encode(state, 1)
        assert float(out[self._base(game) + 1, 0, 0]) == pytest.approx(0.5)

    def test_scalar_planes_are_uniform(self, enc, game):
        """Every scalar plane must have the same value in all 64 squares."""
        state = game.get_initial_state()
        out = enc.encode(state, 1)
        for i in range(EXTRA_PLANES):
            plane = out[self._base(game) + i]
            assert plane.min() == plane.max()


# ── Perspective symmetry ──────────────────────────────────────────────────────

class TestPerspective:
    def test_both_perspectives_sum_same_piece_count(self, enc, game):
        """Total piece count should not change with perspective flip."""
        state = game.get_initial_state()
        out1 = enc.encode(state, 1)
        out2 = enc.encode(state, -1)
        t = game.history_timesteps - 1
        ch = t * PLANES_PER_STEP
        pieces1 = out1[ch:ch + 4].sum()
        pieces2 = out2[ch:ch + 4].sum()
        assert float(pieces1) == pytest.approx(float(pieces2))
