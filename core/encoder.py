"""
State Encoder
=============
Converts a Checkers game state dict into a float32 tensor the neural network
can read.

Why encode at all?
------------------
The neural network needs a fixed-size numerical input. Raw board values
(-2, -1, 0, 1, 2) aren't ideal — binary planes (one channel per piece type)
are easier for a convolutional network to learn from.

Perspective normalisation
-------------------------
The encoding is always produced from the *current player's* point of view.
Player 1's pieces always appear in the "my pieces" channels and player -1's
pieces always appear in the "opponent" channels, regardless of who is
actually moving. This means the network never needs to learn two separate
strategies for each colour — it always reasons as if it is "the attacker"
moving up the board.

For player -1 we rotate the board 180° so their back rank is at the top,
matching how player 1 sees the board.

Tensor layout  (shape: num_channels × 8 × 8)
---------------------------------------------
Channels 0 … 4·T−1   — board history (T timesteps, oldest first):
    ch + 0  my men       (binary 0/1)
    ch + 1  my kings     (binary 0/1)
    ch + 2  opp men      (binary 0/1)
    ch + 3  opp kings    (binary 0/1)

Channels 4·T … 4·T+2 — scalar planes (same value broadcast over all 64 sq):
    4·T + 0  repetition ratio   = #repeats of current board / repetition_limit
    4·T + 1  no-progress ratio  = no_progress_moves / draw_move_limit
    4·T + 2  colour plane       = 1.0 if player == 1 else 0.0
"""

from __future__ import annotations

import numpy as np
import torch

from core.game import Checkers, P1_MAN, P1_KING, P2_MAN, P2_KING

PLANES_PER_STEP = 4  # my_men, my_kings, opp_men, opp_kings
EXTRA_PLANES    = 3  # repetition, no-progress, colour


class StateEncoder:
    """Encodes a game state into a tensor for the neural network."""

    def __init__(self, game: Checkers):
        self.game         = game
        self.num_channels = PLANES_PER_STEP * game.history_timesteps + EXTRA_PLANES

    def encode(self, state: dict, player: int) -> torch.Tensor:
        """
        Encode *state* from *player*'s point of view.

        Parameters
        ----------
        state  : Game state dict (from Checkers.get_initial_state / get_next_state).
        player : +1 or -1 — whose perspective to encode from.

        Returns
        -------
        torch.Tensor of shape (num_channels, 8, 8), dtype=float32.
        """
        H, W = self.game.row_count, self.game.col_count
        T    = self.game.history_timesteps
        out  = torch.zeros(self.num_channels, H, W, dtype=torch.float32)

        # Piece ordering depends on whose perspective we encode.
        # For player -1 we also rotate the board 180° so their back rank
        # sits at the top, matching how player 1 experiences the board.
        if player == 1:
            pieces = np.array([P1_MAN, P1_KING, P2_MAN, P2_KING], dtype=np.int8)
        else:
            pieces = np.array([P2_MAN, P2_KING, P1_MAN, P1_KING], dtype=np.int8)

        # ── Historical board planes ────────────────────────────────────────
        history = state['history']  # list of past boards, most recent last
        for t in range(T):
            # If history is shorter than T, repeat the oldest frame
            board = history[max(0, len(history) - T + t)]
            if player == -1:
                board = board[::-1, ::-1]  # 180° view, no copy
            ch = t * PLANES_PER_STEP

            # Single broadcast comparison: (4,1,1) vs (H,W) → (4,H,W) bool
            planes = (
                (board[np.newaxis] == pieces[:, np.newaxis, np.newaxis])
                .astype(np.float32)
            )
            out[ch:ch + PLANES_PER_STEP] = torch.from_numpy(planes)

        # ── Scalar planes ──────────────────────────────────────────────────
        base      = T * PLANES_PER_STEP
        board_key = state['board'].tobytes()
        reps      = state['repetitions'][board_key]  # defaultdict → 0 if absent

        out[base + 0] = reps / self.game.repetition_limit
        out[base + 1] = state['no_progress'] / self.game.draw_move_limit
        out[base + 2] = 1.0 if player == 1 else 0.0

        return out


