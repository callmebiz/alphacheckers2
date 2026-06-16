from __future__ import annotations

import numpy as np
from collections import defaultdict

# Piece values
EMPTY = 0
P1_MAN = 1    # player 1 normal piece (moves up, row decreasing)
P2_MAN = -1   # player 2 normal piece (moves down, row increasing)
P1_KING = 2
P2_KING = -2

# All 8 directional deltas: 4 normal + 4 jump
_DELTAS = [(-1, -1), (-1, 1), (1, -1), (1, 1),
           (-2, -2), (-2, 2), (2, -2), (2, 2)]


class Checkers:
    """
    Standard 8x8 checkers with American rules:
    - Forced captures (must jump if able)
    - Multi-jump continuation
    - King promotion on back rank
    - Draw by repetition or move-without-progress limit
    """

    def __init__(
        self,
        row_count: int = 8,
        col_count: int = 8,
        buffer_rows: int = 2,
        draw_move_limit: int = 50,
        repetition_limit: int = 3,
        history_timesteps: int = 1,
    ):
        self.row_count = row_count
        self.col_count = col_count
        self.buffer_rows = buffer_rows
        self.draw_move_limit = draw_move_limit
        self.repetition_limit = repetition_limit
        self.history_timesteps = history_timesteps

        # Pre-compute move table: position -> list of (to_r, to_c, action_idx, dr, dc)
        self._move_table: dict[tuple[int, int], list[tuple]] = {}
        self._action_to_move: dict[int, tuple] = {}  # action_idx -> (from_r, from_c, to_r, to_c)
        self.action_size = self._build_move_table()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_move_table(self) -> int:
        idx = 0
        for r in range(self.row_count):
            for c in range(self.col_count):
                if (r + c) % 2 == 0:
                    continue  # only dark squares are playable
                entries = []
                for dr, dc in _DELTAS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.row_count and 0 <= nc < self.col_count:
                        entries.append((nr, nc, idx, dr, dc))
                        self._action_to_move[idx] = (r, c, nr, nc)
                        idx += 1
                self._move_table[(r, c)] = entries
        return idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_initial_state(self) -> dict:
        board = np.zeros((self.row_count, self.col_count), dtype=np.int8)
        piece_rows = (self.row_count - self.buffer_rows) // 2

        for r in range(piece_rows):
            for c in range(self.col_count):
                if (r + c) % 2 == 1:
                    board[r, c] = P2_MAN  # player 2 at top

        for r in range(self.row_count - piece_rows, self.row_count):
            for c in range(self.col_count):
                if (r + c) % 2 == 1:
                    board[r, c] = P1_MAN  # player 1 at bottom

        return {
            "board": board,
            "repetitions": defaultdict(int),
            "no_progress": 0,
            "jump_again": None,          # (r, c) if mid-multi-jump
            "history": [board.copy()],   # list of past boards for encoder
        }

    def get_valid_moves(self, state: dict, player: int) -> np.ndarray:
        """Return a binary array of shape (action_size,) with 1 for legal moves."""
        board = state["board"]
        jump_again = state["jump_again"]
        valid = np.zeros(self.action_size, dtype=np.int8)
        jump_indices: list[int] = []
        has_jump = False

        for (r, c), moves in self._move_table.items():
            piece = board[r, c]
            if not self._owned_by(piece, player):
                continue
            if jump_again is not None and (r, c) != jump_again:
                continue

            is_king = abs(piece) == 2

            for nr, nc, action_idx, dr, dc in moves:
                if abs(dr) == 2:  # jump
                    mid_r, mid_c = (r + nr) // 2, (c + nc) // 2
                    if (
                        self._owned_by(board[mid_r, mid_c], -player)
                        and board[nr, nc] == EMPTY
                        and (is_king or self._forward(player, dr))
                    ):
                        has_jump = True
                        jump_indices.append(action_idx)
                else:  # normal step
                    if (
                        jump_again is None
                        and board[nr, nc] == EMPTY
                        and (is_king or self._forward(player, dr))
                    ):
                        valid[action_idx] = 1

        if has_jump:
            valid[:] = 0
            for i in jump_indices:
                valid[i] = 1

        return valid

    def get_next_state(self, state: dict, action: int, player: int) -> dict:
        """Apply action and return new state. Does NOT switch player — caller handles that."""
        fr, fc, tr, tc = self._action_to_move[action]
        board = state["board"].copy()

        board[tr, tc] = board[fr, fc]
        board[fr, fc] = EMPTY

        capture_made = False
        new_jump_again = None

        if abs(tr - fr) == 2:  # capture
            mid_r, mid_c = (fr + tr) // 2, (fc + tc) // 2
            board[mid_r, mid_c] = EMPTY
            capture_made = True

            # Check if the same piece can jump again (multi-jump)
            probe = {**state, "board": board, "jump_again": (tr, tc)}
            if self.get_valid_moves(probe, player).any():
                new_jump_again = (tr, tc)

        # Promotion
        promotion_made = False
        if abs(board[tr, tc]) == 1:
            if (player == 1 and tr == 0) or (player == -1 and tr == self.row_count - 1):
                board[tr, tc] *= 2
                promotion_made = True
                # American rules: crowning via capture ends the turn immediately.
                # The newly-crowned king may not continue jumping in the same move.
                new_jump_again = None

        no_progress = 0 if (capture_made or promotion_made) else state["no_progress"] + 1

        reps = defaultdict(int, state["repetitions"])
        board_key = board.tobytes()
        reps[board_key] += 1

        history = list(state["history"])
        if new_jump_again is None:
            history.append(board.copy())
            if len(history) > self.history_timesteps:
                history = history[-self.history_timesteps:]

        return {
            "board": board,
            "repetitions": reps,
            "no_progress": no_progress,
            "jump_again": new_jump_again,
            "history": history,
        }

    def get_value_and_terminated(self, state: dict, player: int) -> tuple[float, bool]:
        """
        Returns (value, is_terminal) from `player`'s perspective.
        value: +1 win, -1 loss, 0 draw/ongoing.
        """
        if self._has_won(state, player):
            return 1.0, True
        if self._has_won(state, -player):
            return -1.0, True
        if self._is_draw(state):
            return 0.0, True
        return 0.0, False

    def get_opponent(self, player: int) -> int:
        return -player

    # ------------------------------------------------------------------
    # Board query helpers (used by UI / tests)
    # ------------------------------------------------------------------

    def list_moves(self, state: dict, player: int) -> list[dict]:
        """
        Return a list of move dicts for human / UI consumption.
        Each dict: {action, from_rc, to_rc, is_jump}
        """
        valid = self.get_valid_moves(state, player)
        moves = []
        for action_idx in np.where(valid)[0]:
            fr, fc, tr, tc = self._action_to_move[action_idx]
            moves.append({
                "action": int(action_idx),
                "from": (fr, fc),
                "to": (tr, tc),
                "is_jump": abs(tr - fr) == 2,
            })
        return moves

    def board_to_list(self, state: dict) -> list[list[int]]:
        """Return board as a plain Python list of lists (JSON-serialisable)."""
        return state["board"].tolist()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _owned_by(piece: int, player: int) -> bool:
        return piece != EMPTY and (piece == player or piece == 2 * player)

    @staticmethod
    def _forward(player: int, dr: int) -> bool:
        """A man can only move forward (row decreasing for P1, increasing for P2)."""
        return (player == 1 and dr < 0) or (player == -1 and dr > 0)

    def _has_won(self, state: dict, player: int) -> bool:
        board = state["board"]
        opp = -player
        # Opponent has no pieces
        if not np.any((board == opp) | (board == 2 * opp)):
            return True
        # Opponent has no legal moves
        opp_state = {**state, "jump_again": None}
        if not self.get_valid_moves(opp_state, opp).any():
            return True
        return False

    def _is_draw(self, state: dict) -> bool:
        if state["no_progress"] >= self.draw_move_limit:
            return True
        if any(v >= self.repetition_limit for v in state["repetitions"].values()):
            return True
        return False
