"""
Tests for core/game.py — full American checkers rules.
Run with: pytest tests/test_game.py -v
"""
import numpy as np
import pytest
from collections import defaultdict
from core.game import Checkers, P1_MAN, P2_MAN, P1_KING, P2_KING, EMPTY


@pytest.fixture
def game():
    return Checkers()


def blank_state(game: Checkers) -> dict:
    """Empty board, no history noise."""
    board = np.zeros((game.row_count, game.col_count), dtype=np.int8)
    return {
        "board": board,
        "repetitions": defaultdict(int),
        "no_progress": 0,
        "jump_again": None,
        "history": [board.copy()],
    }


def place(state: dict, pieces: dict[tuple[int, int], int]) -> dict:
    """Set specific squares; returns the same state (mutates board in-place for test setup)."""
    for (r, c), val in pieces.items():
        state["board"][r, c] = val
    return state


# ── Initial state ──────────────────────────────────────────────────────────────

class TestInitialState:
    def test_piece_counts(self, game):
        state = game.get_initial_state()
        board = state["board"]
        assert np.sum(board == P1_MAN) == 12
        assert np.sum(board == P2_MAN) == 12
        assert np.sum(board == EMPTY) == 40

    def test_player1_at_bottom(self, game):
        state = game.get_initial_state()
        board = state["board"]
        # P1 occupies last 3 rows
        assert np.all(board[5:, :][board[5:, :] != EMPTY] == P1_MAN)

    def test_player2_at_top(self, game):
        state = game.get_initial_state()
        board = state["board"]
        assert np.all(board[:3, :][board[:3, :] != EMPTY] == P2_MAN)

    def test_only_dark_squares_occupied(self, game):
        state = game.get_initial_state()
        board = state["board"]
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 0:
                    assert board[r, c] == EMPTY

    def test_buffer_rows_empty(self, game):
        state = game.get_initial_state()
        board = state["board"]
        assert np.all(board[3:5, :] == EMPTY)

    def test_no_jump_again(self, game):
        assert game.get_initial_state()["jump_again"] is None


# ── Valid moves ────────────────────────────────────────────────────────────────

class TestValidMoves:
    def test_p1_has_moves_at_start(self, game):
        state = game.get_initial_state()
        moves = game.get_valid_moves(state, 1)
        assert moves.sum() == 7  # standard opening: 7 moves for P1

    def test_p2_has_moves_at_start(self, game):
        state = game.get_initial_state()
        moves = game.get_valid_moves(state, -1)
        assert moves.sum() == 7

    def test_man_cannot_move_backward(self, game):
        state = blank_state(game)
        place(state, {(4, 3): P1_MAN})
        moves = game.list_moves(state, 1)
        for m in moves:
            assert m["to"][0] < 4  # P1 men only move up (row decreases)

    def test_king_can_move_all_directions(self, game):
        state = blank_state(game)
        place(state, {(4, 3): P1_KING})
        moves = game.list_moves(state, 1)
        destinations = {m["to"] for m in moves}
        assert (3, 2) in destinations  # up-left
        assert (3, 4) in destinations  # up-right
        assert (5, 2) in destinations  # down-left
        assert (5, 4) in destinations  # down-right

    def test_forced_capture(self, game):
        """If a jump is available, only jump moves are legal."""
        state = blank_state(game)
        place(state, {(4, 3): P1_MAN, (3, 4): P2_MAN, (5, 2): EMPTY})
        moves = game.list_moves(state, 1)
        assert all(m["is_jump"] for m in moves)

    def test_no_moves_when_blocked(self, game):
        state = blank_state(game)
        # P1 man in corner with no forward squares
        place(state, {(0, 1): P1_MAN})
        moves = game.get_valid_moves(state, 1)
        assert moves.sum() == 0

    def test_cannot_jump_own_piece(self, game):
        state = blank_state(game)
        place(state, {(4, 3): P1_MAN, (3, 4): P1_MAN})
        moves = game.list_moves(state, 1)
        for m in moves:
            assert m["from"] != (4, 3) or m["to"] != (2, 5)


# ── Move application ───────────────────────────────────────────────────────────

class TestNextState:
    def _apply(self, game, state, player):
        moves = game.list_moves(state, player)
        assert moves, "No moves available"
        m = moves[0]
        return game.get_next_state(state, m["action"], player), m

    def test_piece_moves_to_destination(self, game):
        state = blank_state(game)
        place(state, {(5, 2): P1_MAN})
        m = game.list_moves(state, 1)[0]
        new_state = game.get_next_state(state, m["action"], 1)
        assert new_state["board"][m["to"]] == P1_MAN
        assert new_state["board"][m["from"]] == EMPTY

    def test_capture_removes_opponent_piece(self, game):
        state = blank_state(game)
        place(state, {(4, 3): P1_MAN, (3, 4): P2_MAN})
        moves = [m for m in game.list_moves(state, 1) if m["is_jump"]]
        assert moves
        new_state = game.get_next_state(state, moves[0]["action"], 1)
        assert new_state["board"][3, 4] == EMPTY

    def test_promotion_p1(self, game):
        state = blank_state(game)
        place(state, {(1, 2): P1_MAN})
        moves = [m for m in game.list_moves(state, 1) if m["to"][0] == 0]
        assert moves
        new_state = game.get_next_state(state, moves[0]["action"], 1)
        r, c = moves[0]["to"]
        assert new_state["board"][r, c] == P1_KING

    def test_promotion_p2(self, game):
        state = blank_state(game)
        place(state, {(6, 3): P2_MAN})
        moves = [m for m in game.list_moves(state, -1) if m["to"][0] == 7]
        assert moves
        new_state = game.get_next_state(state, moves[0]["action"], -1)
        r, c = moves[0]["to"]
        assert new_state["board"][r, c] == P2_KING

    def test_no_progress_increments_on_normal_move(self, game):
        state = blank_state(game)
        place(state, {(5, 2): P1_MAN})
        m = game.list_moves(state, 1)[0]
        new_state = game.get_next_state(state, m["action"], 1)
        assert new_state["no_progress"] == 1

    def test_no_progress_resets_on_capture(self, game):
        state = blank_state(game)
        state["no_progress"] = 10
        place(state, {(4, 3): P1_MAN, (3, 4): P2_MAN})
        jump = [m for m in game.list_moves(state, 1) if m["is_jump"]][0]
        new_state = game.get_next_state(state, jump["action"], 1)
        assert new_state["no_progress"] == 0

    def test_no_progress_resets_on_promotion(self, game):
        state = blank_state(game)
        state["no_progress"] = 10
        place(state, {(1, 2): P1_MAN})
        moves = [m for m in game.list_moves(state, 1) if m["to"][0] == 0]
        new_state = game.get_next_state(state, moves[0]["action"], 1)
        assert new_state["no_progress"] == 0


# ── Multi-jump ─────────────────────────────────────────────────────────────────

class TestMultiJump:
    def test_jump_again_set_when_further_jump_available(self, game):
        """
        P1 man at (4,1), P2 men at (3,2) and (1,4).
        After jumping (3,2) -> lands at (2,3), can jump (1,4) -> (0,5).
        """
        state = blank_state(game)
        place(state, {(4, 1): P1_MAN, (3, 2): P2_MAN, (1, 4): P2_MAN})
        jump1 = [m for m in game.list_moves(state, 1) if m["is_jump"]]
        assert jump1
        after1 = game.get_next_state(state, jump1[0]["action"], 1)
        assert after1["jump_again"] == (2, 3)

    def test_jump_again_forces_same_piece(self, game):
        state = blank_state(game)
        place(state, {(4, 1): P1_MAN, (3, 2): P2_MAN, (1, 4): P2_MAN})
        jump1 = [m for m in game.list_moves(state, 1) if m["is_jump"]][0]
        after1 = game.get_next_state(state, jump1["action"], 1)
        moves2 = game.list_moves(after1, 1)
        assert all(m["from"] == after1["jump_again"] for m in moves2)

    def test_jump_again_cleared_when_no_further_jump(self, game):
        state = blank_state(game)
        place(state, {(4, 3): P1_MAN, (3, 4): P2_MAN})
        jump = [m for m in game.list_moves(state, 1) if m["is_jump"]][0]
        new_state = game.get_next_state(state, jump["action"], 1)
        assert new_state["jump_again"] is None


# ── Terminal conditions ────────────────────────────────────────────────────────

class TestTerminal:
    def test_win_when_opponent_has_no_pieces(self, game):
        state = blank_state(game)
        place(state, {(4, 3): P1_MAN})
        value, done = game.get_value_and_terminated(state, 1)
        assert done and value == 1.0

    def test_loss_from_opponents_perspective(self, game):
        state = blank_state(game)
        place(state, {(4, 3): P1_MAN})
        value, done = game.get_value_and_terminated(state, -1)
        assert done and value == -1.0

    def test_draw_by_no_progress(self, game):
        state = blank_state(game)
        place(state, {(4, 3): P1_MAN, (2, 3): P2_MAN})
        state["no_progress"] = game.draw_move_limit
        _, done = game.get_value_and_terminated(state, 1)
        assert done

    def test_draw_by_repetition(self, game):
        state = blank_state(game)
        place(state, {(4, 3): P1_MAN, (2, 3): P2_MAN})
        board_key = state["board"].tobytes()
        state["repetitions"][board_key] = game.repetition_limit
        value, done = game.get_value_and_terminated(state, 1)
        assert done and value == 0.0

    def test_win_when_opponent_has_no_moves(self, game):
        """P2 man stuck in bottom-right corner — only forward square blocked, jumps go out of bounds."""
        state = blank_state(game)
        # P2 man at (6,7): forward moves go to (7,6) blocked and (7,8) OOB;
        # jump landings at (8,5) and (8,9) are both OOB → P2 has zero legal moves.
        place(state, {(6, 7): P2_MAN, (7, 6): P1_MAN})
        value, done = game.get_value_and_terminated(state, 1)
        assert done and value == 1.0

    def test_ongoing_game_is_not_terminal(self, game):
        state = game.get_initial_state()
        _, done = game.get_value_and_terminated(state, 1)
        assert not done


# ── Symmetry / player perspective ─────────────────────────────────────────────

class TestSymmetry:
    def test_opponent_helper(self, game):
        assert game.get_opponent(1) == -1
        assert game.get_opponent(-1) == 1

    def test_both_players_have_equal_opening_moves(self, game):
        state = game.get_initial_state()
        assert game.get_valid_moves(state, 1).sum() == game.get_valid_moves(state, -1).sum()


# ── Action encoding round-trip ─────────────────────────────────────────────────

class TestActionEncoding:
    def test_action_size_is_positive(self, game):
        assert game.action_size > 0

    def test_all_actions_map_to_board_squares(self, game):
        for idx, (fr, fc, tr, tc) in game._action_to_move.items():
            assert 0 <= fr < game.row_count
            assert 0 <= fc < game.col_count
            assert 0 <= tr < game.row_count
            assert 0 <= tc < game.col_count

    def test_list_moves_actions_are_valid_indices(self, game):
        state = game.get_initial_state()
        for m in game.list_moves(state, 1):
            assert 0 <= m["action"] < game.action_size

    def test_board_to_list_shape(self, game):
        state = game.get_initial_state()
        grid = game.board_to_list(state)
        assert len(grid) == 8
        assert all(len(row) == 8 for row in grid)
