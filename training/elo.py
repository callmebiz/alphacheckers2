"""
ELO Rating Tracker
==================
Maintains an ELO rating for every model checkpoint so you can see at a
glance how strength evolves across training iterations.

ELO in brief
------------
ELO is a method for rating players based on game outcomes. Two ratings R_A
and R_B produce an expected score for player A:

    E_A = 1 / (1 + 10^((R_B - R_A) / 400))

After each game, the rating is updated by:

    R_A_new = R_A + K * (S_A - E_A)

where S_A is the actual score (1 = win, 0.5 = draw, 0 = loss) and K is a
constant controlling how fast ratings change. A high K makes ratings react
quickly but noisily; a low K gives stable but slow-moving ratings.

The divisor 400 is a convention that makes a 400-point gap correspond to
roughly a 10× win probability — matching human chess rating intuition.

Usage in AlphaCheckers
----------------------
Each training iteration the model plays a tournament against the current best.
Results update both models' ELO scores. The 'best' checkpoint is the one with
the highest ELO, tracked persistently in a JSON file in the run directory.
"""

from __future__ import annotations

import json
import os


class EloTracker:
    """
    Maintains and persists ELO ratings keyed by checkpoint name.

    Ratings start at 1000 for any unseen checkpoint. Updates are applied
    game-by-game inside the tournament loop; call save() to persist to disk.

    Parameters
    ----------
    filepath : Path to the JSON file that stores ratings between runs.
    k        : ELO K-factor. Use ~32 for frequently-played, ~16 for stable.
    """

    START_RATING = 1000.0

    def __init__(self, filepath: str, k: float = 32.0):
        self.filepath = filepath
        self.k        = k
        self._ratings: dict[str, float] = {}
        self._load()

    # ── Rating access ─────────────────────────────────────────────────────────

    def rating(self, name: str) -> float:
        """Return the current ELO rating for *name* (1000 if unseen)."""
        return self._ratings.get(name, self.START_RATING)

    def all_ratings(self) -> dict[str, float]:
        """Return a copy of all stored ratings."""
        return dict(self._ratings)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, name_a: str, name_b: str, score_a: float) -> tuple[float, float]:
        """
        Apply one ELO update for a game between *name_a* and *name_b*.

        Parameters
        ----------
        name_a  : Identifier for model A (e.g. 'checkpoint_5').
        name_b  : Identifier for model B (e.g. 'best').
        score_a : Outcome from A's perspective — 1.0 win, 0.5 draw, 0.0 loss.

        Returns
        -------
        (new_rating_a, new_rating_b) after the update.
        """
        ra = self.rating(name_a)
        rb = self.rating(name_b)

        # Expected scores from the ELO formula
        ea = self._expected(ra, rb)
        eb = 1.0 - ea

        # Actual scores
        sa = score_a
        sb = 1.0 - score_a

        self._ratings[name_a] = ra + self.k * (sa - ea)
        self._ratings[name_b] = rb + self.k * (sb - eb)

        return self._ratings[name_a], self._ratings[name_b]

    def update_from_results(
        self,
        name_a: str,
        name_b: str,
        wins_a: int,
        draws: int,
        wins_b: int,
    ) -> None:
        """
        Apply ELO updates for a completed tournament (multiple games).

        Each game is processed individually so the rating converges correctly
        rather than being updated in one large batch.
        """
        for _ in range(wins_a):
            self.update(name_a, name_b, 1.0)
        for _ in range(draws):
            self.update(name_a, name_b, 0.5)
        for _ in range(wins_b):
            self.update(name_a, name_b, 0.0)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write current ratings to disk."""
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(self._ratings, f, indent=2)

    def _load(self) -> None:
        if os.path.exists(self.filepath):
            with open(self.filepath) as f:
                self._ratings = json.load(f)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _expected(ra: float, rb: float) -> float:
        """Expected score for player A against player B."""
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def __repr__(self) -> str:
        top = sorted(self._ratings.items(), key=lambda x: -x[1])[:5]
        lines = ", ".join(f"{n}={r:.0f}" for n, r in top)
        return f"EloTracker({lines})"
