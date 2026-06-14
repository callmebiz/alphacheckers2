"""
Neural Network (AlphaNet)
=========================
A ResNet with two output heads, following the AlphaZero architecture.

What the network learns
-----------------------
Given a board position (as a tensor from StateEncoder), the network
simultaneously predicts two things:

  policy  — which moves look promising?
             Output: raw logits over all actions. After masking illegal
             moves and applying softmax these become move probabilities.

  value   — who is winning?
             Output: a single scalar in [-1, 1].
             +1 means the current player is very likely to win.
             -1 means the current player is very likely to lose.

Architecture
------------
    Input  (batch, channels, 8, 8)
      │
      ▼
    StartBlock  ── Conv2d(channels → hidden, 3×3) → BatchNorm → ReLU
      │
      ▼  (×num_resblocks)
    ResBlock    ── two conv layers with a skip connection
      │
      ├──▶  PolicyHead  ── Conv2d(→32) → BN → ReLU → Flatten → Linear(→action_size)
      │
      └──▶  ValueHead   ── Conv2d(→3)  → BN → ReLU → Flatten → Linear(→1) → Tanh

Why residual connections?
    Adding the input back to the output of each block (the "skip") lets
    gradients flow directly to early layers without vanishing, enabling
    much deeper networks to train stably.

Why BatchNorm?
    Normalises activations within each mini-batch, keeping them in a
    healthy range and significantly speeding up training convergence.

Why Tanh on the value head?
    Squashes the output to [-1, 1], which matches the game outcome range
    (+1 win, 0 draw, -1 loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    One residual block.

    Two 3×3 convolutions with a skip connection. Input and output share
    the same number of channels so the residual adds cleanly.

        x ──► Conv → BN → ReLU → Conv → BN ──► + ──► ReLU
        │                                        ▲
        └────────────────────────────────────────┘  (skip / residual)
    """

    def __init__(self, num_hidden: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class AlphaNet(nn.Module):
    """
    Shared-backbone ResNet with separate policy and value heads.

    Parameters
    ----------
    num_channels  : Input channels — must match StateEncoder.num_channels.
    action_size   : Total actions — must match Checkers.action_size.
    num_resblocks : Backbone depth. More blocks → stronger but slower to train.
                    Typical: 4 (debug), 6 (dev), 12 (production).
    num_hidden    : Convolutional width. More → more expressive.
                    Typical: 32 (debug), 64 (dev), 128 (production).
    board_h/w     : Board dimensions (default 8×8).
    """

    def __init__(
        self,
        num_channels: int,
        action_size: int,
        num_resblocks: int = 6,
        num_hidden: int = 64,
        board_h: int = 8,
        board_w: int = 8,
    ):
        super().__init__()
        self.start_block = nn.Sequential(
            nn.Conv2d(num_channels, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backbone = nn.Sequential(
            *[ResBlock(num_hidden) for _ in range(num_resblocks)]
        )

        # Where should I move?
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_h * board_w, action_size),
        )

        # Who is winning?
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * board_h * board_w, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, num_channels, 8, 8) float32 tensor.

        Returns
        -------
        policy_logits : (batch, action_size) — raw scores; mask then softmax for probs.
        value         : (batch, 1)           — expected outcome in [-1, 1].
        """
        features = self.backbone(self.start_block(x))
        return self.policy_head(features), self.value_head(features)
